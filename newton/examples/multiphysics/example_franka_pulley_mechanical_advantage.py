# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Franka Pulley Mechanical Advantage
#
# A MuJoCo-simulated Franka follows an IK trajectory that pulls the free end
# of a VBD cable. Upper and lower multi-sheave crane blocks lift a guided
# weight with a configurable mechanical advantage. The hand approaches and
# closes on a thin box attached directly to the cable end. SolverCoupledADMM
# transfers contact force between the MuJoCo gripper and the VBD mechanism
# without a fixed robot-cable attachment.
#
# The cable is constructed in a straight structural rest pose, then its state
# is initialized on an explicitly pre-wrapped route. Validation checks that its
# load motion matches the configured mechanical advantage.
#
# Command: python -m newton.examples franka_pulley_mechanical_advantage
#
###########################################################################

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM, SolverCoupledProxy

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton.solvers import SolverMuJoCo, SolverVBD

DEFAULT_MECHANICAL_ADVANTAGE = 4
DEFAULT_WEIGHT_MASS = 5.0
END_WEIGHT_MASS = 0.2
GRAVITY = 9.81
FRANKA_BASE_X = 1.3

CABLE_RADIUS = 0.004
CABLE_SEGMENT_LENGTH = 0.020
CABLE_STRETCH_STIFFNESS = None  # 1.0e5
CABLE_STRETCH_DAMPING = None  # 1.0e-4
CABLE_BEND_STIFFNESS = None  # 1.0e-2
CABLE_BEND_DAMPING = None  # 5.0e-5
CABLE_PULLEY_FRICTION = 0.1
CABLE_CONTACT_GAP = 0.5 * CABLE_RADIUS
END_WEIGHT_INITIAL_OFFSET = 0.020
END_BOX_HALF_X = 0.025
END_BOX_HALF_Y = 0.025
END_BOX_HALF_Z = 0.030
GRIPPER_FRICTION = 8.0
GRIPPER_CONTACT_STIFFNESS = 2.0e4
GRIPPER_CONTACT_DAMPING = 50.0
GRIPPER_HEIGHT_CORRECTION_FILTER = 0.2
GRIPPER_LATERAL_CORRECTION_MAX = 0.03
GRIPPER_HEIGHT_CORRECTION_MAX = 0.06

PULLEY_RADIUS = 0.045
PULLEY_WRAP_CLEARANCE = 1.25 * CABLE_RADIUS
PULLEY_WRAP_RADIUS = PULLEY_RADIUS + PULLEY_WRAP_CLEARANCE
PULLEY_FIXED_Z = 1.10
PULLEY_MOVING_Z = 0.72
PULLEY_BLOCK_X = 0.50
PULLEY_LEAD_X = PULLEY_BLOCK_X + 3.0 * PULLEY_WRAP_RADIUS
PULL_X = PULLEY_LEAD_X + PULLEY_WRAP_RADIUS
PULLEY_SHEAVE_SPACING = 5.5 * CABLE_RADIUS

PULL_START_Z = 0.48
PULL_DISTANCE = 0.4
GRIPPER_APPROACH_ANGLE = math.pi / 6.0
GRIPPER_GRASP_POSITION = (0.555, 0.0, 0.437)

INITIAL_HOLD_DURATION = 0.0
APPROACH_DURATION = 1.2
SIDE_APPROACH_DURATION = 0.8
GRASP_DURATION = 1.0
PULL_DURATION = 8.0
FINAL_HOLD_DURATION = 2.0
GRASP_END_TIME = INITIAL_HOLD_DURATION + APPROACH_DURATION + SIDE_APPROACH_DURATION + GRASP_DURATION
PULL_END_TIME = GRASP_END_TIME + PULL_DURATION
EXAMPLE_DURATION = PULL_END_TIME + FINAL_HOLD_DURATION

GRIPPER_APPROACH_ORIENTATION = (
    0.0,
    -math.cos(GRIPPER_APPROACH_ANGLE),
    0.0,
    math.sin(GRIPPER_APPROACH_ANGLE),
)
GRIP_OPEN = 0.040
GRIP_CLOSE = 0.007

# Raised-arm starting point for the IK solve (7 arm + 2 finger coordinates).
FRANKA_Q = [
    0.0,
    -0.569,
    0.0,
    -2.810,
    0.0,
    3.037,
    0.741,
    GRIP_OPEN,
    GRIP_OPEN,
]


@wp.kernel
def set_task_target(
    target_positions: wp.array[wp.vec3],
    target_rotations: wp.array[wp.vec4],
    finger_position: wp.array[float],
    position: wp.vec3,
    rotation: wp.vec4,
    grip_width: float,
):
    """Set the single-world IK target without reallocating device arrays."""
    target_positions[0] = position
    target_rotations[0] = rotation
    finger_position[0] = grip_width


@wp.kernel
def set_gripper_target(joint_q: wp.array2d[float], finger_position: wp.array[float], index_0: int, index_1: int):
    """Set both Franka finger coordinates to the commanded half-width."""
    joint_q[0, index_0] = finger_position[0]
    joint_q[0, index_1] = finger_position[0]


def _capture_frame_graph(model: newton.Model, simulate: Callable[[], None], *, enabled: bool):
    if not enabled or not model.device.is_cuda:
        return None

    with wp.ScopedDevice(model.device), wp.ScopedCapture() as capture:
        simulate()
    return capture.graph


def _launch_frame_graph(model: newton.Model, graph) -> bool:
    if graph is None:
        return False

    with wp.ScopedDevice(model.device):
        wp.capture_launch(graph)
    return True


def _find_label_index(labels: list[str], suffix: str) -> int:
    for index, label in enumerate(labels):
        if label.endswith(suffix):
            return index
    raise ValueError(f"Could not find label ending in {suffix!r}")


def _append_route_point(points: list[wp.vec3], point: wp.vec3) -> None:
    if not points or float(wp.length(point - points[-1])) > 1.0e-8:
        points.append(point)


def _append_arc_xz(
    points: list[wp.vec3],
    center: wp.vec3,
    radius: float,
    start_angle: float,
    end_angle: float,
    segment_length: float,
    *,
    direction: str,
) -> None:
    """Append a pulley arc in the vertical XZ plane."""
    delta = (end_angle - start_angle + math.pi) % (2.0 * math.pi) - math.pi
    if direction == "cw" and delta > 0.0:
        delta -= 2.0 * math.pi
    elif direction == "ccw" and delta < 0.0:
        delta += 2.0 * math.pi

    count = max(3, int(math.ceil(abs(delta) * radius / segment_length)))
    for i in range(count + 1):
        angle = start_angle + delta * float(i) / float(count)
        _append_route_point(
            points,
            wp.vec3(
                float(center[0]) + radius * math.cos(angle),
                float(center[1]),
                float(center[2]) + radius * math.sin(angle),
            ),
        )


def _resample_equal_length_segments(route_points: list[wp.vec3], segment_length: float) -> tuple[list[wp.vec3], float]:
    """Resample a polyline route into equal-length cable segments."""
    points = [route_points[0]]
    distances = [0.0]
    total_length = 0.0
    for route_point in route_points[1:]:
        length = float(wp.length(route_point - points[-1]))
        if length <= 1.0e-8:
            continue
        total_length += length
        points.append(route_point)
        distances.append(total_length)

    segment_count = max(2, int(math.ceil(total_length / segment_length)))
    equal_length = total_length / float(segment_count)
    resampled = [points[0]]
    point_index = 1
    for segment_index in range(1, segment_count):
        target_distance = equal_length * float(segment_index)
        while point_index < len(points) - 1 and distances[point_index] < target_distance:
            point_index += 1

        previous_distance = distances[point_index - 1]
        next_distance = distances[point_index]
        alpha = (target_distance - previous_distance) / (next_distance - previous_distance)
        resampled.append(points[point_index - 1] * (1.0 - alpha) + points[point_index] * alpha)

    resampled.append(points[-1])
    return resampled, equal_length


def create_block_and_tackle_cable_points(
    moving_centers: list[wp.vec3],
    fixed_centers: list[wp.vec3],
    lead_center: wp.vec3,
    pull_end: wp.vec3,
    segment_length: float,
) -> tuple[list[wp.vec3], float]:
    """Create the pre-wrapped route through upper and lower crane blocks."""
    # Keep the dead end below the upper block so it does not touch an unused sheave.
    start = wp.vec3(
        float(moving_centers[0][0]) + PULLEY_WRAP_RADIUS,
        float(moving_centers[0][1]),
        float(fixed_centers[0][2]) - 2.0 * PULLEY_WRAP_RADIUS,
    )
    points = [start]

    for index, (moving_center, fixed_center) in enumerate(zip(moving_centers, fixed_centers, strict=True)):
        _append_arc_xz(
            points,
            moving_center,
            PULLEY_WRAP_RADIUS,
            0.0,
            -math.pi,
            segment_length,
            direction="cw",
        )
        _append_arc_xz(
            points,
            fixed_center,
            PULLEY_WRAP_RADIUS,
            math.pi,
            0.0 if index < len(fixed_centers) - 1 else 0.5 * math.pi,
            segment_length,
            direction="cw",
        )

    _append_arc_xz(
        points,
        lead_center,
        PULLEY_WRAP_RADIUS,
        0.5 * math.pi,
        0.0,
        segment_length,
        direction="cw",
    )
    _append_route_point(points, pull_end)
    return _resample_equal_length_segments(points, segment_length)


def _filter_body_group_collisions(builder: newton.ModelBuilder, bodies: list[int]) -> None:
    """Disable cable self-collision while preserving cable-pulley contact."""
    for i, body_a in enumerate(bodies):
        for body_b in bodies[i + 1 :]:
            for shape_a in builder.body_shapes.get(body_a, []):
                for shape_b in builder.body_shapes.get(body_b, []):
                    builder.add_shape_collision_filter_pair(int(shape_a), int(shape_b))


def _add_visual_box(
    builder: newton.ModelBuilder,
    *,
    body: int,
    center: wp.vec3,
    half_extents: tuple[float, float, float],
    color: tuple[float, float, float],
    label: str,
    density: float = 0.0,
    collision: bool = False,
) -> int:
    cfg = newton.ModelBuilder.ShapeConfig(
        density=density,
        ke=1.0e5,
        kd=20.0,
        mu=0.8,
        has_shape_collision=collision,
        has_particle_collision=collision,
    )
    return builder.add_shape_box(
        body=body,
        xform=wp.transform(center, wp.quat_identity()),
        hx=half_extents[0],
        hy=half_extents[1],
        hz=half_extents[2],
        cfg=cfg,
        color=color,
        label=label,
    )


def _add_pulley_stack_mount(
    builder: newton.ModelBuilder,
    *,
    body: int,
    axle_center: wp.vec3,
    mount_z: float,
    stack_half_y: float,
    support_color: tuple[float, float, float],
    label: str,
) -> None:
    """Add a shared axle and two bearing supports to a pulley stack."""
    axle_radius = 2.0 * CABLE_RADIUS
    bearing_half_y = 2.0 * CABLE_RADIUS
    bearing_center_y = stack_half_y + bearing_half_y + 0.002
    axle_half_height = stack_half_y + 2.0 * bearing_half_y
    axle_x = float(axle_center[0])
    axle_y = float(axle_center[1])
    axle_z = float(axle_center[2])
    align_cylinder_to_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi)
    axle_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=False,
        has_particle_collision=False,
    )
    builder.add_shape_cylinder(
        body=body,
        xform=wp.transform(axle_center, align_cylinder_to_y),
        radius=axle_radius,
        half_height=axle_half_height + 0.003,
        cfg=axle_cfg,
        color=(0.42, 0.45, 0.48),
        label=f"{label}_axle",
    )

    support_center_z = 0.5 * (axle_z + mount_z)
    support_half_z = 0.5 * abs(mount_z - axle_z) + axle_radius
    for side, suffix in ((-1.0, "neg"), (1.0, "pos")):
        _add_visual_box(
            builder,
            body=body,
            center=wp.vec3(axle_x, axle_y + side * bearing_center_y, support_center_z),
            half_extents=(0.025, bearing_half_y, support_half_z),
            color=support_color,
            label=f"{label}_bearing_{suffix}",
        )


def add_pulley(
    builder: newton.ModelBuilder,
    *,
    center: wp.vec3,
    parent: int,
    parent_origin: wp.vec3,
    color: tuple[float, float, float],
    label: str,
    density: float,
) -> tuple[int, int]:
    """Add a passive grooved pulley adapted from cable_cross_slide_table."""
    body = builder.add_link(
        xform=wp.transform(center, wp.quat_identity()),
        label=f"{label}_body",
    )
    joint = builder.add_joint_revolute(
        parent=parent,
        child=body,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(center - parent_origin, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        armature=1.0e-4,
        friction=0.0,
        label=f"{label}_axle",
    )

    # Match the cross-slide sample's close-fitting groove so the cable cannot
    # shuttle along the pulley axle before contacting a flange.
    groove_half_width = 1.55 * CABLE_RADIUS
    flange_half_thickness = 0.6 * CABLE_RADIUS
    flange_radius = PULLEY_RADIUS + 3.2 * CABLE_RADIUS
    align_cylinder_to_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi)
    sheave_cfg = newton.ModelBuilder.ShapeConfig(
        density=density,
        ke=1.0e5,
        kd=0.0,
        mu=CABLE_PULLEY_FRICTION,
    )
    flange_cfg = newton.ModelBuilder.ShapeConfig(
        density=density,
        ke=1.0e5,
        kd=0.0,
        mu=0.5 * CABLE_PULLEY_FRICTION,
    )

    for suffix, y, radius, half_height, cfg, shade in (
        ("sheave", 0.0, PULLEY_RADIUS, groove_half_width, sheave_cfg, color),
        (
            "flange_neg",
            -(groove_half_width + flange_half_thickness),
            flange_radius,
            flange_half_thickness,
            flange_cfg,
            tuple(0.68 * component for component in color),
        ),
        (
            "flange_pos",
            groove_half_width + flange_half_thickness,
            flange_radius,
            flange_half_thickness,
            flange_cfg,
            tuple(0.68 * component for component in color),
        ),
    ):
        builder.add_shape_cylinder(
            body=body,
            xform=wp.transform(wp.vec3(0.0, y, 0.0), align_cylinder_to_y),
            radius=radius,
            half_height=half_height,
            cfg=cfg,
            color=shade,
            label=f"{label}_{suffix}",
        )

    return body, joint


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, int(args.substeps))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.use_graph = bool(args.graph_capture)
        self.coupling_solver = str(args.coupling_solver)
        self.mechanical_advantage = int(args.mechanical_advantage)
        if self.mechanical_advantage < 2 or self.mechanical_advantage % 2 != 0:
            raise ValueError("Mechanical advantage must be an even integer of at least 4")
        self.weight_mass = float(args.weight_mass)
        if self.weight_mass <= 0.0:
            raise ValueError("Weight mass must be positive")

        self.pull_target_origin: np.ndarray | None = None
        self.latest_robot_downward_force = 0.0
        self.gripper_lateral_correction = np.zeros(2, dtype=np.float64)
        self.gripper_height_correction = 0.0
        self.last_commanded_tcp_z = 0.0

        self._build_scene()
        self._build_keyframes()
        self.control = self.model.control()
        self._build_ik()
        self._build_solver(args)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        initial_cable_xforms = wp.array(
            self.initial_cable_xforms,
            dtype=wp.transform,
            device=self.device,
        )
        wp.copy(
            self.state_0.body_q,
            initial_cable_xforms,
            dest_offset=self.cable_bodies[0],
            count=len(self.cable_bodies),
        )
        wp.copy(
            self.state_1.body_q,
            initial_cable_xforms,
            dest_offset=self.cable_bodies[0],
            count=len(self.cable_bodies),
        )
        initial_body_q = self.state_0.body_q.numpy()
        self.box_target_center = self._end_box_center(initial_body_q)
        if not np.allclose(self.box_target_center[:2], self.pull_line_xy, atol=1.0e-5):
            raise ValueError("Graspable box is not initialized on the vertical free-rope line")
        self.gripper_pad_midpoint = self._gripper_pad_midpoint(initial_body_q)
        self.gripper_tcp_position = self._gripper_tcp_position(initial_body_q)
        self.last_commanded_tcp_z = float(self.targets[0, 2])
        self.initial_load_z = float(initial_body_q[self.weight_body, 2])
        self.solver.sync_entry_states(self.state_0)

        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.solver.prepare_contacts(self.contacts)

        newton.examples.configure_coupled_view(self, args)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.set_camera(pos=wp.vec3(1.35, -2.1, 1.25), pitch=-10.0, yaw=145.0)
            if hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.45, 0.0, 0.62))

        self.graph = _capture_frame_graph(self.model, self.simulate, enabled=self.use_graph)

    @staticmethod
    def _add_franka(builder: newton.ModelBuilder) -> None:
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(
                wp.vec3(FRANKA_BASE_X, 0.0, 0.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi),
            ),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        builder.joint_q[: len(FRANKA_Q)] = FRANKA_Q
        builder.joint_target_q[: len(FRANKA_Q)] = FRANKA_Q

        for finger_label in ("fr3_leftfinger", "fr3_rightfinger"):
            finger_body = _find_label_index(builder.body_label, finger_label)
            rubber_pad_shape = int(builder.body_shapes[finger_body][-1])
            builder.shape_label[rubber_pad_shape] = f"{finger_label}_rubber_contact_pad"

    def _build_scene(self) -> None:
        builder = newton.ModelBuilder(gravity=-GRAVITY)
        builder.rigid_gap = CABLE_CONTACT_GAP
        SolverMuJoCo.register_custom_attributes(builder)
        SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)

        franka_body_start = builder.body_count
        franka_joint_start = builder.joint_count
        franka_shape_start = builder.shape_count
        self._add_franka(builder)
        builder.joint_target_ke[:7] = [900.0] * 7
        builder.joint_target_kd[:7] = [90.0] * 7
        builder.joint_target_ke[7:9] = [5000.0, 5000.0]
        builder.joint_target_kd[7:9] = [200.0, 200.0]
        builder.joint_effort_limit[:7] = [80.0] * 7
        builder.joint_effort_limit[7:9] = [1000.0, 1000.0]
        builder.joint_armature[:7] = [0.05] * 7
        self.franka_bodies = list(range(franka_body_start, builder.body_count))
        self.franka_joints = list(range(franka_joint_start, builder.joint_count))
        self.franka_shapes = list(range(franka_shape_start, builder.shape_count))
        self.hand_body = _find_label_index(builder.body_label, "fr3_hand")

        vbd_body_start = builder.body_count
        vbd_shape_start = builder.shape_count
        vbd_joints: list[int] = []

        pulley_pair_count = self.mechanical_advantage // 2
        pulley_sheave_y = tuple(
            (index - 0.5 * (pulley_pair_count - 1)) * PULLEY_SHEAVE_SPACING for index in range(pulley_pair_count)
        )
        pulley_block_half_y = 0.5 * pulley_pair_count * PULLEY_SHEAVE_SPACING
        pull_y = pulley_sheave_y[-1]
        self.pull_line_xy = np.array([PULL_X, pull_y], dtype=np.float64)

        frame_color = (0.16, 0.22, 0.30)
        top_beam_center_z = PULLEY_FIXED_Z + 0.09
        top_beam_half_z = 0.025
        _add_visual_box(
            builder,
            body=-1,
            center=wp.vec3(0.5 * (PULLEY_BLOCK_X + PULL_X), 0.0, top_beam_center_z),
            half_extents=(
                0.5 * (PULL_X - PULLEY_BLOCK_X) + 0.08,
                pulley_block_half_y + 0.03,
                top_beam_half_z,
            ),
            color=frame_color,
            label="pulley_frame_top_beam",
        )

        load_center = wp.vec3(PULLEY_BLOCK_X, 0.0, 0.49)
        load_half_x = 0.085
        weight_half_y = max(0.06, pulley_block_half_y + 0.02)
        _add_visual_box(
            builder,
            body=-1,
            center=wp.vec3(float(load_center[0]), 0.0, 0.39),
            half_extents=(0.14, weight_half_y + 0.06, 0.01),
            color=(0.24, 0.27, 0.30),
            label="weight_floor",
            collision=True,
        )
        self.weight_body = builder.add_link(
            xform=wp.transform(load_center, wp.quat_identity()),
            label="load_weight_body",
        )
        weight_half_extents = (load_half_x, weight_half_y, 0.09)
        weight_volume = 8.0 * math.prod(weight_half_extents)
        _add_visual_box(
            builder,
            body=self.weight_body,
            center=wp.vec3(0.0),
            half_extents=weight_half_extents,
            density=(self.weight_mass + END_WEIGHT_MASS * self.mechanical_advantage) / weight_volume,
            collision=True,
            color=(0.72, 0.16, 0.12),
            label="load_weight",
        )
        load_slide = builder.add_joint_prismatic(
            parent=-1,
            child=self.weight_body,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(load_center, wp.quat_identity()),
            child_xform=wp.transform_identity(),
            target_kd=20.0,
            label="weight_vertical_guide",
        )
        vbd_joints.append(load_slide)

        moving_color = (0.88, 0.54, 0.12)
        fixed_color = (0.12, 0.38, 0.78)
        moving_centers = [wp.vec3(PULLEY_BLOCK_X, y, PULLEY_MOVING_Z) for y in pulley_sheave_y]
        fixed_centers = [wp.vec3(PULLEY_BLOCK_X, y, PULLEY_FIXED_Z) for y in pulley_sheave_y]
        lead_center = wp.vec3(PULLEY_LEAD_X, pull_y, PULLEY_FIXED_Z)

        _add_pulley_stack_mount(
            builder,
            body=-1,
            axle_center=wp.vec3(PULLEY_BLOCK_X, 0.0, PULLEY_FIXED_Z),
            mount_z=top_beam_center_z - top_beam_half_z,
            stack_half_y=pulley_block_half_y,
            support_color=frame_color,
            label="fixed_pulley_stack",
        )
        _add_pulley_stack_mount(
            builder,
            body=-1,
            axle_center=lead_center,
            mount_z=top_beam_center_z - top_beam_half_z,
            stack_half_y=0.5 * PULLEY_SHEAVE_SPACING,
            support_color=frame_color,
            label="lead_pulley_mount",
        )

        _add_pulley_stack_mount(
            builder,
            body=self.weight_body,
            axle_center=wp.vec3(0.0, 0.0, PULLEY_MOVING_Z - float(load_center[2])),
            mount_z=weight_half_extents[2],
            stack_half_y=pulley_block_half_y,
            support_color=(0.50, 0.18, 0.12),
            label="moving_pulley_stack",
        )

        moving_bodies = []
        moving_joints = []
        for index, center in enumerate(moving_centers):
            body, joint = add_pulley(
                builder,
                center=center,
                parent=self.weight_body,
                parent_origin=load_center,
                color=moving_color,
                label=f"moving_pulley_{index}",
                density=1.0e-1,
            )
            moving_bodies.append(body)
            moving_joints.append(joint)
            vbd_joints.append(joint)

        fixed_bodies = []
        fixed_joints = []
        for index, center in enumerate(fixed_centers):
            body, joint = add_pulley(
                builder,
                center=center,
                parent=-1,
                parent_origin=wp.vec3(0.0),
                color=fixed_color,
                label=f"fixed_pulley_{index}",
                density=1.0e-1,
            )
            fixed_bodies.append(body)
            fixed_joints.append(joint)
            vbd_joints.append(joint)

        lead_body, lead_joint = add_pulley(
            builder,
            center=lead_center,
            parent=-1,
            parent_origin=wp.vec3(0.0),
            color=fixed_color,
            label="lead_pulley",
            density=1.0e-1,
        )
        fixed_bodies.append(lead_body)
        vbd_joints.append(lead_joint)

        _filter_body_group_collisions(builder, moving_bodies)
        _filter_body_group_collisions(builder, fixed_bodies)
        builder.add_articulation([load_slide, *moving_joints], label="moving_pulley_block")
        for index, joint in enumerate(fixed_joints):
            builder.add_articulation([joint], label=f"fixed_pulley_{index}_articulation")
        builder.add_articulation([lead_joint], label="lead_pulley_articulation")

        pull_end = wp.vec3(PULL_X, pull_y, PULL_START_Z + END_WEIGHT_INITIAL_OFFSET)
        cable_points, route_segment_length = create_block_and_tackle_cable_points(
            moving_centers,
            fixed_centers,
            lead_center,
            pull_end,
            CABLE_SEGMENT_LENGTH,
        )
        cable_quats = newton.utils.create_parallel_transport_cable_quaternions(cable_points)
        cable_segment_count = len(cable_points) - 1
        self.cable_segment_length = route_segment_length
        straight_points, straight_quats = newton.utils.create_straight_cable_points_and_quaternions(
            start=cable_points[0],
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=cable_segment_count * route_segment_length,
            num_segments=cable_segment_count,
        )
        cable_cfg = newton.ModelBuilder.ShapeConfig(
            density=10.0,
            ke=1.0e5,
            kd=0.0,
            mu=CABLE_PULLEY_FRICTION,
            gap=CABLE_CONTACT_GAP,
        )
        self.cable_bodies, self.cable_joints = builder.add_rod(
            positions=straight_points,
            quaternions=straight_quats,
            radius=CABLE_RADIUS,
            body_frame_origin="com",
            cfg=cable_cfg,
            stretch_stiffness=CABLE_STRETCH_STIFFNESS,
            stretch_damping=CABLE_STRETCH_DAMPING,
            bend_stiffness=CABLE_BEND_STIFFNESS,
            bend_damping=CABLE_BEND_DAMPING,
            wrap_in_articulation=False,  # articulation is created manually later with anchor ball joint
            color=(0.82, 0.72, 0.46),
            label="block_and_tackle_cable",
        )
        self.initial_cable_xforms = [
            wp.transform(cable_points[i] + 0.5 * (cable_points[i + 1] - cable_points[i]), cable_quats[i])
            for i in range(len(self.cable_bodies))
        ]
        _filter_body_group_collisions(builder, self.cable_bodies)
        vbd_joints.extend(self.cable_joints)

        endpoint = 0.5 * self.cable_segment_length
        self.end_box_center_offset = endpoint + END_BOX_HALF_Z
        first_endpoint_xform = wp.transform(wp.vec3(0.0, 0.0, -endpoint), wp.quat_identity())
        anchor_joint = builder.add_joint_ball(
            parent=-1,
            child=self.cable_bodies[0],
            parent_xform=wp.transform(cable_points[0], wp.quat_identity()),
            child_xform=first_endpoint_xform,
            label="fixed_cable_anchor",
        )
        vbd_joints.append(anchor_joint)

        end_weight_volume = 8.0 * END_BOX_HALF_X * END_BOX_HALF_Y * END_BOX_HALF_Z
        end_weight_cfg = newton.ModelBuilder.ShapeConfig(
            density=END_WEIGHT_MASS / end_weight_volume,
            ke=GRIPPER_CONTACT_STIFFNESS,
            kd=GRIPPER_CONTACT_DAMPING,
            mu=GRIPPER_FRICTION,
            gap=0.002,
        )
        builder.add_shape_box(
            body=self.cable_bodies[-1],
            xform=wp.transform(
                wp.vec3(0.0, 0.0, self.end_box_center_offset),
                wp.quat_identity(),
            ),
            hx=END_BOX_HALF_X,
            hy=END_BOX_HALF_Y,
            hz=END_BOX_HALF_Z,
            cfg=end_weight_cfg,
            color=(0.96, 0.78, 0.26),
            label="graspable_end_weight",
        )
        builder.add_articulation(
            [*self.cable_joints, anchor_joint],
            label="block_and_tackle_cable",
        )

        anchor_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        builder.add_shape_sphere(
            body=-1,
            xform=wp.transform(cable_points[0], wp.quat_identity()),
            radius=2.0 * CABLE_RADIUS,
            cfg=anchor_cfg,
            color=(0.86, 0.82, 0.70),
            label="fixed_cable_anchor_marker",
        )

        self.vbd_bodies = list(range(vbd_body_start, builder.body_count))
        self.vbd_joints = vbd_joints
        self.vbd_shapes = list(range(vbd_shape_start, builder.shape_count))

        self.gripper_bodies = [body for body in self.franka_bodies if "finger" in builder.body_label[body]]
        self.gripper_contact_shapes = {
            shape for shape in self.franka_shapes if builder.shape_label[shape].endswith("_contact_pad")
        }
        for franka_shape in self.franka_shapes:
            if franka_shape in self.gripper_contact_shapes:
                continue
            for vbd_shape in self.vbd_shapes:
                builder.add_shape_collision_filter_pair(franka_shape, vbd_shape)

        builder.color(balance_colors=False)
        self.model = builder.finalize()
        shape_body = self.model.shape_body.numpy()
        shape_transform = self.model.shape_transform.numpy()
        self.gripper_pad_local_centers = tuple(
            (
                int(shape_body[shape]),
                np.asarray(shape_transform[shape, :3], dtype=np.float64),
            )
            for shape in sorted(self.gripper_contact_shapes)
        )
        self.device = self.model.device

        gripper_contact_shapes = list(self.gripper_contact_shapes)
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke[gripper_contact_shapes] = GRIPPER_CONTACT_STIFFNESS
        shape_kd[gripper_contact_shapes] = GRIPPER_CONTACT_DAMPING
        shape_mu[gripper_contact_shapes] = GRIPPER_FRICTION
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)
        self.model.shape_material_mu.assign(shape_mu)

    def _build_solver(self, args) -> None:
        entries = [
            SolverCoupled.Entry(
                name="mjc",
                solver=lambda view: SolverMuJoCo(
                    model=view,
                    solver="newton",
                    integrator="implicitfast",
                    iterations=int(args.mujoco_iterations),
                    ls_iterations=int(args.mujoco_ls_iterations),
                    use_mujoco_contacts=False,
                    njmax=256,
                    nconmax=64,
                ),
                bodies=self.franka_bodies,
                joints=self.franka_joints,
                shapes=self.franka_shapes,
            ),
            SolverCoupled.Entry(
                name="vbd",
                solver=lambda view: SolverVBD(
                    model=view,
                    iterations=int(args.vbd_iterations),
                    rigid_body_contact_buffer_size=512,
                    rigid_contact_hard=False,
                    rigid_contact_history=False,
                ),
                bodies=self.vbd_bodies,
                joints=self.vbd_joints,
                shapes=self.vbd_shapes,
            ),
        ]
        if self.coupling_solver == "admm":
            self.solver = SolverCoupledADMM(
                model=self.model,
                entries=entries,
                coupling=SolverCoupledADMM.Config(
                    iterations=int(args.admm_iterations),
                    rho=float(args.rho),
                    gamma=float(args.gamma),
                    baumgarte=float(args.baumgarte),
                    rigid_contact_matching=str(args.rigid_contact_matching),
                    contact_matching_force_scale=float(args.contact_matching_force_scale),
                    contact_pairs=[SolverCoupledADMM.ContactPair(source="mjc", destination="vbd")],
                ),
            )
        else:
            self.solver = SolverCoupledProxy(
                model=self.model,
                entries=entries,
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="mjc",
                            destination="vbd",
                            bodies=self.gripper_bodies,
                            mass_scale=float(args.mass_scale),
                            mode=str(args.coupling_mode),
                            collision_pipeline=lambda model: newton.examples.create_collision_pipeline(
                                model,
                                broad_phase="explicit",
                            ),
                            collide_interval=1,
                        )
                    ],
                    iterations=int(args.proxy_iterations),
                ),
            )
        self.mujoco_solver = self.solver.solver("mjc")
        mujoco_view = self.solver.view("mjc")
        finger_local_bodies = {i for i, label in enumerate(mujoco_view.body_label) if "finger" in label}
        mujoco_body_to_newton = self.mujoco_solver.mjc_body_to_newton.numpy()[0]
        self.mujoco_finger_bodies = [
            i for i, body in enumerate(mujoco_body_to_newton) if int(body) in finger_local_bodies
        ]
        if len(self.mujoco_finger_bodies) != 2:
            raise ValueError("Could not map both Franka finger bodies into MuJoCo")

        self.robot_proximal_gravity_compensation = 0.0
        if self.coupling_solver == "admm":
            global_finger_by_label = {self.model.body_label[body]: body for body in self.gripper_bodies}
            local_mass = mujoco_view.body_mass.numpy()
            global_mass = self.model.body_mass.numpy()
            local_gravity = wp.empty(mujoco_view.body_count, dtype=wp.vec3, device=self.device)
            self.mujoco_solver.coupling_eval_gravity_acceleration(local_gravity, None)
            local_gravity = local_gravity.numpy()

            for local_body in finger_local_bodies:
                label = mujoco_view.body_label[local_body]
                if label not in global_finger_by_label:
                    raise ValueError(f"Could not map MuJoCo finger body {label!r} into the full model")
                global_body = global_finger_by_label[label]
                proximal_mass = max(0.0, float(local_mass[local_body] - global_mass[global_body]))
                self.robot_proximal_gravity_compensation -= proximal_mass * float(local_gravity[local_body, 2])

    def _build_keyframes(self) -> None:
        start_position, start_rotation = self._initial_tcp_pose()
        approach_rotation = np.asarray(GRIPPER_APPROACH_ORIENTATION, dtype=np.float32)
        if float(np.dot(start_rotation, approach_rotation)) < 0.0:
            start_rotation = -start_rotation

        grasp_position = np.asarray(GRIPPER_GRASP_POSITION, dtype=np.float32)
        approach_position = grasp_position + np.array(
            [0.14, 0.0, 0.14 * math.tan(GRIPPER_APPROACH_ANGLE)],
            dtype=np.float32,
        )
        pull_position = grasp_position - np.array([0.0, 0.0, PULL_DISTANCE], dtype=np.float32)

        def keyframe(duration: float, position: np.ndarray, grip: float) -> list[float]:
            return [duration, *position, *approach_rotation, grip]

        poses = np.array(
            [
                [INITIAL_HOLD_DURATION, *start_position.tolist(), *start_rotation.tolist(), GRIP_OPEN],
                keyframe(APPROACH_DURATION, approach_position, GRIP_OPEN),
                keyframe(SIDE_APPROACH_DURATION, grasp_position, GRIP_OPEN),
                keyframe(GRASP_DURATION, grasp_position, GRIP_CLOSE),
                keyframe(PULL_DURATION, pull_position, GRIP_CLOSE),
                keyframe(FINAL_HOLD_DURATION, pull_position, GRIP_CLOSE),
            ],
            dtype=np.float32,
        )
        self.targets = poses[:, 1:]
        self.key_times = np.cumsum(poses[:, 0])
        self.nominal_grasp_position = grasp_position
        self.pull_keyframe_index = len(poses) - 2

    def _initial_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
        hand_q = state.body_q.numpy()[self.hand_body]
        position = self._transform_point(hand_q, np.array([0.0, 0.0, 0.107], dtype=np.float64))
        rotation = np.asarray(hand_q[3:7], dtype=np.float32)
        return position.astype(np.float32), rotation

    def _build_ik(self) -> None:
        ik_builder = newton.ModelBuilder(gravity=-GRAVITY)
        self._add_franka(ik_builder)
        self.ik_model = ik_builder.finalize(device=self.device)
        self.n_coords = self.ik_model.joint_coord_count
        self.ik_joint_q = wp.clone(self.model.joint_q.reshape((1, -1))[:, : self.n_coords])
        self.control_joint_target_q = self.control.joint_target_q.reshape((1, -1))
        self.finger_index_0 = self.n_coords - 2
        self.finger_index_1 = self.n_coords - 1
        self.finger_position = wp.full(1, GRIP_OPEN, dtype=float, device=self.device)

        target = self.targets[0]
        self.ik_target_positions = wp.array([wp.vec3(*target[:3].tolist())], dtype=wp.vec3, device=self.device)
        self.ik_target_rotations = wp.array([wp.vec4(*target[3:7].tolist())], dtype=wp.vec4, device=self.device)
        ik_hand_body = _find_label_index(self.ik_model.body_label, "fr3_hand")
        position_objective = ik.IKObjectivePosition(
            link_index=ik_hand_body,
            link_offset=wp.vec3(0.0, 0.0, 0.107),
            target_positions=self.ik_target_positions,
        )
        rotation_objective = ik.IKObjectiveRotation(
            link_index=ik_hand_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=self.ik_target_rotations,
        )
        joint_limits_objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=wp.clone(self.model.joint_limit_lower[: self.n_coords]),
            joint_limit_upper=wp.clone(self.model.joint_limit_upper[: self.n_coords]),
            weight=10.0,
        )
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=[position_objective, rotation_objective, joint_limits_objective],
            lambda_initial=0.05,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 24

    def update_ik_target(self) -> None:
        t = min(self.sim_time, float(self.key_times[-1]) - 1.0e-6)
        interval = int(np.searchsorted(self.key_times, t))
        start_time = self.key_times[interval - 1] if interval > 0 else 0.0
        end_time = self.key_times[interval]
        alpha = float(np.clip((t - start_time) / max(end_time - start_time, 1.0e-6), 0.0, 1.0))
        current = self.targets[interval]
        previous = self.targets[interval - 1] if interval > 0 else current
        target = (1.0 - alpha) * previous + alpha * current

        if interval > 0:
            if interval >= self.pull_keyframe_index:
                if self.pull_target_origin is None:
                    # Box motion must not feed back into the commanded pull distance.
                    self.pull_target_origin = self.box_target_center.copy()
                target_center = self.pull_target_origin
            else:
                target_center = self.box_target_center
            target_center = target_center.copy()
            target_center[:2] = self.pull_line_xy

            center_offset = target_center - self.nominal_grasp_position
            current_position = current[:3] + center_offset
            if interval == 1:
                previous_position = previous[:3]
            else:
                previous_position = previous[:3] + center_offset
            target[:3] = (1.0 - alpha) * previous_position + alpha * current_position

            if interval < self.pull_keyframe_index:
                measured_lateral_offset = self.gripper_tcp_position[:2] - self.gripper_pad_midpoint[:2]
                self.gripper_lateral_correction = np.clip(
                    (1.0 - GRIPPER_HEIGHT_CORRECTION_FILTER) * self.gripper_lateral_correction
                    + GRIPPER_HEIGHT_CORRECTION_FILTER * measured_lateral_offset,
                    -GRIPPER_LATERAL_CORRECTION_MAX,
                    GRIPPER_LATERAL_CORRECTION_MAX,
                )
                measured_height_offset = self.last_commanded_tcp_z - float(self.gripper_pad_midpoint[2])
                self.gripper_height_correction = float(
                    np.clip(
                        (1.0 - GRIPPER_HEIGHT_CORRECTION_FILTER) * self.gripper_height_correction
                        + GRIPPER_HEIGHT_CORRECTION_FILTER * measured_height_offset,
                        0.0,
                        GRIPPER_HEIGHT_CORRECTION_MAX,
                    )
                )
            target[:2] += self.gripper_lateral_correction
            target[2] += self.gripper_height_correction

        self.last_commanded_tcp_z = float(target[2])

        wp.launch(
            set_task_target,
            dim=1,
            inputs=[
                self.ik_target_positions,
                self.ik_target_rotations,
                self.finger_position,
                wp.vec3(*target[:3].tolist()),
                wp.vec4(*target[3:7].tolist()),
                float(target[-1]),
            ],
            device=self.device,
        )

    def simulate(self) -> None:
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            set_gripper_target,
            dim=1,
            inputs=[self.ik_joint_q, self.finger_position, self.finger_index_0, self.finger_index_1],
            device=self.device,
        )
        wp.copy(dest=self.control_joint_target_q[:, : self.n_coords], src=self.ik_joint_q)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.model.collide(self.state_0, self.contacts, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

    @staticmethod
    def _transform_point(pose: np.ndarray, local_point: np.ndarray) -> np.ndarray:
        rotation = wp.quat(float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6]))
        rotated = wp.quat_rotate(
            rotation,
            wp.vec3(float(local_point[0]), float(local_point[1]), float(local_point[2])),
        )
        return np.array(
            [
                float(pose[0]) + float(rotated[0]),
                float(pose[1]) + float(rotated[1]),
                float(pose[2]) + float(rotated[2]),
            ],
            dtype=np.float64,
        )

    def _end_box_center(self, body_q: np.ndarray) -> np.ndarray:
        return self._transform_point(
            body_q[self.cable_bodies[-1]],
            np.array([0.0, 0.0, self.end_box_center_offset], dtype=np.float64),
        )

    def _gripper_pad_midpoint(self, body_q: np.ndarray) -> np.ndarray:
        return np.mean(
            [
                self._transform_point(body_q[body], local_center)
                for body, local_center in self.gripper_pad_local_centers
            ],
            axis=0,
        )

    def _gripper_tcp_position(self, body_q: np.ndarray) -> np.ndarray:
        return self._transform_point(body_q[self.hand_body], np.array([0.0, 0.0, 0.107], dtype=np.float64))

    def _measure_robot_downward_force(self) -> float:
        if self.mujoco_solver.use_mujoco_cpu:
            applied_wrenches = np.asarray(self.mujoco_solver.mj_data.xfrc_applied)
        else:
            applied_wrenches = self.mujoco_solver.mjw_data.xfrc_applied.numpy()[0]
        upward_reaction = float(np.sum(applied_wrenches[self.mujoco_finger_bodies, 2]))
        physical_reaction = upward_reaction - self.robot_proximal_gravity_compensation
        return max(0.0, physical_reaction)

    def _record_diagnostics(self) -> None:
        body_q = self.state_0.body_q.numpy()
        self.latest_robot_downward_force = self._measure_robot_downward_force()
        load_z = float(body_q[self.weight_body, 2])
        self.box_target_center = self._end_box_center(body_q)
        self.gripper_pad_midpoint = self._gripper_pad_midpoint(body_q)
        self.gripper_tcp_position = self._gripper_tcp_position(body_q)

        force_ratio = (
            self.weight_mass * GRAVITY / self.latest_robot_downward_force
            if self.latest_robot_downward_force > 1.0e-3
            else 0.0
        )
        self.viewer.log_scalar("Robot downward force [N]", self.latest_robot_downward_force, smoothing=10)
        self.viewer.log_scalar("Measured force advantage", force_ratio, smoothing=10)
        self.viewer.log_scalar("Gripper height correction [m]", self.gripper_height_correction, smoothing=10)
        self.viewer.log_scalar("Grasped box height [m]", float(self.box_target_center[2]), smoothing=10)
        self.viewer.log_scalar("Weight height [m]", load_z, smoothing=10)

    def step(self) -> None:
        self.update_ik_target()
        if not _launch_frame_graph(self.model, self.graph):
            self.simulate()
        self.sim_time += self.frame_dt
        self._record_diagnostics()

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    def test_post_step(self) -> None:
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        if not np.all(np.isfinite(body_q)) or not np.all(np.isfinite(body_qd)):
            raise ValueError("Pulley mechanism contains NaN or inf body state")

    def test_final(self) -> None:
        body_q = self.state_0.body_q.numpy()
        final_load_z = float(body_q[self.weight_body, 2])
        load_lift = final_load_z - self.initial_load_z
        expected_load_lift = PULL_DISTANCE / self.mechanical_advantage

        if not math.isclose(load_lift, expected_load_lift, rel_tol=0.4):
            raise ValueError(
                f"The {self.weight_mass:.0f} kg weight lifted only {load_lift:.3f} m; "
                f"expected approximately {expected_load_lift:.3f} m for a "
                f"{self.mechanical_advantage}:1 mechanical advantage"
            )
        if self.use_graph and self.device.is_cuda and self.graph is None:
            raise ValueError("CUDA graph capture was requested but no graph was captured")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        parser.add_argument(
            "--mechanical-advantage",
            type=int,
            default=DEFAULT_MECHANICAL_ADVANTAGE,
            help="Even supporting-strand count (minimum 2).",
        )
        parser.add_argument(
            "--weight-mass",
            type=float,
            default=DEFAULT_WEIGHT_MASS,
            help="Intended lifted mass before cable-end compensation [kg].",
        )
        parser.add_argument("--substeps", type=int, default=16, help="Coupled substeps per rendered frame.")
        parser.add_argument(
            "--coupling-solver",
            choices=["admm", "proxy"],
            default="admm",
            help="Cross-solver contact coupling method.",
        )
        parser.add_argument("--admm-iterations", type=int, default=1, help="ADMM iterations per coupled substep.")
        parser.add_argument("--rho", type=float, default=200.0, help="ADMM penalty parameter.")
        parser.add_argument("--gamma", type=float, default=0.001, help="ADMM proximal metric scale.")
        parser.add_argument("--baumgarte", type=float, default=0.5, help="ADMM position error correction fraction.")
        parser.add_argument(
            "--rigid-contact-matching",
            choices=["disabled", "latest", "sticky"],
            default="sticky",
            help="ADMM gripper contact matching mode.",
        )
        parser.add_argument(
            "--contact-matching-force-scale",
            type=float,
            default=1.0,
            help="Multiplier for matched previous-step ADMM contact-force warm starts.",
        )
        parser.add_argument(
            "--proxy-iterations", type=int, default=1, help="Proxy relaxation passes per coupled substep."
        )
        parser.add_argument(
            "--mass-scale",
            type=float,
            default=1.0,
            help="Scale applied to the gripper effective mass in the VBD proxy solve.",
        )
        parser.add_argument(
            "--coupling-mode",
            choices=["lagged", "staggered"],
            default="lagged",
            help="Proxy transfer mode.",
        )
        parser.add_argument("--vbd-iterations", type=int, default=10, help="VBD iterations per coupled substep.")
        parser.add_argument("--mujoco-iterations", type=int, default=100, help="MuJoCo solver iterations.")
        parser.add_argument("--mujoco-ls-iterations", type=int, default=50, help="MuJoCo line-search iterations.")
        parser.add_argument(
            "--no-graph-capture",
            action="store_false",
            dest="graph_capture",
            default=True,
            help="Disable CUDA graph capture.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    parser.set_defaults(num_frames=int(math.ceil(EXAMPLE_DURATION * 60.0)))
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
