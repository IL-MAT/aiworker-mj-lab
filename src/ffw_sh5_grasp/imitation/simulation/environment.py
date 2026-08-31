"""Arm-only ALOHA-style environment backed by the existing MuJoCo controllers."""

import hashlib
from pathlib import Path

import mujoco
import numpy as np

from ...config import SETTINGS
from ...control import arm, grasp
from ...paths import MODEL_PATH
from .action import ARM_JOINTS, SIDES, ActionAdapter
from .cameras import MujocoCameraManager
from .state import PolicyStateAdapter
from .task import create_task


def _required_id(model, kind, name):
    object_id = mujoco.mj_name2id(model, kind, name)
    if object_id < 0:
        raise ValueError(f"required MuJoCo object is missing: {name}")
    return object_id


def enable_task_collisions(model, bin_body_names):
    """Enable selected task fixtures and task-relevant right-hand contacts."""
    active_geom_ids = []
    for body_name in bin_body_names:
        body_id = _required_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        bin_geom_ids = np.flatnonzero(
            (model.geom_bodyid == body_id) & (model.geom_contype != 0)
        ).astype(int)
        if bin_geom_ids.size == 0:
            raise ValueError(f"{body_name} must contain collision geometry")
        model.geom_contype[bin_geom_ids] = 1
        model.geom_conaffinity[bin_geom_ids] = 1
        model.body_contype[body_id] = 1
        model.body_conaffinity[body_id] = 1
        active_geom_ids.extend(bin_geom_ids.tolist())

    disabled_signature = np.iinfo(model.exclude_signature.dtype).max
    for index in range(13, 21):
        finger_body = _required_id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"finger_r_link{index}"
        )
        matches = np.flatnonzero(model.exclude_signature == finger_body)
        if matches.size == 0:
            continue
        if matches.size != 1:
            raise ValueError(f"expected one world exclusion for finger_r_link{index}")
        model.exclude_signature[matches[0]] = disabled_signature
    model.exclude_signature.sort()
    return np.asarray(active_geom_ids, dtype=int)


class AIWorkerMujocoEnv:
    """Execute 16D absolute joint targets without whole-body IK.

    Base, lift and head are held at their home references. Only the two arm
    torque controllers and the existing hand-synergy controller receive policy
    commands. Robot qpos is never overwritten during :meth:`step`. A standalone
    instance creates its own model/data; the main teleop can instead attach its
    existing model/data so policy inference stays in the same window.
    """

    def __init__(
        self,
        model_path=None,
        *,
        model=None,
        data=None,
        control_hz=None,
        camera_width=None,
        camera_height=None,
        camera_names=None,
        render_images=True,
        seed=None,
        reset_on_init=True,
        render_context=None,
        make_context_current=None,
        task_name="can_to_box",
        task=None,
        object_variants=None,
        randomize_bin_colors=False,
    ):
        if (model is None) != (data is None):
            raise ValueError("model and data must be provided together")
        self.model_path = Path(MODEL_PATH if model_path is None else model_path)
        if model is None:
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
            self.data = mujoco.MjData(self.model)
        else:
            self.model = model
            self.data = data
            if self.data.qpos.shape != (self.model.nq,):
                raise ValueError("data does not match the supplied model")
        self.home_key = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            SETTINGS.get("application.home_keyframe"),
        )
        if self.home_key < 0:
            raise ValueError("configured home keyframe does not exist")
        self.control_hz = float(
            SETTINGS.number("imitation.control_hz", positive=True)
            if control_hz is None
            else control_hz
        )
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self.steps_per_control = max(
            1, round((1.0 / self.control_hz) / self.model.opt.timestep)
        )
        self.actual_control_hz = 1.0 / (
            self.steps_per_control * self.model.opt.timestep
        )

        self.action_adapter = ActionAdapter(self.model)
        self.state_adapter = PolicyStateAdapter(self.model)
        self.head_fixed_position = np.asarray(
            SETTINGS.get("imitation.head_fixed_position_rad"), dtype=float
        )
        if self.head_fixed_position.shape != (2,):
            raise ValueError("imitation.head_fixed_position_rad must contain 2 values")
        head_joint_ids = np.asarray(
            [
                self._name_id(mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ("head_joint1", "head_joint2")
            ],
            dtype=int,
        )
        self.head_qpos = self.model.jnt_qposadr[head_joint_ids].copy()
        self.head_dofs = self.model.jnt_dofadr[head_joint_ids].copy()
        head_ranges = self.model.jnt_range[head_joint_ids]
        if np.any(self.head_fixed_position < head_ranges[:, 0]) or np.any(
            self.head_fixed_position > head_ranges[:, 1]
        ):
            raise ValueError("configured head position exceeds joint limits")
        self.left_arm_fixed = bool(SETTINGS.get("imitation.left_arm_fixed"))
        self.left_arm_park_position = np.asarray(
            SETTINGS.get("imitation.left_arm_park_position_rad"), dtype=float
        )
        self.left_grasp_fixed = SETTINGS.number(
            "imitation.left_grasp_fixed", minimum=0.0
        )
        if self.left_arm_park_position.shape != (7,):
            raise ValueError(
                "imitation.left_arm_park_position_rad must contain 7 values"
            )
        left_ranges = self.action_adapter.arm_ranges["l"]
        if np.any(self.left_arm_park_position < left_ranges[:, 0]) or np.any(
            self.left_arm_park_position > left_ranges[:, 1]
        ):
            raise ValueError("configured left-arm park position exceeds joint limits")
        if self.left_grasp_fixed > 1.0:
            raise ValueError("imitation.left_grasp_fixed must not exceed 1")
        self.arm_controllers = {
            side: arm.ArmTorqueController(self.model, ARM_JOINTS[side])
            for side in SIDES
        }
        if task is not None and getattr(task, "model", None) is not self.model:
            raise ValueError("shared task must belong to the supplied model")
        self.task = create_task(self.model, task_name) if task is None else task
        self.object_variants = (
            None if object_variants is None else tuple(object_variants)
        )
        if self.object_variants is not None:
            unknown = set(self.object_variants) - set(self.task.variant_names)
            if unknown:
                raise ValueError(
                    f"unknown variants for {self.task.name}: {sorted(unknown)}"
                )
            if not self.object_variants:
                raise ValueError("object_variants must not be empty")
        self.randomize_bin_colors = bool(randomize_bin_colors)
        self.right_arm_start_position = (
            None
            if self.task.scenario.right_arm_start_position is None
            else np.asarray(self.task.scenario.right_arm_start_position, dtype=float)
        )
        if self.right_arm_start_position is not None:
            right_ranges = self.action_adapter.arm_ranges["r"]
            below_range = np.any(self.right_arm_start_position < right_ranges[:, 0])
            above_range = np.any(self.right_arm_start_position > right_ranges[:, 1])
            if below_range or above_range:
                raise ValueError(
                    "configured right-arm start position exceeds joint limits"
                )
        self._enable_target_bin_collisions()
        self._bind_fixed_actuators()
        self._configure_passive_base_hold()

        self.render_images = bool(render_images)
        names = (
            SETTINGS.get("imitation.camera.names")
            if camera_names is None
            else camera_names
        )
        width = (
            SETTINGS.integer("imitation.camera.width", minimum=1)
            if camera_width is None
            else camera_width
        )
        height = (
            SETTINGS.integer("imitation.camera.height", minimum=1)
            if camera_height is None
            else camera_height
        )
        self.cameras = (
            MujocoCameraManager(
                self.model,
                self.data,
                width=width,
                height=height,
                camera_names=names,
                render_context=render_context,
                make_context_current=make_context_current,
            )
            if self.render_images
            else None
        )
        self.camera_names = tuple(names)

        self.rng = np.random.default_rng(seed)
        self.last_seed = seed
        self.last_action = None
        self.initial_can_position = None
        if reset_on_init:
            self.reset(seed=seed)
        else:
            mujoco.mj_forward(self.model, self.data)
            self.initial_can_position = self.task.metrics(self.data).can_position.copy()
            self.data.ctrl[self.fixed_actuators] = self.fixed_ctrl
            self.last_action = self.prepare_action(self.get_qpos())

    def _name_id(self, kind, name):
        return _required_id(self.model, kind, name)

    def _bind_fixed_actuators(self):
        fixed_names = (
            "lift_joint",
            "head_joint1",
            "head_joint2",
            "left_wheel_steer",
            "right_wheel_steer",
            "rear_wheel_steer",
            "left_wheel_drive",
            "right_wheel_drive",
            "rear_wheel_drive",
        )
        self.fixed_actuators = np.asarray(
            [self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in fixed_names],
            dtype=int,
        )
        self.fixed_ctrl = np.asarray(
            self.model.key_ctrl[self.home_key, self.fixed_actuators], dtype=float
        ).copy()
        for name, target in zip(
            ("head_joint1", "head_joint2"), self.head_fixed_position
        ):
            actuator_id = self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            fixed_index = np.flatnonzero(self.fixed_actuators == actuator_id)
            if fixed_index.size != 1:
                raise ValueError(f"head actuator is not fixed exactly once: {name}")
            self.fixed_ctrl[fixed_index[0]] = target

    def _enable_target_bin_collisions(self):
        """Enable physical contacts for task-local target bins and fixtures.

        The shared MJCF keeps these geoms on an isolated collision bit so other
        regression tasks are unchanged. Policy mode promotes active task
        geometry to the normal contact group on its model while retaining a
        target-bin-only id list for metrics and camera masks.
        """
        self.task_collision_geom_ids = enable_task_collisions(
            self.model, self.task.collision_body_names
        )
        target_body_ids = {
            _required_id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            for body_name in self.task.bin_body_names
        }
        self.target_bin_geom_ids = np.asarray(
            [
                geom_id
                for geom_id in self.task_collision_geom_ids
                if int(self.model.geom_bodyid[geom_id]) in target_body_ids
            ],
            dtype=int,
        )

    def _configure_passive_base_hold(self):
        """Hold unactuated planar base joints with physical spring/damping forces."""
        for name in ("base_x", "base_y", "base_yaw"):
            joint_id = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, name)
            dof = int(self.model.jnt_dofadr[joint_id])
            self.model.jnt_stiffness[joint_id] = 2.0e4
            self.model.dof_damping[dof] = 2.0e3

    @property
    def model_hash(self):
        return hashlib.sha256(self.model_path.read_bytes()).hexdigest()

    def reset(self, seed=None):
        """Restore the entire robot home state and randomize only the can pose."""
        if seed is None:
            seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
        self.rng = np.random.default_rng(seed)
        self.last_seed = int(seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        self.data.qpos[self.head_qpos] = self.head_fixed_position
        self.data.qvel[self.head_dofs] = 0.0
        if self.left_arm_fixed:
            self.data.qpos[self.state_adapter.arm_qpos["l"]] = (
                self.left_arm_park_position
            )
            self.data.qvel[self.state_adapter.arm_dofs["l"]] = 0.0
        if self.right_arm_start_position is not None:
            self.data.qpos[self.state_adapter.arm_qpos["r"]] = (
                self.right_arm_start_position
            )
            self.data.qvel[self.state_adapter.arm_dofs["r"]] = 0.0
        # CanInBoxTask derives its spawn anchor from the target site's world
        # position, so refresh kinematics after restoring the robot keyframe.
        mujoco.mj_forward(self.model, self.data)
        self.initial_can_position = self.task.reset(
            self.data,
            self.rng,
            allowed_variants=self.object_variants,
            randomize_bin_colors=self.randomize_bin_colors,
        )
        self.data.ctrl[self.fixed_actuators] = self.fixed_ctrl
        mujoco.mj_forward(self.model, self.data)
        self.last_action = self.prepare_action(self.get_qpos())
        return self.get_observation()

    def _apply_action_once(self, decoded):
        for side in SIDES:
            self.arm_controllers[side].apply(self.data, decoded.arm_positions[side])
            grasp.apply_grasp(
                self.model,
                self.data,
                grasp=decoded.grasp[side],
                thumb=decoded.grasp[side],
                side=side,
            )
        self.data.ctrl[self.fixed_actuators] = self.fixed_ctrl

    def step(self, action):
        """Advance one 25 Hz control frame through actuator-level physics."""
        bounded = self.prepare_action(action)
        decoded = self.action_adapter.decode(bounded)
        for _ in range(self.steps_per_control):
            self._apply_action_once(decoded)
            mujoco.mj_step(self.model, self.data)
        self.last_action = bounded
        return self.get_observation()

    def prepare_action(self, action):
        """Clip policy output and replace the locked-left fields with constants."""
        bounded = self.action_adapter.validate(action, clip=True)
        if self.left_arm_fixed:
            bounded[:7] = self.left_arm_park_position
            bounded[7] = self.left_grasp_fixed
        return bounded

    def get_qpos(self):
        return self.state_adapter.get_qpos(self.data)

    def get_qvel(self):
        return self.state_adapter.get_qvel(self.data)

    def get_images(self):
        if self.cameras is None:
            return {}
        return self.cameras.render()

    def get_observation(self):
        metrics = self.task.metrics(self.data)
        ee_pose = {
            "left": self._site_pose("grasp_target_l"),
            "right": self._site_pose("grasp_target_r"),
        }
        return {
            "qpos": self.get_qpos(),
            "qvel": self.get_qvel(),
            "ee_pose": ee_pose,
            "images": self.get_images(),
            "task": {
                "success": metrics.success,
                "object_position_error": metrics.object_position_error,
                "object_speed": metrics.object_speed,
                "scenario_name": self.task.name,
                "object_variant": metrics.object_variant,
                "target_label": metrics.target_label,
            },
            "debug": {
                "full_qpos": np.asarray(self.data.qpos).copy(),
                "full_qvel": np.asarray(self.data.qvel).copy(),
                "task_object_pose": np.concatenate(
                    (
                        metrics.can_position,
                        self.data.qpos[self.task.can_qpos + 3 : self.task.can_qpos + 7],
                    )
                ).copy(),
                "target_position": metrics.target_position.copy(),
            },
        }

    def _site_pose(self, name):
        site_id = self._name_id(mujoco.mjtObj.mjOBJ_SITE, name)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, self.data.site_xmat[site_id])
        return np.concatenate((self.data.site_xpos[site_id], quaternion))

    def close(self):
        if self.cameras is not None:
            self.cameras.close()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()


__all__ = [
    "AIWorkerMujocoEnv",
    "enable_task_collisions",
]
