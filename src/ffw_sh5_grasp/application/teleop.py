"""``models/full_scene.xml``을 실행하는 단일 창 텔레오퍼레이션 앱.

참조 영상의 인터페이스를 3D 화면 위에서 이동 가능한 ImGui 도구 창으로 재현한다.
홈 기준 X/Y/Z/Roll/Pitch/Yaw 말단 자세 슬라이더가 ROS 없는 전신 IK(베이스, 리프트,
양팔), 팔 토크 제어, 파지·엄지 시너지, 관절 모니터와 HUD를 구동한다.

렌더링은 같은 GLFW 창에서 MuJoCo 저수준 API와 ImGui를 함께 사용한다. ImGui OpenGL
렌더러가 호환되는 GLX 문맥을 받도록 GLFW는 X11 백엔드를 사용해야 한다. UI, 제어,
물리와 렌더링 단계는 모두 한 스레드에서 실행된다.

RPY 슬라이더는 각 손의 시작 자세를 기준으로 한 로컬 오프셋을 나타낸다. 전신 자세
목표는 월드 좌표계에 고정되며, 수동 주행 중에는 차체와 함께 이동한다.

**전신 제어와 모바일 베이스**: 말단 목표는 시작 월드 자세에 고정된 홈 기준 UI
값이다. `control/whole_body.py`는 베이스 x/y/yaw, 리프트와 양팔 7자유도를 하나의
경계 제한 가중 미분 IK 문제로 푼다. 베이스 속도는 차체 좌표계 `BodyTwist`로 변환한
뒤 `base.SwerveDrive`로 전달한다. 실제 조향·구동 액추에이터만 명령하므로 베이스의
모든 이동은 바퀴와 지면의 마찰로 발생한다. 키보드 차체 속도는 누르는 동안 우선하며
동일한 스워브 경로를 사용한다. ROS/MoveIt 의존성이나 로봇 qpos 덮어쓰기는 없다.

실행: `python3 src/teleop_app.py`
마우스: 왼쪽 드래그 회전, 오른쪽 드래그 이동, 스크롤 확대·축소
키보드: 위/아래는 베이스 전진·후진, 왼쪽/오른쪽은 베이스 회전, 대괄호는 좌우 이동,
Q/E는 리프트 하강·상승, R은 캔 무작위 재배치, G는 접촉력 표시, V는 충돌 형상과
CBF 표시, C는 카메라 프리셋 전환이다. R/G/V/C 기능은 화면 도구 창의 버튼으로도
실행할 수 있다.

"""

import argparse
import math
import sys
import time
from pathlib import Path

import glfw

# Linux에서는 호환되는 GLX 문맥을 선택하도록 ``glfw.init()`` 전에 X11을 지정한다.
# macOS에는 X11 backend가 없으므로 GLFW가 Cocoa backend를 자동 선택하게 둔다.
if sys.platform.startswith("linux"):
    glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)

import mujoco
import numpy as np

from ffw_sh5_grasp.config import CONFIG_ENV_VAR, SETTINGS  # noqa: E402
from ffw_sh5_grasp.control import (  # noqa: E402
    arm,
    base,
    grasp,  # noqa: E402
    whole_body,  # noqa: E402
)
from ffw_sh5_grasp.imitation.runtime.catalog import (  # noqa: E402
    ACT_OUTPUT_DIRS,
    discover_policy_runs,
)
from ffw_sh5_grasp.imitation.runtime.task_space import (  # noqa: E402
    task_action_to_joint,
)
from ffw_sh5_grasp.imitation.simulation.environment import (  # noqa: E402
    enable_task_collisions,
)
from ffw_sh5_grasp.imitation.simulation.task import (  # noqa: E402
    TASK_NAMES,
    create_task,
)
from ffw_sh5_grasp.paths import MODEL_PATH  # noqa: E402
from ffw_sh5_grasp.visualization import render, ui  # noqa: E402

from . import control_loop, state, targets  # noqa: E402

# 양팔 관절 이름 목록 (IK solver / 토크 제어기에 그대로 넘겨진다).
ARM_R = [f"arm_r_joint{i}" for i in range(1, 8)]
ARM_L = [f"arm_l_joint{i}" for i in range(1, 8)]
SIDES = ("r", "l")
ARM_JOINTS = {"r": ARM_R, "l": ARM_L}
WHEELS = base.WHEELS
# ``models/full_scene.xml``의 ``home`` 키프레임과 일치한다. 이 자세는 관련
# ffw-sh5-mujoco 저장소의 휴지 자세와 같으며, 팔꿈치인 4번 관절만 -90도이고
# 나머지 관절은 모두 0도다.
HOME_Q_R = np.asarray(SETTINGS.get("application.home_arm_position_rad"), dtype=float)
HOME_Q_L = HOME_Q_R.copy()
LIFT_RANGE = tuple(float(value) for value in SETTINGS.get("application.lift_range_m"))
VIRTUAL_OBJECT_HOME_POS = np.asarray(
    SETTINGS.get("application.virtual_object_home_position_m"), dtype=float
)
HOME_KEYFRAME = SETTINGS.get("application.home_keyframe")
# 패널의 "Joint position monitor"에 진행률 막대로 표시할 관절 전체 목록.
MONITOR_JOINTS = (
    [f"arm_r_joint{i}" for i in range(1, 8)]
    + [f"arm_l_joint{i}" for i in range(1, 8)]
    + [f"finger_r_joint{i}" for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)]
    + [f"finger_l_joint{i}" for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)]
    + ["lift_joint", "head_joint1", "head_joint2"]
)

WINDOW_W = SETTINGS.integer("application.window.width", minimum=1)
WINDOW_H = SETTINGS.integer("application.window.height", minimum=1)
LOOP_HZ = SETTINGS.number("application.loop_hz", positive=True)
ENV_TASK_NAMES = {0: "can_to_box", 1: "can_color_sort"}
POLICY_REPRESENTATION_NAMES = ("auto", "joint", "task")
DEFAULT_TASK_POLICY_IK_SPEED_SCALE = SETTINGS.number(
    "imitation.policy.task_ik_speed_scale", positive=True
)
DEFAULT_POLICY_PTE_STEPS = SETTINGS.integer("imitation.policy.pte_steps", minimum=0)
DEFAULT_POLICY_RERUN_LOG_HZ = SETTINGS.number(
    "imitation.policy.rerun_log_hz", positive=True
)
# 목표 변화율 제한은 원시 슬라이더 값의 급격한 변화와 무관하게 실제 IK 목표가
# 렌더링 한 프레임에서 이동할 수 있는 거리를 제한한다. 25 Hz에서 프레임당 0.03 m는
# 0.75 m/s, 프레임당 8도는 200 deg/s에 해당해 빠르지만 추종 가능한 범위다.
MAX_POS_STEP_PER_FRAME = SETTINGS.number(
    "application.max_position_step_per_frame_m", positive=True
)
MAX_RPY_STEP_PER_FRAME_DEG = SETTINGS.number(
    "application.max_rotation_step_per_frame_deg", positive=True
)
LIFT_JOG_SPEED = SETTINGS.number("application.lift_jog_speed_m_s", positive=True)
GRASP_COMMAND_RATE = SETTINGS.number(
    "application.grasp_command_rate_per_s", positive=True
)
MANUAL_STOP_LINEAR_SPEED = SETTINGS.number(
    "application.manual_stop_linear_speed_m_s", minimum=0.0
)
MANUAL_STOP_ANGULAR_SPEED = SETTINGS.number(
    "application.manual_stop_angular_speed_rad_s", minimum=0.0
)
if LIFT_RANGE[0] >= LIFT_RANGE[1]:
    raise ValueError("application.lift_range_m은 [최솟값, 최댓값] 순서여야 합니다.")


def _named_id(model, object_type, name):
    """필수 MuJoCo 객체 이름을 ID로 바꾸고 누락 시 문맥이 있는 오류를 낸다."""
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo object not found: {name!r}")
    return object_id


def _joint_address(model, name, addresses):
    """관절 이름을 qpos 또는 dof 주소 배열의 정수 인덱스로 변환한다."""
    joint_id = _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(addresses[joint_id])


def _is_relative_to(path, root):
    """Return whether a resolved path is contained by a resolved root."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class KeyEdge:
    """키를 누른 순간에만 한 번 참을 반환하는 엣지 입력 검사기."""

    def __init__(self):
        """이전 프레임에 눌려 있던 키 집합을 빈 상태로 초기화한다."""
        self._prev = set()

    def pressed(self, window, key):
        """지정 GLFW 키가 이번 프레임에 새로 눌렸을 때만 ``True``를 반환한다."""
        down = glfw.get_key(window, key) == glfw.PRESS
        was_down = key in self._prev
        if down:
            self._prev.add(key)
        else:
            self._prev.discard(key)
        return down and not was_down


class TeleopApp:
    """MuJoCo 주 창과 분리형 도구 창을 쓰는 텔레옵 앱. `run()`의 메인 루프가 매 프레임 아래 단계를
    순서대로 실행한다: 마우스 카메라 -> 엣지 키(R/G/V/C) -> 연속 키(주행/리프트) ->
    UI 패널 -> 물리 스텝 -> 렌더링. 상태는 전부 인스턴스 속성(self.*)에 있다."""

    def __init__(
        self,
        *,
        policy_checkpoint=None,
        policy_stats=None,
        policy_device="auto",
        policy_seed=1000,
        policy_max_steps=500,
        task_name="can_to_box",
        policy_representation="auto",
        policy_rerun=False,
        policy_rerun_port=9877,
        policy_ik_speed_scale=DEFAULT_TASK_POLICY_IK_SPEED_SCALE,
        policy_pte_steps=DEFAULT_POLICY_PTE_STEPS,
        policy_rerun_hz=DEFAULT_POLICY_RERUN_LOG_HZ,
    ):
        """시뮬레이션·렌더링·루프 상태를 순서대로 구성해 실행 가능한 앱을 만든다."""
        self.act_policy_device = policy_device
        self.act_policy_stats = policy_stats
        self.act_policy_seed = int(policy_seed)
        self.act_policy_representation = str(policy_representation).lower()
        if self.act_policy_representation not in POLICY_REPRESENTATION_NAMES:
            raise ValueError("policy representation must be auto, joint, or task")
        self.act_policy_rerun_enabled = bool(policy_rerun)
        self.act_policy_rerun_port = int(policy_rerun_port)
        self.act_policy_rerun_hz = float(policy_rerun_hz)
        if not np.isfinite(self.act_policy_rerun_hz) or self.act_policy_rerun_hz <= 0.0:
            raise ValueError("policy Rerun frequency must be finite and positive")
        self.act_policy_ik_speed_scale = float(policy_ik_speed_scale)
        if (
            not np.isfinite(self.act_policy_ik_speed_scale)
            or self.act_policy_ik_speed_scale <= 0.0
        ):
            raise ValueError("policy IK speed scale must be finite and positive")
        self.act_policy_pte_steps = int(policy_pte_steps)
        if (
            isinstance(policy_pte_steps, bool)
            or self.act_policy_pte_steps != policy_pte_steps
            or self.act_policy_pte_steps < 0
        ):
            raise ValueError("policy PTE steps must be a non-negative integer")
        if task_name not in TASK_NAMES:
            raise ValueError(f"unsupported teleop environment: {task_name}")
        self.task_name = task_name
        self._initial_policy_checkpoint = policy_checkpoint
        self._initial_policy_max_steps = int(policy_max_steps)
        self._setup_sim()
        render.setup_render(self, WINDOW_W, WINDOW_H)
        self._setup_loop_state()

    # 초기화

    def _setup_sim(self):
        """렌더링과 독립적인 모델·제어기·주소·목표 상태를 순서대로 구성한다."""
        self._load_model_state()
        self.task_name = getattr(self, "task_name", "can_to_box")
        self.imitation_task = create_task(self.model, self.task_name)
        # WholeBodyIK builds its collision-pair catalog at construction time.
        # Enable the task bin first so both physics and the CBF include it.
        enable_task_collisions(self.model, self.imitation_task.bin_body_names)
        self._setup_control_systems()
        self._bind_model_entities()
        self._setup_target_state()
        self.reset_can()

    def _load_model_state(self):
        """MJCF를 로드하고 home keyframe의 초기 ``MjData``를 만든다."""
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEYFRAME)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        self.model = model
        self.data = data

    def _setup_control_systems(self):
        """팔 토크, Whole-body IK와 스워브 제어기를 모델에 연결한다."""
        model = self.model
        self.arm_controllers = {
            side: arm.ArmTorqueController(model, joint_names)
            for side, joint_names in ARM_JOINTS.items()
        }
        self.whole_body_enabled = True
        self.whole_body_solver = whole_body.WholeBodyIK(
            model,
            {"r": "grasp_target_r", "l": "grasp_target_l"},
            ARM_JOINTS,
        )
        # FK(관절각 직접 제어) 모드에서 패널 슬라이더의 최소/최대값으로 쓸, 각 팔
        # 관절 범위를 도 단위로 미리 계산한다.
        self.arm_joint_ranges_deg = {
            side: [
                tuple(
                    math.degrees(value)
                    for value in model.jnt_range[
                        _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    ]
                )
                for name in joints
            ]
            for side, joints in ARM_JOINTS.items()
        }
        self.base_drive = base.SwerveDrive()

    def _bind_model_entities(self):
        """프레임 루프가 반복 사용할 actuator, joint와 marker 주소를 저장한다."""
        model = self.model
        base_bindings = state.BaseBindings(
            x_qpos=_joint_address(model, "base_x", model.jnt_qposadr),
            y_qpos=_joint_address(model, "base_y", model.jnt_qposadr),
            yaw_qpos=_joint_address(model, "base_yaw", model.jnt_qposadr),
            x_dof=_joint_address(model, "base_x", model.jnt_dofadr),
            y_dof=_joint_address(model, "base_y", model.jnt_dofadr),
            yaw_dof=_joint_address(model, "base_yaw", model.jnt_dofadr),
        )
        # 바퀴마다 실제 조향 위치 액추에이터와 구동 속도 액추에이터를 사용한다.
        # 자세한 변환은 ``control/base.py``의 ``SwerveDrive``가 담당한다. 베이스 이동은
        # base_x/base_y/base_yaw 직접 구동이 아니라 바퀴와 지면의 마찰 결과다. 해당
        # 관절은 base_link가 넘어지지 않게 남아 있지만 직접 구동되지는 않는다.
        wheel_bindings = {
            wheel: state.WheelBinding(
                steer_actuator=_named_id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{wheel}_steer"
                ),
                drive_actuator=_named_id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{wheel}_drive"
                ),
                steer_qpos=_joint_address(
                    model, f"{wheel}_steer_joint", model.jnt_qposadr
                ),
                drive_dof=_joint_address(
                    model, f"{wheel}_drive_joint", model.jnt_dofadr
                ),
            )
            for wheel in WHEELS
        }
        monitor_joint_ids = {
            name: _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in MONITOR_JOINTS
        }
        monitor_qpos = {
            name: int(model.jnt_qposadr[joint_id])
            for name, joint_id in monitor_joint_ids.items()
        }
        monitor_ranges = {
            name: model.jnt_range[joint_id]
            for name, joint_id in monitor_joint_ids.items()
        }
        self.rng = np.random.default_rng(getattr(self, "act_policy_seed", 1000))

        hand_mocap_ids = {
            side: model.body_mocapid[
                _named_id(model, mujoco.mjtObj.mjOBJ_BODY, f"ik_target_{side}")
            ]
            for side in SIDES
        }
        virtual_mocap_id = model.body_mocapid[
            _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "virtual_object_marker")
        ]
        virtual_geom_id = _named_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "virtual_object_marker_geom"
        )
        virtual_site_id = _named_id(
            model, mujoco.mjtObj.mjOBJ_SITE, "virtual_object_marker_site"
        )
        self.bindings = state.ModelBindings(
            lift_actuator=_named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_joint"),
            lift_qpos=_joint_address(model, "lift_joint", model.jnt_qposadr),
            base=base_bindings,
            wheels=wheel_bindings,
            monitor_qpos=monitor_qpos,
            monitor_ranges=monitor_ranges,
            markers=state.MarkerBindings(
                hand_mocap_ids=hand_mocap_ids,
                virtual_mocap_id=virtual_mocap_id,
                virtual_geom_id=virtual_geom_id,
                virtual_site_id=virtual_site_id,
                virtual_geom_rgba=model.geom_rgba[virtual_geom_id],
                virtual_site_rgba=model.site_rgba[virtual_site_id],
            ),
            can_joint=_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "can_free"),
            can_geom=_named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "can_geom"),
        )
        self._apply_default_imitation_pose()
        targets.set_home_references(self)

    def _apply_default_imitation_pose(self):
        """Apply the policy collection pose before teleop targets are created."""
        self.arm_qpos_indices = {
            side: np.asarray(
                [
                    _joint_address(self.model, name, self.model.jnt_qposadr)
                    for name in ARM_JOINTS[side]
                ],
                dtype=int,
            )
            for side in SIDES
        }
        left_park = np.asarray(
            SETTINGS.get("imitation.left_arm_park_position_rad"), dtype=float
        )
        head_fixed = np.asarray(
            SETTINGS.get("imitation.head_fixed_position_rad"), dtype=float
        )
        self.data.qpos[self.arm_qpos_indices["l"]] = left_park
        right_start = self.imitation_task.scenario.right_arm_start_position
        if right_start is not None:
            self.data.qpos[self.arm_qpos_indices["r"]] = right_start
        for name, value in zip(("head_joint1", "head_joint2"), head_fixed):
            joint_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos = int(self.model.jnt_qposadr[joint_id])
            dof = int(self.model.jnt_dofadr[joint_id])
            actuator = _named_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.data.qpos[qpos] = value
            self.data.qvel[dof] = 0.0
            self.data.ctrl[actuator] = value
        self.home_arm_positions = {
            "r": np.asarray(
                self.data.qpos[self.arm_qpos_indices["r"]], dtype=float
            ).copy(),
            "l": left_park.copy(),
        }

    def _setup_target_state(self):
        """UI 목표, 평활화 상태와 실행 중 표시·모드 상태를 초기화한다."""
        data = self.data
        self.targets = {
            **{
                f"{field}_{side}": [0.0, 0.0, 0.0]
                for side in SIDES
                for field in ("pos", "rpy")
            },
            **{
                f"{field}_{side}": 0.0 for side in SIDES for field in ("grasp", "thumb")
            },
            "virtual_object_pos": VIRTUAL_OBJECT_HOME_POS.tolist(),
            "virtual_object_rpy": [0.0, 0.0, 0.0],
            "lift": float(data.qpos[self.bindings.lift_qpos]),
        }
        # 입력하거나 드래그한 목표가 급변하지 않도록 전신 IK는 평활화된 사본을 추종한다.
        self.smoothed_pos = {
            side: np.array(self.targets[f"pos_{side}"]) for side in SIDES
        }
        self.smoothed_rpy = {
            side: np.array(self.targets[f"rpy_{side}"]) for side in SIDES
        }
        self.lift_cmd = self.targets["lift"]
        self.whole_body_base_twist = base.BodyTwist()
        self.commanded_base_twist = base.BodyTwist()
        self._manual_override_active = False
        base_bindings = self.bindings.base
        self._manual_reference_base_pose = np.array(
            [
                data.qpos[base_bindings.x_qpos],
                data.qpos[base_bindings.y_qpos],
                data.qpos[base_bindings.yaw_qpos],
            ],
            dtype=float,
        )
        targets.sync_ik_mocaps_from_targets(self)
        self.contact_viz = False
        self.collision_viz = False
        self.collision_active_pairs = ()
        self.collision_min_distance = math.inf
        self.collision_constraint_violation = 0.0
        self.camera_preset = 0
        self.grab_state = dict.fromkeys(SIDES)
        self.cyclo_controller = "movel"
        self.cyclo_grasp_captured = False
        self.cyclo_capture_offsets = None
        self.cyclo_status = "ready"
        self.lift_range = LIFT_RANGE

    def _setup_loop_state(self):
        """메인 루프에서만 쓰는 상태(IK 웜스타트 값, 타이밍, 입력 헬퍼)."""
        self.q_des = {
            side: np.asarray(
                self.data.qpos[self.arm_qpos_indices[side]], dtype=float
            ).copy()
            for side in SIDES
        }
        # 손별 제어 모드: "ik"(EE 포즈 슬라이더 -> whole-body solver) 또는
        # "fk"(관절각 슬라이더를 그대로 토크 제어기 목표로 사용, IK 자체를 건너뜀).
        # 리프트를 움직이는 동안 IK가 매 프레임에서만 어깨 높이를 다시 읽어들여서
        # 생기는 출렁임(리프트가 프레임 사이에도 계속 움직이는데 IK는 프레임당
        # 한 번만 풀림)을 피하고 싶을 때 FK로 전환해 관절각을 고정해두면, 팔 전체가
        # 리프트에 강체로 붙어 그대로 오르내리기만 해서 흔들림이 아예 없다.
        self.arm_mode = {"r": "ik", "l": "ik"}
        self.fk_q_deg = {
            side: np.degrees(q_des).tolist() for side, q_des in self.q_des.items()
        }
        self.frame_dt = 1.0 / LOOP_HZ
        self.steps_per_frame = max(1, round(self.frame_dt / self.model.opt.timestep))
        self.freq_ema = LOOP_HZ
        self.wall_start = time.perf_counter()
        self.last_mouse = list(glfw.get_cursor_pos(self.window))
        self.keys = KeyEdge()
        self.ik_err_mm = {"l": 0.0, "r": 0.0}
        self.gizmo_mouse_active = False
        self.act_policy_runs = ()
        self.act_policy_run_index = 0
        self.act_policy_checkpoint_index = 0
        self.act_policy_max_steps = max(
            1, getattr(self, "_initial_policy_max_steps", 500)
        )
        self.act_policy_status = ""
        self.act_policy_load_request = None
        self.act_policy_env = None
        self.act_policy_runner = None
        self.act_policy_rerun_logger = None
        self.act_policy_rerun_frame = 0
        self.act_policy_rerun_path = None
        self.act_policy_checkpoint = None
        self.act_policy_observation = None
        self.act_policy_info = None
        self.act_policy_frame = 0
        self.act_policy_running = False
        self.act_policy_step_requested = False
        # ImGui button callbacks run while the current UI frame is being
        # assembled.  Closing the policy renderer from such a callback can
        # destroy/switch its OpenGL context underneath ImGui, so handoff is
        # deferred until the physics phase immediately after UI drawing.
        self.act_policy_finish_request = None
        self._act_policy_base_hold_backup = None
        self.refresh_act_policies()
        initial_checkpoint = getattr(self, "_initial_policy_checkpoint", None)
        if initial_checkpoint is not None:
            self.request_act_policy_launch(
                initial_checkpoint, self.act_policy_max_steps
            )

    def refresh_act_policies(self):
        """Refresh checkpoints and apply the selected representation filter."""
        selected_run = None
        selected_checkpoint = None
        if self.act_policy_runs:
            run_index = min(self.act_policy_run_index, len(self.act_policy_runs) - 1)
            selected = self.act_policy_runs[run_index]
            selected_run = selected.name
            if selected.checkpoints:
                checkpoint_index = min(
                    self.act_policy_checkpoint_index,
                    len(selected.checkpoints) - 1,
                )
                selected_checkpoint = selected.checkpoints[checkpoint_index].name

        discovered_runs = discover_policy_runs()
        requested_representation = getattr(self, "act_policy_representation", "auto")
        self.act_policy_runs = tuple(
            run
            for run in discovered_runs
            if (
                requested_representation == "auto"
                or run.representation == requested_representation
            )
        )
        self.act_policy_run_index = next(
            (
                index
                for index, run in enumerate(self.act_policy_runs)
                if run.name == selected_run
            ),
            0,
        )
        self.act_policy_checkpoint_index = 0
        if self.act_policy_runs:
            run = self.act_policy_runs[self.act_policy_run_index]
            self.act_policy_checkpoint_index = next(
                (
                    index
                    for index, checkpoint in enumerate(run.checkpoints)
                    if checkpoint.name == selected_checkpoint
                ),
                0,
            )
            self.act_policy_status = (
                f"Found {len(self.act_policy_runs)} "
                f"{requested_representation.upper()} policy run(s)"
            )
        else:
            roots = ", ".join(str(path) for path in ACT_OUTPUT_DIRS)
            self.act_policy_status = f"No ACT checkpoints found in {roots}"

    def set_act_policy_representation(self, representation):
        """Change the checkpoint-list filter and next-load representation."""
        representation = str(representation).lower()
        if representation not in POLICY_REPRESENTATION_NAMES:
            raise ValueError("policy representation must be auto, joint, or task")
        if representation == self.act_policy_representation:
            return False
        self.act_policy_representation = representation
        self.act_policy_run_index = 0
        self.act_policy_checkpoint_index = 0
        self.refresh_act_policies()
        return True

    def set_act_policy_pte_steps(self, steps):
        """Apply a PTE look-ahead now, resetting incompatible temporal state."""
        if isinstance(steps, bool) or int(steps) != steps or int(steps) < 0:
            self.act_policy_status = "PTE steps must be a non-negative integer"
            return False
        steps = int(steps)
        runner = getattr(self, "act_policy_runner", None)
        if runner is not None:
            try:
                runner.set_proleptic_steps(steps)
            except ValueError as error:
                self.act_policy_status = str(error)
                return False
        self.act_policy_pte_steps = steps
        if runner is not None:
            seconds = steps / self.act_policy_env.actual_control_hz
            self.act_policy_status = (
                f"PTE f={steps} ({seconds:.3f}s); temporal buffer reset"
            )
        return True

    def request_act_policy_launch(self, checkpoint, max_steps):
        """Validate and queue a checkpoint for loading in the current window."""
        checkpoint = Path(checkpoint).resolve()
        if not any(
            _is_relative_to(checkpoint, root.resolve()) for root in ACT_OUTPUT_DIRS
        ):
            self.act_policy_status = (
                "Checkpoint must be inside outputs/act or outputs/act_modular"
            )
            return False
        if not checkpoint.is_file() or checkpoint.suffix != ".ckpt":
            self.act_policy_status = "Selected checkpoint is unavailable"
            return False
        if max_steps <= 0:
            self.act_policy_status = "Max steps must be positive"
            return False
        self.act_policy_max_steps = int(max_steps)
        self.act_policy_load_request = checkpoint
        self.act_policy_status = f"Loading {checkpoint.name}..."
        return True

    def _close_act_policy(self):
        rerun_logger = getattr(self, "act_policy_rerun_logger", None)
        # Make cleanup idempotent even when the optional Rerun sink fails.
        self.act_policy_rerun_logger = None
        if rerun_logger is not None:
            rerun_logger.__exit__(None, None, None)
        if self.act_policy_env is not None:
            self.act_policy_env.close()
        self._restore_base_hold_after_policy()
        self.act_policy_env = None
        self.act_policy_runner = None
        self.act_policy_rerun_frame = 0
        self.act_policy_rerun_path = None
        self.act_policy_checkpoint = None
        self.act_policy_observation = None
        self.act_policy_info = None
        self.act_policy_running = False
        self.act_policy_step_requested = False
        self.act_policy_finish_request = None

    def _capture_base_hold_before_policy(self):
        backup = []
        for name in ("base_x", "base_y", "base_yaw"):
            joint_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            dof = int(self.model.jnt_dofadr[joint_id])
            backup.append(
                (
                    joint_id,
                    dof,
                    float(self.model.jnt_stiffness[joint_id]),
                    float(self.model.dof_damping[dof]),
                )
            )
        self._act_policy_base_hold_backup = tuple(backup)

    def _restore_base_hold_after_policy(self):
        if self._act_policy_base_hold_backup is None:
            return
        for joint_id, dof, stiffness, damping in self._act_policy_base_hold_backup:
            self.model.jnt_stiffness[joint_id] = stiffness
            self.model.dof_damping[dof] = damping
        self._act_policy_base_hold_backup = None

    def _load_requested_act_policy(self):
        checkpoint = getattr(self, "act_policy_load_request", None)
        if checkpoint is None:
            return
        self.act_policy_load_request = None
        self._close_act_policy()
        environment = None
        rerun_logger = None
        try:
            from ffw_sh5_grasp.imitation.runtime.runner import ACTPolicyRunner
            from ffw_sh5_grasp.imitation.simulation.environment import (
                AIWorkerMujocoEnv,
            )
            from ffw_sh5_grasp.imitation.visualization.rerun_rollout import (
                RolloutRerunLogger,
            )

            runner = ACTPolicyRunner(
                checkpoint,
                self.act_policy_stats,
                device=self.act_policy_device,
                representation=getattr(self, "act_policy_representation", "auto"),
                proleptic_steps=getattr(
                    self, "act_policy_pte_steps", DEFAULT_POLICY_PTE_STEPS
                ),
            )
            self._capture_base_hold_before_policy()
            environment = AIWorkerMujocoEnv(
                model_path=MODEL_PATH,
                model=self.model,
                data=self.data,
                camera_names=runner.camera_names,
                render_images=True,
                seed=self.act_policy_seed,
                reset_on_init=False,
                task=self.imitation_task,
                render_context=self.context,
                make_context_current=(lambda: glfw.make_context_current(self.window)),
            )
            if not np.isclose(environment.actual_control_hz, 1.0 / self.frame_dt):
                raise ValueError("ACT and teleop control frequencies must match")
            if getattr(self, "act_policy_rerun_enabled", False):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                suffix = time.time_ns() % 1_000_000_000
                rerun_path = (
                    checkpoint.parent.parent
                    / "rerun"
                    / (f"teleop_{runner.representation}_{timestamp}_{suffix:09d}.rrd")
                )
                rerun_logger = RolloutRerunLogger(
                    rerun_path,
                    runner.camera_names,
                    application_id="aiworker_teleop_policy",
                    live=True,
                    port=getattr(self, "act_policy_rerun_port", 9877),
                    frame_stride=max(
                        1,
                        round(
                            environment.actual_control_hz
                            / getattr(
                                self, "act_policy_rerun_hz", DEFAULT_POLICY_RERUN_LOG_HZ
                            )
                        ),
                    ),
                )
                rerun_logger.__enter__()
        except Exception as error:
            if rerun_logger is not None:
                rerun_logger.__exit__(None, None, None)
            if environment is not None:
                environment.close()
            self._restore_base_hold_after_policy()
            self.act_policy_status = f"Load failed: {error}"
            glfw.make_context_current(self.window)
            return

        self.act_policy_runner = runner
        self.act_policy_env = environment
        self.act_policy_rerun_logger = rerun_logger
        self.act_policy_rerun_frame = 0
        self.act_policy_rerun_path = None if rerun_logger is None else rerun_logger.path
        self.act_policy_checkpoint = checkpoint
        self.act_policy_observation = environment.get_observation()
        self.act_policy_info = None
        self.act_policy_frame = 0
        self.act_policy_running = True
        self.act_policy_step_requested = False
        self.act_policy_finish_request = None
        self.act_policy_status = (
            f"ACT {runner.representation}-space policy running; "
            f"PTE f={runner.proleptic_steps}"
        )
        if runner.representation == "task":
            self.act_policy_status += (
                f"; IK speed {self.act_policy_ik_speed_scale:.2f}x"
            )
        if rerun_logger is not None:
            self.act_policy_status += (
                f"; Rerun :{self.act_policy_rerun_port} -> "
                f"{rerun_logger.path.name} "
                f"(~{environment.actual_control_hz / rerun_logger.frame_stride:.1f}Hz)"
            )
        # Operator-only IK targets are not part of policy observations and are
        # distracting while the arm-only controller owns the robot.
        self.opt.geomgroup[5] = 0
        self.opt.sitegroup[5] = 0
        glfw.make_context_current(self.window)

    def reset_act_policy(self):
        if self.act_policy_env is None:
            return
        self.act_policy_env.initial_can_position = self.reset_can().copy()
        self.act_policy_observation = self.act_policy_env.get_observation()
        self.act_policy_runner.reset()
        self.act_policy_info = None
        self.act_policy_frame = 0
        self.act_policy_running = False
        self.act_policy_step_requested = False
        self.act_policy_finish_request = None
        self.act_policy_status = "Can reset; ACT temporal state reset"
        glfw.make_context_current(self.window)

    def request_finish_act_policy(self, reason="ACT policy finished"):
        """Queue a policy-to-IK handoff outside the active ImGui callback."""
        if self.act_policy_env is None:
            return
        self.act_policy_running = False
        self.act_policy_step_requested = False
        self.act_policy_finish_request = reason
        self.act_policy_status = f"{reason}; switching to IK..."

    def finish_act_policy(self, reason="ACT policy finished"):
        """Release policy ownership and synchronize IK to the measured pose."""
        if self.act_policy_env is None:
            return
        policy_qpos = self.act_policy_env.get_qpos()
        hand_poses = {
            side: (measured.position.copy(), measured.quaternion.copy())
            for side in SIDES
            for measured in (self.whole_body_solver.site_state(self.data, side),)
        }
        self._close_act_policy()
        # ``mujoco.Renderer.close`` tears down an offscreen render context.
        # Rebind both the visible GLFW context and MuJoCo's window framebuffer
        # before the operator view is drawn again.
        render.restore_window_render_target(self)

        self.whole_body_enabled = True
        self.cyclo_grasp_captured = False
        self.cyclo_capture_offsets = None
        self.cyclo_controller = "movel"
        self.whole_body_solver.set_rigid_grasp(self.data, False)
        for side in SIDES:
            self.arm_mode[side] = "ik"
            position, quaternion = hand_poses[side]
            self.targets[f"pos_{side}"] = targets.world_to_target_pos(
                self, side, position
            )
            self.targets[f"rpy_{side}"] = list(
                targets.world_quat_to_target_rpy(self, side, quaternion)
            )
            self.smoothed_pos[side] = np.asarray(
                self.targets[f"pos_{side}"], dtype=float
            ).copy()
            self.smoothed_rpy[side] = np.asarray(
                self.targets[f"rpy_{side}"], dtype=float
            ).copy()
            measured_arm = np.asarray(
                self.data.qpos[self.arm_qpos_indices[side]], dtype=float
            ).copy()
            self.q_des[side] = measured_arm
            self.fk_q_deg[side] = np.degrees(measured_arm).tolist()
        self.targets["grasp_l"] = float(policy_qpos[7])
        self.targets["thumb_l"] = float(policy_qpos[7])
        self.targets["grasp_r"] = float(policy_qpos[15])
        self.targets["thumb_r"] = float(policy_qpos[15])
        self.grab_state = dict.fromkeys(SIDES)
        self.targets["lift"] = float(self.data.qpos[self.bindings.lift_qpos])
        self.lift_cmd = self.targets["lift"]
        self.whole_body_solver.rebase(self.data, hand_poses)
        self.whole_body_base_twist = base.BodyTwist()
        self.commanded_base_twist = base.BodyTwist()
        targets.sync_ik_mocaps_from_targets(self)
        self.opt.geomgroup[5] = 1
        self.opt.sitegroup[5] = 1
        self.act_policy_status = f"{reason}; IK control restored"

    def toggle_act_policy(self):
        if self.act_policy_env is None:
            return
        if self.act_policy_frame >= self.act_policy_max_steps:
            self.reset_act_policy()
        self.act_policy_running = not self.act_policy_running
        self.act_policy_status = (
            "ACT policy running" if self.act_policy_running else "ACT policy paused"
        )

    def request_act_policy_step(self):
        if (
            self.act_policy_env is not None
            and self.act_policy_frame < self.act_policy_max_steps
        ):
            self.act_policy_step_requested = True

    def _task_policy_action_to_joint(self, task_action):
        """Convert one world-frame right EE pose action through arm-only IK."""
        speed_scale = float(
            getattr(
                self, "act_policy_ik_speed_scale", DEFAULT_TASK_POLICY_IK_SPEED_SCALE
            )
        )
        joint_action, diagnostics = task_action_to_joint(
            self.act_policy_env,
            self.whole_body_solver,
            task_action,
            speed_scale=speed_scale,
        )
        self.ik_err_mm["r"] = diagnostics.position_error_mm
        self.collision_min_distance = diagnostics.minimum_collision_distance_m
        self.collision_constraint_violation = diagnostics.collision_constraint_violation
        self.collision_active_pairs = diagnostics.active_collision_pairs
        return joint_action

    def _step_act_policy(self):
        """Run one embedded ACT frame and report whether policy owns physics."""
        if not hasattr(self, "act_policy_load_request"):
            return False
        self._load_requested_act_policy()
        if self.act_policy_env is None:
            return False
        if self.act_policy_finish_request is not None:
            reason = self.act_policy_finish_request
            self.act_policy_finish_request = None
            self.finish_act_policy(reason)
            # Keep ownership of this transition frame. Normal IK begins on
            # the next frame, after renderer teardown and target rebasing.
            return True
        should_step = self.act_policy_running or self.act_policy_step_requested
        self.act_policy_step_requested = False
        if not should_step:
            return True
        if self.act_policy_frame >= self.act_policy_max_steps:
            self.finish_act_policy("Max steps reached")
            return True
        try:
            action, self.act_policy_info = self.act_policy_runner.get_action(
                self.act_policy_observation
            )
            if self.act_policy_runner.representation == "task":
                task_action = action.copy()
                action = self._task_policy_action_to_joint(task_action)
                self.act_policy_info["task_action"] = task_action
            action = self.act_policy_env.prepare_action(action)
            if self.act_policy_runner.representation == "task":
                self.act_policy_info["executed_joint_action"] = action.copy()
            rerun_logger = self.act_policy_rerun_logger
            if rerun_logger is not None:
                ik_metrics = None
                if self.act_policy_runner.representation == "task":
                    ik_metrics = {
                        "position_error_mm": self.ik_err_mm["r"],
                        "speed_scale": self.act_policy_ik_speed_scale,
                        "collision_constraint_violation": (
                            self.collision_constraint_violation
                        ),
                    }
                    if np.isfinite(self.collision_min_distance):
                        ik_metrics["minimum_collision_distance_m"] = (
                            self.collision_min_distance
                        )
                try:
                    rerun_logger.log(
                        self.act_policy_rerun_frame,
                        self.act_policy_observation,
                        action,
                        predicted_chunk=self.act_policy_info["predicted_chunk"],
                        task_action=self.act_policy_info.get("task_action"),
                        representation=self.act_policy_runner.representation,
                        temporal_metrics={
                            "proleptic_steps": self.act_policy_info["proleptic_steps"],
                            "target_timestep": self.act_policy_info["target_timestep"],
                            "ensemble_candidate_count": self.act_policy_info[
                                "ensemble_candidate_count"
                            ],
                        },
                        ik_metrics=ik_metrics,
                    )
                except Exception as error:
                    # Live visualization is optional. If its viewer/proxy is
                    # closed, keep control running and detach only Rerun.
                    self.act_policy_rerun_logger = None
                    rerun_logger.__exit__(None, None, None)
                    self.act_policy_status = (
                        "Rerun disconnected; policy continues without live "
                        f"logging ({type(error).__name__})"
                    )
                else:
                    self.act_policy_rerun_frame += 1
            self.act_policy_observation = self.act_policy_env.step(action)
        finally:
            # Policy cameras temporarily select the shared offscreen buffer.
            # Restore both visible context and window buffer on every path.
            render.restore_window_render_target(self)
        self.act_policy_frame += 1
        if self.act_policy_frame >= self.act_policy_max_steps:
            self.finish_act_policy("Max steps reached")
        return True

    # R/G/V/C 동작 -- 키보드(_handle_edge_keys)와 패널 버튼
    # (ui.draw_panel) 양쪽에서 똑같이 호출하는 공용 메서드.

    def reset_can(self):
        """학습과 같은 분포에서 캔만 재배치하고 로봇 자세는 유지한다."""
        self.initial_can_position = self.imitation_task.reset(
            self.data, self.rng, randomize_bin_colors=self.task_name == "can_color_sort"
        )
        return self.initial_can_position

    def observe(self):
        """현재 로봇 상태를 live MuJoCo 배열과 분리된 스냅샷으로 반환한다."""
        return state.RobotObservation.capture(self)

    def reset_active_object(self):
        """캔과 파지·가상 물체 상태를 함께 초기화해 새 작업을 시작할 준비를 한다."""
        self.reset_can()
        self.grab_state = {"r": None, "l": None}
        self.cyclo_grasp_captured = False
        self.cyclo_capture_offsets = None
        self.whole_body_solver.set_rigid_grasp(self.data, False)
        self.cyclo_controller = "movel"
        self.cyclo_status = "ready"

    def cycle_camera(self):
        """두 개의 사전 정의 카메라 프리셋을 번갈아 선택하고 렌더 카메라에 적용한다."""
        self.camera_preset = 1 - self.camera_preset
        render.set_camera_preset(self.cam, self.camera_preset)

    def toggle_collision_visualization(self):
        """충돌 거리와 최근접점을 나타내는 진단 오버레이 표시 여부를 전환한다."""
        self.collision_viz = not self.collision_viz

    def set_arm_mode(self, side, mode):
        """이 손을 "ik"(EE 포즈 슬라이더) <-> "fk"(관절각 슬라이더)로 전환한다.
        전환 순간 팔이 튀지 않도록 방향에 따라 다르게 동기화한다:

        - ik -> fk: 지금 토크 제어기가 추종 중이던 관절각(q_des)을 그대로 FK
          슬라이더 값으로 복사 -- 전환 직후 첫 스텝의 목표 관절각이 전환 직전과
          정확히 같아서 포즈 점프가 없다.
        - fk -> ik: 반대로 EE 포즈 슬라이더 쪽이 전환 전 값(어쩌면 한참 전에 마지막
          으로 IK 모드였을 때의 낡은 목표)을 그대로 들고 있으므로, 그걸 쓰는 대신
          "지금 실제 site가 있는 월드 포즈"를 홈 기준 XYZ offset/RPY로 역산해 targets/
          smoothed_pos/smoothed_rpy에 다시 채워 넣는다 -- 그래야 IK가 방금 있던
          자리에서부터 이어서 풀리지, 옛 목표로 갑자기 끌려가지 않는다.
        """
        if mode == self.arm_mode[side]:
            return
        q_des = self.q_des[side]

        if mode == "fk":
            self.fk_q_deg[side] = [math.degrees(v) for v in q_des]
        else:
            state = self.whole_body_solver.site_state(self.data, side)
            target_pos = targets.world_to_target_pos(self, side, state.position)
            rpy_deg = targets.world_quat_to_target_rpy(self, side, state.quaternion)

            self.targets[f"pos_{side}"] = target_pos
            self.targets[f"rpy_{side}"] = rpy_deg
            self.smoothed_pos[side] = np.array(target_pos)
            self.smoothed_rpy[side] = np.array(rpy_deg)

        self.arm_mode[side] = mode

    def set_whole_body_enabled(self, enabled):
        """월드 목표를 움직이지 않고 전신 IK와 팔 전용 IK를 전환한다.

        두 모드는 의도적으로 서로 다른 UI 좌표계를 사용한다. 전신 목표는 시작 월드
        기준에 머물고, 팔 전용 목표는 현재 차체를 따라간다. 따라서 전환할 때 수치 목표를
        새 좌표계로 다시 표현해야 한다. 또한 캐시된 베이스 명령을 모두 지워 OFF가 가중치
        변경 지연이 아니라 즉시 정지 요청으로 동작하게 한다.
        """
        enabled = bool(enabled)
        if enabled == self.whole_body_enabled:
            return

        hand_world_poses = {
            side: tuple(value.copy() for value in targets.target_world_pose(self, side))
            for side in SIDES
        }
        virtual_world_pose = tuple(
            value.copy() for value in targets.virtual_object_world_pose(self)
        )
        self.whole_body_enabled = enabled

        virtual_pos, virtual_quat = virtual_world_pose
        if enabled:
            self.targets["virtual_object_pos"] = targets.world_to_anchor_local_pos(
                self, virtual_pos
            ).tolist()
        else:
            self.targets["virtual_object_pos"] = targets.world_to_base_pos(
                self, virtual_pos
            ).tolist()
        self.targets["virtual_object_rpy"] = list(
            targets.world_quat_to_virtual_rpy(self, virtual_quat)
        )

        if self.cyclo_grasp_captured:
            # 양손 MoveL 모드에서는 캡처된 가상 물체가 최종 목표의 기준이다.
            self.apply_virtual_object_target()
        else:
            for side, (world_pos, world_quat) in hand_world_poses.items():
                self.targets[f"pos_{side}"] = targets.world_to_target_pos(
                    self, side, world_pos
                )
                self.targets[f"rpy_{side}"] = list(
                    targets.world_quat_to_target_rpy(self, side, world_quat)
                )

        for side in SIDES:
            self.smoothed_pos[side] = np.asarray(
                self.targets[f"pos_{side}"], dtype=float
            ).copy()
            self.smoothed_rpy[side] = np.asarray(
                self.targets[f"rpy_{side}"], dtype=float
            ).copy()

        rebased_targets = {
            side: targets.target_world_pose(self, side) for side in SIDES
        }
        self.whole_body_solver.rebase(self.data, rebased_targets)
        self.whole_body_base_twist = base.BodyTwist()
        self.commanded_base_twist = base.BodyTwist()
        targets.sync_ik_mocaps_from_targets(self)

    def toggle_whole_body_control(self):
        """현재 상태의 반대 값으로 전신 IK 활성 여부를 안전하게 전환한다."""
        self.set_whole_body_enabled(not self.whole_body_enabled)

    def capture_grasp(self):
        """Cyclo 방식의 ``/capture_grasp true`` 동작을 수행한다.

        가상 물체 마커를 기준으로 양손 목표 자세를 기록한다. 이후에는
        ``virtual_object_pos/rpy``가 명령 원점이 되고, 캡처한 오프셋으로 양손 MoveL
        목표를 계산한다.
        """
        targets.capture_grasp(self)
        self.whole_body_solver.set_rigid_grasp(self.data, True)

    def release_grasp(self):
        """Cyclo 방식의 ``/capture_grasp false``로 독립적인 양손 MoveL 목표로 돌아간다."""
        targets.release_grasp(self)
        self.whole_body_solver.set_rigid_grasp(self.data, False)

    def apply_virtual_object_target(self):
        """가상 물체 목표를 캡처된 상대 변환에 따라 양손 목표로 반영한다."""
        targets.apply_virtual_object_target(self)

    # 메인 루프

    def run(self):
        """창이 닫힐 때까지 입력·제어·물리·렌더링 프레임 루프를 실행한다.

        종료 조건을 만나면 렌더러, ImGui와 GLFW 자원을 ``render.shutdown``으로
        정리한다. 이 메서드는 앱의 최상위 blocking 실행 진입점이다.
        """
        # 매 프레임 (1) 입력 처리 (2) IK 풀기 (3) 물리 스텝 (4) 렌더링을 전부 한
        # 스레드/한 루프 안에서 순서대로 실행한다.
        try:
            while not glfw.window_should_close(self.window):
                t0 = time.perf_counter()
                io = render.begin_frame(self)

                render.handle_camera_mouse(self, io)
                self._handle_edge_keys(io)
                drive_keys = self._read_drive_and_lift_keys(io)
                ui.draw_panel(self)
                self._step_physics(drive_keys)
                render.render_scene(self)
                render.end_frame(self, t0)
        finally:
            self._close_act_policy()
            glfw.make_context_current(self.window)
            render.shutdown(self)

    def _handle_edge_keys(self, io):
        """눌렀다 뗄 때 한 번만 반응하는 R/G/V/C 유틸리티 키."""
        if io.want_capture_keyboard:
            return
        if self.act_policy_env is not None:
            if self.keys.pressed(self.window, glfw.KEY_R):
                self.reset_act_policy()
            if self.keys.pressed(self.window, glfw.KEY_SPACE):
                self.toggle_act_policy()
            if self.keys.pressed(self.window, glfw.KEY_N):
                self.request_act_policy_step()
            if self.keys.pressed(self.window, glfw.KEY_G):
                self.contact_viz = not self.contact_viz
            if self.keys.pressed(self.window, glfw.KEY_C):
                self.cycle_camera()
            return
        if self.keys.pressed(self.window, glfw.KEY_R):
            self.reset_active_object()
        if self.keys.pressed(self.window, glfw.KEY_G):
            self.contact_viz = not self.contact_viz
        if self.keys.pressed(self.window, glfw.KEY_V):
            self.toggle_collision_visualization()
        if self.keys.pressed(self.window, glfw.KEY_C):
            self.cycle_camera()

    def _read_drive_and_lift_keys(self, io):
        """주행과 리프트 키의 현재 눌림 상태를 매 프레임 읽는다.

        R/G/V/C와 달리 주행 키는 누르는 동안 계속 반응해야 하므로 엣지 입력이 아니라
        레벨 입력으로 처리한다. 위/아래 방향키는 전진·후진, 왼쪽/오른쪽 방향키는 회전,
        대괄호 키는 좌우 이동에 사용한다. WASD는 다른 MuJoCo 도구에서 익숙한 단축키와
        충돌하므로 사용하지 않는다. 좌우 이동도 회전 키나 보조키와 겹치지 않도록 별도
        대괄호 키에 배치했다.
        """
        drive_keys = {
            "w": False,
            "a": False,
            "s": False,
            "d": False,
            "left": False,
            "right": False,
        }
        lift_dir = 0.0
        if not io.want_capture_keyboard:
            drive_keys["w"] = glfw.get_key(self.window, glfw.KEY_UP) == glfw.PRESS
            drive_keys["s"] = glfw.get_key(self.window, glfw.KEY_DOWN) == glfw.PRESS
            drive_keys["left"] = glfw.get_key(self.window, glfw.KEY_LEFT) == glfw.PRESS
            drive_keys["right"] = (
                glfw.get_key(self.window, glfw.KEY_RIGHT) == glfw.PRESS
            )
            drive_keys["a"] = (
                glfw.get_key(self.window, glfw.KEY_LEFT_BRACKET) == glfw.PRESS
            )
            drive_keys["d"] = (
                glfw.get_key(self.window, glfw.KEY_RIGHT_BRACKET) == glfw.PRESS
            )
            if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
                lift_dir += 1.0
            if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
                lift_dir -= 1.0
        if lift_dir != 0.0:
            self.targets["lift"] = float(
                np.clip(
                    self.targets["lift"] + lift_dir * LIFT_JOG_SPEED * self.frame_dt,
                    LIFT_RANGE[0],
                    LIFT_RANGE[1],
                )
            )
        return drive_keys

    def _update_grasp_targets(self):
        """원터치 열기·닫기 명령을 변화율 제한해 손 슬라이더 값으로 반영한다."""
        for side in SIDES:
            state = self.grab_state[side]
            if state is None:
                continue
            desired = 1.0 if state else 0.0
            for name in (f"grasp_{side}", f"thumb_{side}"):
                delta = np.clip(
                    desired - self.targets[name],
                    -GRASP_COMMAND_RATE * self.frame_dt,
                    GRASP_COMMAND_RATE * self.frame_dt,
                )
                self.targets[name] += float(delta)

    def _smooth_hand_targets(self):
        """슬라이더 목표의 이동 속도를 IK와 제어기가 추종 가능한 범위로 제한한다."""
        for side in SIDES:
            raw_position = np.asarray(self.targets[f"pos_{side}"], dtype=float)
            position_delta = np.clip(
                raw_position - self.smoothed_pos[side],
                -MAX_POS_STEP_PER_FRAME,
                MAX_POS_STEP_PER_FRAME,
            )
            self.smoothed_pos[side] += position_delta

            raw_rpy = np.asarray(self.targets[f"rpy_{side}"], dtype=float)
            rpy_delta = np.clip(
                raw_rpy - self.smoothed_rpy[side],
                -MAX_RPY_STEP_PER_FRAME_DEG,
                MAX_RPY_STEP_PER_FRAME_DEG,
            )
            self.smoothed_rpy[side] += rpy_delta

    def _smoothed_target_poses(self):
        """변화율이 제한된 UI 목표에서 월드 좌표계 자세를 만든다."""
        return {
            side: (
                targets.target_pos_to_world_pos(self, side, self.smoothed_pos[side]),
                targets.target_rpy_to_world_quat(self, side, self.smoothed_rpy[side]),
            )
            for side in SIDES
        }

    def _step_actuators(self, command):
        """렌더링 한 프레임 동안 현재 명령 묶음을 모든 물리 서브스텝에 적용한다."""
        data = self.data
        for _ in range(self.steps_per_frame):
            for side in SIDES:
                self.arm_controllers[side].apply(data, command.arm_positions[side])
            data.ctrl[self.bindings.lift_actuator] = command.lift_position
            for wheel, (steer_angle, drive_speed) in command.wheel_commands.items():
                wheel_binding = self.bindings.wheels[wheel]
                data.ctrl[wheel_binding.steer_actuator] = steer_angle
                data.ctrl[wheel_binding.drive_actuator] = drive_speed
            for side in SIDES:
                grasp.apply_grasp(
                    self.model,
                    data,
                    grasp=command.grasp[side],
                    thumb=command.thumb[side],
                    side=side,
                )
            mujoco.mj_step(self.model, data)

    def _step_physics(self, drive_keys):
        """실제 물리 반영: target rate-limit -> world-fixed pose -> whole-body solve ->
        팔 torque/lift position/swerve/grasp actuator ctrl -> ``mj_step``. Solver는 live
        qpos를 읽기만 하며, robot qpos를 직접 쓰는 kinematic override는 없다."""
        if self._step_act_policy():
            self.last_observation = self.observe()
            return

        manual_state = control_loop.update_manual_drive(
            self,
            drive_keys,
            stop_linear_speed=MANUAL_STOP_LINEAR_SPEED,
            stop_angular_speed=MANUAL_STOP_ANGULAR_SPEED,
        )

        if self.cyclo_grasp_captured:
            self.apply_virtual_object_target()
        self._update_grasp_targets()
        self._smooth_hand_targets()

        # 전신 IK 목표는 월드 좌표계에 고정되어야 한다. 매 프레임 현재 베이스 자세에서
        # 목표를 다시 만들면 베이스와 목표가 같은 만큼 움직여 작업 오차를 줄일 수 없다.
        task_command = state.TaskCommand.create(
            self._smoothed_target_poses(),
            self.targets["lift"],
            base_twist=(
                manual_state.command if manual_state.keys_active else base.BodyTwist()
            ),
            grasp={side: self.targets[f"grasp_{side}"] for side in SIDES},
            thumb={side: self.targets[f"thumb_{side}"] for side in SIDES},
        )
        self.last_task_command = task_command

        if manual_state.carry_targets:
            # 전환 시 새 차체 자세를 공통 이동의 영점으로 다시 정의해야 한다. 그렇지 않으면
            # 시작 기준 때문에 WBIK가 관성처럼 느껴지는 역방향 명령을 낸다.
            self.whole_body_solver.rebase(self.data, task_command.hand_poses)

        control_loop.apply_whole_body_solution(
            self,
            task_command,
            sides=SIDES,
            arm_nominal=self.home_arm_positions,
        )
        wheel_cmds = control_loop.select_base_command(self, manual_state)
        control_command = control_loop.build_control_command(
            self, task_command, wheel_cmds
        )
        self.last_control_command = control_command
        self._step_actuators(control_command)
        self.last_observation = self.observe()


def _parse_args(argv):
    """텔레옵 CLI 인자를 파싱하고 설정 파일이 import 전에 적용됐는지 확인한다."""
    parser = argparse.ArgumentParser(description="FFW-SH5 teleop app")
    parser.add_argument(
        "--config",
        metavar="YAML",
        help=(
            f"설정 파일 경로. 실행 전에 {CONFIG_ENV_VAR} 환경 변수로 적용되며 "
            "src/teleop_app.py 진입점을 사용할 때 지원됩니다."
        ),
    )
    parser.add_argument(
        "--env",
        type=int,
        choices=sorted(ENV_TASK_NAMES),
        default=0,
        help=(
            "실행 환경 번호입니다: 0=기존 초록 캔→파랑 상자, "
            "1=네 색 캔 분류 (기본값: 0)."
        ),
    )
    parser.add_argument(
        "--policy-checkpoint",
        metavar="CKPT",
        help="ACT checkpoint를 현재 teleop 창에 바로 로드합니다.",
    )
    parser.add_argument(
        "--policy-stats",
        metavar="PKL",
        help="선택적인 ACT dataset_stats.pkl 경로입니다.",
    )
    parser.add_argument(
        "--policy-device", default="auto", help="ACT 추론 장치입니다 (기본값: auto)."
    )
    parser.add_argument(
        "--policy-representation",
        choices=("auto", "joint", "task"),
        default="auto",
        help=(
            "정책 입출력 표현입니다. auto는 checkpoint metadata에서 "
            "자동 판별합니다 (기본값: auto)."
        ),
    )
    parser.add_argument(
        "--policy-rerun",
        action="store_true",
        help="정책 rollout을 Rerun Viewer로 열고 .rrd 파일로 저장합니다.",
    )
    parser.add_argument(
        "--policy-rerun-port",
        type=int,
        default=9877,
        help="정책 Rerun Viewer 포트입니다 (기본값: 9877).",
    )
    parser.add_argument(
        "--policy-rerun-hz",
        type=float,
        default=DEFAULT_POLICY_RERUN_LOG_HZ,
        help=(
            "정책 제어 주기와 독립적인 Rerun 기록 주기입니다 "
            f"(기본값: {DEFAULT_POLICY_RERUN_LOG_HZ:g} Hz)."
        ),
    )
    parser.add_argument(
        "--policy-ik-speed-scale",
        type=float,
        default=DEFAULT_TASK_POLICY_IK_SPEED_SCALE,
        help=(
            "task-space 정책의 오른팔 IK pose 추종 속도 배율입니다. "
            "관절/충돌 안전 상한은 유지됩니다 (기본값: "
            f"{DEFAULT_TASK_POLICY_IK_SPEED_SCALE:g})."
        ),
    )
    parser.add_argument(
        "--policy-pte-steps",
        type=int,
        default=DEFAULT_POLICY_PTE_STEPS,
        help=(
            "현재보다 몇 control step 미래의 ACT action을 실행할지 정합니다. "
            "0은 기존 temporal ensemble입니다 (기본값: "
            f"{DEFAULT_POLICY_PTE_STEPS})."
        ),
    )
    parser.add_argument(
        "--policy-seed", type=int, default=1000, help="ACT policy UI reset seed입니다."
    )
    parser.add_argument(
        "--policy-max-steps",
        type=int,
        default=500,
        help="ACT policy UI rollout 최대 step입니다 (기본값: 500).",
    )
    args = parser.parse_args(argv)
    if args.config:
        requested = str(Path(args.config).expanduser().resolve())
        if requested != str(SETTINGS.path.resolve()):
            raise RuntimeError(
                "--config는 모듈 import 전에 적용해야 합니다. "
                "python3 src/teleop_app.py --config <파일>로 실행하세요."
            )
    return args


def main(argv=None):
    """명령행 인자를 검사한 뒤 :class:`TeleopApp`을 생성하고 메인 루프를 시작한다."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    TeleopApp(
        policy_checkpoint=args.policy_checkpoint,
        policy_stats=args.policy_stats,
        policy_device=args.policy_device,
        policy_representation=args.policy_representation,
        policy_rerun=args.policy_rerun,
        policy_rerun_port=args.policy_rerun_port,
        policy_rerun_hz=args.policy_rerun_hz,
        policy_ik_speed_scale=args.policy_ik_speed_scale,
        policy_pte_steps=args.policy_pte_steps,
        policy_seed=args.policy_seed,
        policy_max_steps=args.policy_max_steps,
        task_name=ENV_TASK_NAMES[args.env],
    ).run()


if __name__ == "__main__":
    main()
