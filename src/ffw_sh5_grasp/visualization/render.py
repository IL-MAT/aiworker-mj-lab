"""``teleop_app.TeleopApp``의 렌더링과 뷰포트 상호작용 보조 함수.

시각적으로 중요하지만 텔레옵 상태, IK, 파지와 베이스 주행에는 독립적인
GLFW/ImGui/MuJoCo 렌더링 연결을 담당한다. 순환 import를 피하기 위해 ``TeleopApp``을
직접 import하지 않고 ``teleop_app``이 전달한 ``app`` 객체를 사용한다. 이는
``visualization/ui.py``와 같은 덕 타이핑 경계다.
"""

import ctypes
import sys
import time

import glfw

# Linux에서는 호환되는 GLX 문맥을 선택하도록 ``glfw.init()`` 전에 X11을 지정한다.
# macOS에는 X11 backend가 없으므로 GLFW가 Cocoa backend를 자동 선택하게 둔다.
if sys.platform.startswith("linux"):
    glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)

import mujoco
import numpy as np
from imgui_bundle import imgui, imguizmo

from ..application import targets
from ..config import SETTINGS
from ..kinematics import rotations

CAMERA_PRESETS = (
    SETTINGS.get("render.camera_presets.overview"),
    SETTINGS.get("render.camera_presets.hand_closeup"),
)
# 렌더링 스타일과 내부 버퍼 크기는 로봇 제어 튜닝값이 아니므로 구현과 함께 고정한다.
MOUSE_ZOOM_SCALE = 0.05
GIZMO_SIZE = 0.18
MAX_SCENE_GEOMETRIES = 10_000
COLLISION_GEOMETRY_RGBA = np.array([0.05, 0.75, 1.0, 0.28], dtype=np.float32)
PENETRATION_RGBA = np.array([1.0, 0.02, 0.02, 1.0], dtype=np.float32)
UNSAFE_RGBA = np.array([1.0, 0.18, 0.02, 1.0], dtype=np.float32)
BUFFER_RGBA = np.array([1.0, 0.78, 0.05, 1.0], dtype=np.float32)
COLLISION_POINT_RADIUS = 0.007
COLLISION_LINE_WIDTH = 4.0
FREQUENCY_EMA_PREVIOUS_WEIGHT = 0.9


def set_camera_preset(cam, preset):
    """YAML에 정의된 카메라 프리셋의 시점·거리·방위·고도를 MuJoCo 카메라에 적용한다."""
    settings = CAMERA_PRESETS[0 if preset == 0 else 1]
    cam.lookat[:] = settings["lookat"]
    cam.distance = float(settings["distance"])
    cam.azimuth = float(settings["azimuth_deg"])
    cam.elevation = float(settings["elevation_deg"])


def setup_render(app, window_w, window_h):
    """GLFW 창, ImGui 다중 뷰포트와 MuJoCo 렌더링 자원을 생성해 ``app``에 연결한다.

    초기화 단계가 실패하면 이미 만든 하위 자원을 역순으로 정리하고 예외를 발생시킨다.
    성공하면 scene, camera, option, perturbation과 GPU context를 사용할 수 있다.
    """
    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    if sys.platform == "darwin":
        # MuJoCo expects the default macOS compatibility context; ImGui's
        # shader must therefore target the GLSL version exposed by it.
        glsl_version = "#version 120"
    else:
        glsl_version = "#version 130"
    # ``mujoco.Renderer``의 GLFW offscreen context는 숨김 창을 만들기 위해
    # GLFW_VISIBLE=false window hint를 설정한다. Hint는 다음 create_window()에도
    # 남아 있으므로 policy camera를 먼저 만든 recorder에서는 주 창까지 UnMapped
    # 상태가 됐다. 사용자 창은 항상 보이도록 생성 직전에 명시적으로 복원한다.
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    window = glfw.create_window(window_w, window_h, "FFW-SH5 Teleop", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("glfw.create_window() failed")
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    app.window = window
    app.window_h = window_h

    imgui.create_context()
    io = imgui.get_io()
    io.config_flags |= imgui.ConfigFlags_.viewports_enable
    window_address = ctypes.cast(window, ctypes.c_void_p).value
    if not imgui.backends.glfw_init_for_opengl(window_address, True):
        imgui.destroy_context()
        glfw.destroy_window(window)
        glfw.terminate()
        raise RuntimeError("ImGui GLFW backend initialization failed")
    if not imgui.backends.opengl3_init(glsl_version):
        imgui.backends.glfw_shutdown()
        imgui.destroy_context()
        glfw.destroy_window(window)
        glfw.terminate()
        raise RuntimeError("ImGui OpenGL3 backend initialization failed")
    app.imgui_multi_viewport = True

    app.scene = mujoco.MjvScene(app.model, maxgeom=MAX_SCENE_GEOMETRIES)
    app.cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(app.cam)
    set_camera_preset(app.cam, 0)
    app.opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(app.opt)
    # Group 4 is hidden only from policy-camera rendering because the imported
    # head mesh encloses the calibrated optical origin. The operator view must
    # continue to display the physical head housing.
    app.opt.geomgroup[4] = 1
    # Group 5 contains Gizmo targets and diagnostic sites. They remain useful
    # in this operator view but policy-camera observations explicitly hide it.
    app.opt.geomgroup[5] = 1
    app.opt.sitegroup[5] = 1
    app.pert = mujoco.MjvPerturb()
    app.context = mujoco.MjrContext(app.model, mujoco.mjtFontScale.mjFONTSCALE_150)


def begin_frame(app):
    """운영체제 이벤트를 처리하고 새 ImGui 프레임을 시작한 뒤 입력 상태를 반환한다."""
    glfw.poll_events()
    imgui.backends.opengl3_new_frame()
    imgui.backends.glfw_new_frame()
    imgui.new_frame()
    return imgui.get_io()


def restore_window_render_target(app):
    """Restore the visible GLFW context and MuJoCo window framebuffer.

    Policy cameras use ``mujoco.Renderer``, which makes a hidden offscreen GL
    context current. Restoring only the GLFW context can leave subsequent
    MuJoCo drawing associated with an offscreen render target on some drivers,
    producing a dark or partially cleared operator window.
    """
    glfw.make_context_current(app.window)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, app.context)


def shutdown(app):
    """ImGui backend·다중 창·주 GLFW 창을 안전한 역순으로 종료한다."""
    imgui.destroy_platform_windows()
    imgui.backends.opengl3_shutdown()
    imgui.backends.glfw_shutdown()
    imgui.destroy_context()
    glfw.destroy_window(app.window)
    glfw.terminate()


def _move_camera(app, action, relative_x, relative_y):
    """MuJoCo 버전에 맞는 ``mjv_moveCamera`` 서명으로 free camera를 움직인다.

    MuJoCo 3.11은 더 이상 ``MjvScene`` 인자를 받지 않지만 이전 버전은 scene을
    camera 앞에 요구한다. 현재 서명을 먼저 사용하고 구버전에서만 fallback한다.
    """
    try:
        mujoco.mjv_moveCamera(app.model, action, relative_x, relative_y, app.cam)
    except TypeError as current_error:
        try:
            mujoco.mjv_moveCamera(
                app.model, action, relative_x, relative_y, app.scene, app.cam
            )
        except TypeError:
            raise current_error


def handle_camera_mouse(app, io):
    """UI와 기즈모가 사용하지 않은 마우스 드래그·휠 입력으로 MuJoCo 카메라를 조작한다."""
    cur_mouse = list(glfw.get_cursor_pos(app.window))
    dx, dy = cur_mouse[0] - app.last_mouse[0], cur_mouse[1] - app.last_mouse[1]
    app.last_mouse = cur_mouse
    if io.want_capture_mouse or app.gizmo_mouse_active:
        return

    left = glfw.get_mouse_button(app.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    right = glfw.get_mouse_button(app.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    middle = glfw.get_mouse_button(app.window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    if left or right or middle:
        _, win_h = glfw.get_window_size(app.window)
        mod_shift = (
            glfw.get_key(app.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(app.window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if right:
            action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H
                if mod_shift
                else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
        elif left:
            action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H
                if mod_shift
                else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        _move_camera(app, action, dx / win_h, dy / win_h)
    if io.mouse_wheel != 0:
        _move_camera(
            app, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -MOUSE_ZOOM_SCALE * io.mouse_wheel
        )


def pose_to_imguizmo_matrix(world_pos, world_quat):
    """월드 위치와 wxyz 쿼터니언을 ImGuizmo 열 우선 4×4 변환 행렬로 바꾼다."""
    mat = np.eye(4)
    mat[:3, :3] = rotations.rotation_from_quaternion(world_quat)
    mat[:3, 3] = world_pos
    return imguizmo.im_guizmo.Matrix16(mat.astype(float).reshape(16, order="F"))


def imguizmo_matrix_to_pose(matrix):
    """ImGuizmo 열 우선 변환 행렬에서 월드 위치와 wxyz 쿼터니언을 복원한다."""
    mat = np.array(matrix.values, dtype=float).reshape((4, 4), order="F")
    world_pos = mat[:3, 3].copy()
    world_quat = rotations.quaternion_from_rotation(mat[:3, :3])
    return world_pos, world_quat


def _imguizmo_camera_matrices(app, viewport):
    """MuJoCo 렌더 카메라를 ImGuizmo가 요구하는 view·projection 행렬로 변환한다."""
    glcam = app.scene.camera[0]
    forward = np.array(glcam.forward, dtype=float)
    forward /= max(np.linalg.norm(forward), 1e-9)
    up = np.array(glcam.up, dtype=float)
    up /= max(np.linalg.norm(up), 1e-9)
    right = np.cross(forward, up)
    right /= max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, forward)
    pos = np.array(glcam.pos, dtype=float)

    view = np.eye(4)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = -forward
    view[0, 3] = -np.dot(right, pos)
    view[1, 3] = -np.dot(up, pos)
    view[2, 3] = np.dot(forward, pos)

    near = float(glcam.frustum_near)
    far = float(glcam.frustum_far)
    top = float(glcam.frustum_top)
    bottom = float(glcam.frustum_bottom)
    aspect = viewport.width / max(1.0, float(viewport.height))
    right_f = top * aspect
    left_f = bottom * aspect
    proj = np.zeros((4, 4))
    proj[0, 0] = 2.0 * near / (right_f - left_f)
    proj[0, 2] = (right_f + left_f) / (right_f - left_f)
    proj[1, 1] = 2.0 * near / (top - bottom)
    proj[1, 2] = (top + bottom) / (top - bottom)
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -(2.0 * far * near) / (far - near)
    proj[3, 2] = -1.0

    return (
        imguizmo.im_guizmo.Matrix16(view.reshape(16, order="F")),
        imguizmo.im_guizmo.Matrix16(proj.reshape(16, order="F")),
    )


def draw_transform_gizmo(app, viewport):
    """MuJoCo 주 뷰포트의 활성 목표 위에 ImGuizmo를 그린다.

    ``MjrRect``는 프레임버퍼 로컬 픽셀을 사용하지만 ImGui 다중 뷰포트 draw list는
    데스크톱 논리 좌표를 사용한다. 여기서 ``viewport.left/bottom``을 사용하면 주 GLFW
    창이 데스크톱 원점에서 벗어나는 순간 기즈모가 렌더링 목표와 어긋난다. 카메라
    종횡비는 MuJoCo 프레임버퍼에서 가져오되 그리기 기준은 ImGui 주 뷰포트 사각형으로
    잡는다.
    """
    target = targets.active_gizmo_target(app)
    world_pos, world_quat = targets.gizmo_target_world_pose(app, target)
    object_matrix = pose_to_imguizmo_matrix(world_pos, world_quat)
    view_matrix, proj_matrix = _imguizmo_camera_matrices(app, viewport)
    main_viewport = imgui.get_main_viewport()

    gizmo = imguizmo.im_guizmo
    gizmo.begin_frame()
    gizmo.set_drawlist(imgui.get_foreground_draw_list(main_viewport))
    gizmo.set_rect(
        float(main_viewport.pos.x),
        float(main_viewport.pos.y),
        float(main_viewport.size.x),
        float(main_viewport.size.y),
    )
    gizmo.set_orthographic(False)
    gizmo.set_gizmo_size_clip_space(GIZMO_SIZE)
    changed_translate = gizmo.manipulate(
        view_matrix,
        proj_matrix,
        gizmo.OPERATION.translate,
        gizmo.MODE.world,
        object_matrix,
    )
    changed_rotate = gizmo.manipulate(
        view_matrix,
        proj_matrix,
        gizmo.OPERATION.rotate,
        gizmo.MODE.local,
        object_matrix,
    )
    app.gizmo_mouse_active = bool(gizmo.is_using_any() or gizmo.is_over())
    if changed_translate or changed_rotate:
        new_pos, new_quat = imguizmo_matrix_to_pose(object_matrix)
        targets.set_gizmo_target_world_pose(app, target, new_pos, new_quat)


def collision_visualization_data(app):
    """현재 CBF가 실제로 감시하는 활성 충돌 쌍을 반환한다."""
    if not getattr(app, "collision_viz", False):
        return ()
    solver = getattr(app, "whole_body_solver", None)
    if solver is None:
        return ()
    return solver.collision_distances(app.data)


def _collision_color(distance, safe_distance):
    """충돌 거리의 관통·위험·buffer 상태에 대응하는 RGBA 색상 복사본을 반환한다."""
    if distance <= 0.0:
        return PENETRATION_RGBA.copy()
    if distance < safe_distance:
        return UNSAFE_RGBA.copy()
    return BUFFER_RGBA.copy()


def _append_visual_geom(scene, geom_type, size, pos, mat, rgba):
    """scene 용량 안에서 장식용 geom 하나를 추가하고 생성된 geom을 반환한다."""
    if scene.ngeom >= scene.maxgeom:
        return None
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.asarray(size, dtype=float),
        np.asarray(pos, dtype=float),
        np.asarray(mat, dtype=float).reshape(9),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1
    return geom


def _append_collision_overlay(app, constraints):
    """충돌 mesh에 색을 입히고 최근접점과 두 점을 잇는 선분을 그린다."""
    for index in range(app.scene.ngeom):
        geom = app.scene.geoms[index]
        if (
            int(geom.objtype) == int(mujoco.mjtObj.mjOBJ_GEOM)
            and 0 <= int(geom.objid) < app.model.ngeom
            and int(app.model.geom_group[int(geom.objid)]) == 3
        ):
            geom.rgba[:] = COLLISION_GEOMETRY_RGBA
            geom.transparent = 1

    identity = np.eye(3)
    safe_distance = app.whole_body_solver.collision_safe_distance
    for constraint in constraints:
        color = _collision_color(constraint.distance, safe_distance)
        for point in (constraint.point_a, constraint.point_b):
            _append_visual_geom(
                app.scene,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                [COLLISION_POINT_RADIUS] * 3,
                point,
                identity,
                color,
            )
        line = _append_visual_geom(
            app.scene,
            mujoco.mjtGeom.mjGEOM_LINE,
            [0.0, 0.0, 0.0],
            np.zeros(3),
            identity,
            color,
        )
        if line is not None:
            mujoco.mjv_connector(
                line,
                mujoco.mjtGeom.mjGEOM_LINE,
                COLLISION_LINE_WIDTH,
                constraint.point_a,
                constraint.point_b,
            )


def render_scene(app):
    """목표 마커·충돌 오버레이·MuJoCo scene·ImGui 다중 창을 한 프레임 렌더링한다."""
    restore_window_render_target(app)
    targets.sync_ik_mocaps_from_targets(app)
    app.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = app.contact_viz
    app.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = app.contact_viz
    # 모델 충돌 형상은 3번 그룹에 있고 MuJoCo 기본 옵션에서는 숨겨진다. CBF 겹침
    # 표시는 같은 토글을 사용하지만 접촉력 시각화와는 독립적이다.
    app.opt.geomgroup[3] = bool(getattr(app, "collision_viz", False))
    fb_w, fb_h = glfw.get_framebuffer_size(app.window)
    viewport = mujoco.MjrRect(0, 0, fb_w, fb_h)
    mujoco.mjv_updateScene(
        app.model,
        app.data,
        app.opt,
        app.pert,
        app.cam,
        mujoco.mjtCatBit.mjCAT_ALL,
        app.scene,
    )
    collision_data = collision_visualization_data(app)
    if app.collision_viz:
        _append_collision_overlay(app, collision_data)
    mujoco.mjr_render(viewport, app.scene, app.context)
    draw_transform_gizmo(app, viewport)

    imgui.render()
    imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())
    if imgui.get_io().config_flags & imgui.ConfigFlags_.viewports_enable:
        main_context = glfw.get_current_context()
        imgui.update_platform_windows()
        imgui.render_platform_windows_default()
        glfw.make_context_current(main_context)
    glfw.swap_buffers(app.window)


def end_frame(app, t0):
    """프레임 주파수 EMA를 갱신하고 목표 제어 주기를 넘지 않도록 남은 시간을 쉰다."""
    elapsed = time.perf_counter() - t0
    current_weight = 1.0 - FREQUENCY_EMA_PREVIOUS_WEIGHT
    app.freq_ema = FREQUENCY_EMA_PREVIOUS_WEIGHT * app.freq_ema + current_weight * (
        1.0 / max(elapsed, 1e-6)
    )
    sleep_time = app.frame_dt - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)
