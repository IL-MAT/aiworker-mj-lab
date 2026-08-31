"""MuJoCo geometry 거리와 기구학 트리 Jacobian을 결합한 충돌 제약 계산.

트리 FK와 달리 geometry 최근접점과 contact normal은 현재 물리 상태에 의존하므로
이 모듈만 live ``MjData``를 읽는다. 계산 결과는 whole-body IK의 CBF와 렌더링 진단이
같이 사용한다.
"""

import itertools
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class CollisionPair:
    """감시할 geometry 쌍과 거리 계산 방식."""

    name: str
    geom_a: int
    geom_b: int
    mode: str = "geom"


@dataclass(frozen=True)
class CollisionConstraint:
    """signed distance, controlled-DOF gradient와 두 최근접점."""

    name: str
    distance: float
    gradient: np.ndarray
    point_a: np.ndarray
    point_b: np.ndarray


def collision_distance_gradient(
    model, data, pair, tree, joint_ids, max_distance, frame_cache=None
):
    """거리 제한 안의 geometry 쌍에 대한 signed-distance gradient를 반환한다.

    분리된 geometry에서 미분은 ``n.T @ (J_b - J_a)``이다. 여기서 n은 A에서 B로
    향한다. 침투 상태에서는 MuJoCo의 최근접점 선분 방향이 뒤집히므로 gradient의
    부호도 함께 보정한다.
    """
    max_distance = max(float(max_distance), 0.0)
    if pair.mode == "table_top":
        return _table_top_distance_gradient(
            model, data, pair, tree, joint_ids, max_distance, frame_cache
        )
    if pair.mode == "bounding_sphere":
        return _bounding_sphere_distance_gradient(
            model, data, pair, tree, joint_ids, max_distance, frame_cache
        )

    fromto = np.zeros(6)
    raw_distance = float(
        mujoco.mj_geomDistance(
            model, data, pair.geom_a, pair.geom_b, max_distance, fromto
        )
    )
    point_a, point_b = fromto[:3].copy(), fromto[3:].copy()
    segment = point_b - point_a
    segment_length = float(np.linalg.norm(segment))

    # 거리 제한 밖이면 MuJoCo가 fromto를 0으로 둔 채 distmax를 반환한다.
    if raw_distance >= max_distance - 1e-12 and segment_length < 1e-12:
        return None
    distance = raw_distance
    # 일부 convex mesh/box 조합은 떨어져 있어도 0을 반환하므로 선분 길이로 보정한다.
    if abs(raw_distance) < 1e-12 and segment_length > 1e-7:
        distance = segment_length
    if distance > max_distance:
        return None

    if segment_length > 1e-10:
        normal = segment / segment_length
    else:
        normal = _contact_normal(data, pair.geom_a, pair.geom_b)
        if normal is None:
            center_delta = data.geom_xpos[pair.geom_b] - data.geom_xpos[pair.geom_a]
            center_norm = float(np.linalg.norm(center_delta))
            normal = (
                center_delta / center_norm
                if center_norm > 1e-10
                else np.array([1.0, 0.0, 0.0])
            )
        point_a = data.geom_xpos[pair.geom_a].copy()
        point_b = data.geom_xpos[pair.geom_b].copy()

    body_a = int(model.geom_bodyid[pair.geom_a])
    body_b = int(model.geom_bodyid[pair.geom_b])
    jacobian_a = tree.point_jacobian(data.qpos, body_a, point_a, joint_ids, frame_cache)
    jacobian_b = tree.point_jacobian(data.qpos, body_b, point_b, joint_ids, frame_cache)
    gradient = normal @ (jacobian_b - jacobian_a)
    if raw_distance < 0.0:
        gradient *= -1.0
    if not np.isfinite(distance) or not np.isfinite(gradient).all():
        return None
    return CollisionConstraint(
        pair.name, float(distance), np.asarray(gradient, dtype=float), point_a, point_b
    )


def default_collision_pairs(model):
    """손 접촉과 바퀴 운동을 막지 않는 whole-body 감시 쌍을 구성한다."""
    collision_geoms_by_body = {}
    for geom_id in range(model.ngeom):
        if int(model.geom_contype[geom_id]) == 0:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        collision_geoms_by_body.setdefault(body_id, []).append(geom_id)

    body_geom = {}
    for body_name in (
        *(f"arm_{side}_link{i}" for side in ("r", "l") for i in range(1, 8)),
        "hx5_r_base",
        "hx5_l_base",
        "base_link",
        "lift_link",
        "arm_base_link",
        "head_link1",
        "head_link2",
    ):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue
        if body_name == "base_link":
            proxy_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "base_chassis_cbf_proxy"
            )
            if proxy_id >= 0:
                body_geom[body_name] = proxy_id
                continue
        candidates = collision_geoms_by_body.get(body_id, []).copy()
        if candidates:
            # group 3 collision mesh를 우선하되 base의 group 0 box도 fallback으로 남긴다.
            candidates.sort(key=lambda geom_id: int(model.geom_group[geom_id]) != 3)
            body_geom[body_name] = candidates[0]

    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    pairs = {}

    def add(kind, body_a, body_b, geom_b=None, mode="geom"):
        """유효하고 중복되지 않은 geom 조합을 이름 붙인 충돌 쌍으로 등록한다."""
        geom_a = body_geom.get(body_a)
        other = body_geom.get(body_b) if geom_b is None else geom_b
        if geom_a is None or other is None or geom_a == other:
            return
        key = tuple(sorted((int(geom_a), int(other))))
        if key in pairs:
            return
        if mode == "table_top":
            pairs[key] = CollisionPair(
                f"{kind}:{body_a}/{body_b}", int(geom_a), int(other), mode
            )
        else:
            pairs[key] = CollisionPair(f"{kind}:{body_a}/{body_b}", *key, mode)

    arms = {
        side: [f"arm_{side}_link{i}" for i in range(1, 8)] + [f"hx5_{side}_base"]
        for side in ("r", "l")
    }
    # 양팔 조합에서 구조상 늘 가까운 shoulder link1은 제외한다.
    for body_a, body_b in itertools.product(arms["r"][1:], arms["l"][1:]):
        mode = (
            "bounding_sphere"
            if body_a.startswith("hx5_") and body_b.startswith("hx5_")
            else "geom"
        )
        add("cross-arm", body_a, body_b, mode=mode)

    # 같은 팔에서는 중간에 body가 두 개 이상 있는 비인접 link만 감시한다.
    for side in ("r", "l"):
        for first, second in itertools.combinations(range(len(arms[side])), 2):
            if second - first >= 3:
                add("folded-arm", arms[side][first], arms[side][second])

    central_bodies = (
        "base_link",
        "lift_link",
        "arm_base_link",
        "head_link1",
        "head_link2",
    )
    for side in ("r", "l"):
        for arm_body, central_body in itertools.product(arms[side][2:], central_bodies):
            add("body", arm_body, central_body)

    if table_id >= 0:
        for side in ("r", "l"):
            for arm_body in arms[side][1:]:
                mode = "table_top" if arm_body.startswith("hx5_") else "geom"
                add("workspace", arm_body, "table", table_id, mode)

    # The shelf-sort scene is activated at runtime by switching its collision
    # bits from 4/0 to 1/1. Physics still uses every proxy, while the real-time
    # CBF monitors only horizontal work surfaces to keep IK responsive.
    shelf_geoms = []
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if (
            int(model.geom_contype[geom_id]) != 0
            and int(model.geom_conaffinity[geom_id]) != 0
            and (
                geom_name.startswith("source_shelf_level_")
                or geom_name.endswith("_top")
                and geom_name.startswith("side_table_")
            )
        ):
            shelf_geoms.append((geom_name, geom_id))
    for side in ("r", "l"):
        for geom_name, geom_id in shelf_geoms:
            add("shelf", f"hx5_{side}_base", geom_name, geom_id)
    return tuple(pairs.values())


def _table_top_distance_gradient(
    model, data, pair, tree, joint_ids, max_distance, frame_cache=None
):
    """유한한 tabletop 위쪽에 대한 연속적인 support-point clearance를 계산한다."""
    robot_geom, table_geom = pair.geom_a, pair.geom_b
    robot_rotation = data.geom_xmat[robot_geom].reshape(3, 3)
    table_rotation = data.geom_xmat[table_geom].reshape(3, 3)
    robot_local_center = model.geom_aabb[robot_geom, :3]
    robot_half_size = model.geom_aabb[robot_geom, 3:]
    table_local_center = model.geom_aabb[table_geom, :3]
    table_half_size = model.geom_aabb[table_geom, 3:]
    robot_center = data.geom_xpos[robot_geom] + robot_rotation @ robot_local_center
    table_center = data.geom_xpos[table_geom] + table_rotation @ table_local_center
    table_normal = table_rotation[:, 2]

    # tabletop의 유한한 XY footprint 바깥에서는 무한 평면처럼 동작하지 않는다.
    robot_in_table = table_rotation.T @ (robot_center - table_center)
    relative_rotation = table_rotation.T @ robot_rotation
    robot_extent_table = np.abs(relative_rotation) @ robot_half_size
    for axis in (0, 1):
        gap = abs(robot_in_table[axis]) - (
            table_half_size[axis] + robot_extent_table[axis]
        )
        if gap > max_distance:
            return None

    normal_local = robot_rotation.T @ table_normal
    support_local = robot_local_center - np.sign(normal_local) * robot_half_size
    point_robot = data.geom_xpos[robot_geom] + robot_rotation @ support_local
    point_table = point_robot - table_normal * (
        table_normal @ (point_robot - table_center) - table_half_size[2]
    )
    distance = float(table_normal @ (point_robot - table_center) - table_half_size[2])
    if distance > max_distance:
        return None

    jacobian_robot = tree.point_jacobian(
        data.qpos,
        int(model.geom_bodyid[robot_geom]),
        point_robot,
        joint_ids,
        frame_cache,
    )
    jacobian_table = tree.point_jacobian(
        data.qpos,
        int(model.geom_bodyid[table_geom]),
        point_table,
        joint_ids,
        frame_cache,
    )
    gradient = table_normal @ (jacobian_robot - jacobian_table)
    return CollisionConstraint(
        pair.name,
        distance,
        np.asarray(gradient, dtype=float),
        point_robot.copy(),
        point_table.copy(),
    )


def _bounding_sphere_distance_gradient(
    model, data, pair, tree, joint_ids, max_distance, frame_cache=None
):
    """palm box끼리의 불연속을 피하기 위한 보수적 bounding-sphere 거리."""
    geom_a, geom_b = pair.geom_a, pair.geom_b
    rotation_a = data.geom_xmat[geom_a].reshape(3, 3)
    rotation_b = data.geom_xmat[geom_b].reshape(3, 3)
    center_a = data.geom_xpos[geom_a] + rotation_a @ model.geom_aabb[geom_a, :3]
    center_b = data.geom_xpos[geom_b] + rotation_b @ model.geom_aabb[geom_b, :3]
    radius_a = float(np.linalg.norm(model.geom_aabb[geom_a, 3:]))
    radius_b = float(np.linalg.norm(model.geom_aabb[geom_b, 3:]))
    delta = center_b - center_a
    center_distance = float(np.linalg.norm(delta))
    normal = (
        np.array([1.0, 0.0, 0.0])
        if center_distance < 1e-10
        else delta / center_distance
    )
    distance = center_distance - radius_a - radius_b
    if distance > max_distance:
        return None

    jacobian_a = tree.point_jacobian(
        data.qpos, int(model.geom_bodyid[geom_a]), center_a, joint_ids, frame_cache
    )
    jacobian_b = tree.point_jacobian(
        data.qpos, int(model.geom_bodyid[geom_b]), center_b, joint_ids, frame_cache
    )
    gradient = normal @ (jacobian_b - jacobian_a)
    return CollisionConstraint(
        pair.name,
        float(distance),
        np.asarray(gradient, dtype=float),
        center_a + radius_a * normal,
        center_b - radius_b * normal,
    )


def _contact_normal(data, geom_a, geom_b):
    """현재 contact에서 geom A→B 방향의 normal을 찾는다."""
    for contact in data.contact:
        first, second = int(contact.geom1), int(contact.geom2)
        if first == geom_a and second == geom_b:
            return np.asarray(contact.frame[:3], dtype=float).copy()
        if first == geom_b and second == geom_a:
            return -np.asarray(contact.frame[:3], dtype=float).copy()
    return None
