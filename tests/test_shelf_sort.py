"""Regression tests for the three-tier shelf sorting workspace."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from ffw_sh5_grasp.imitation.simulation.environment import enable_task_collisions
from ffw_sh5_grasp.imitation.simulation.task import create_task
from ffw_sh5_grasp.kinematics.collision import default_collision_pairs
from ffw_sh5_grasp.visualization import render as teleop_render

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "full_scene.xml"


def _shelf_fixture(seed=0):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home)
    task = create_task(model, "shelf_color_sort")
    task.collision_geom_ids = enable_task_collisions(model, task.collision_body_names)
    task.reset(data, np.random.default_rng(seed))
    return model, data, task


def test_shelf_layout_and_color_targets():
    model, data, task = _shelf_fixture(seed=1)
    source = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "source_shelf")
    red = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "side_table_red")
    blue = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "side_table_blue")

    assert model.body_pos[source, 0] > model.body_pos[red, 0]
    assert model.body_pos[red, 1] > 0.0  # robot-left
    assert model.body_pos[blue, 1] < 0.0  # robot-right
    assert np.allclose(model.body_pos[red, :2], [0.39, 0.62])
    assert np.allclose(model.body_pos[blue, :2], [0.39, -0.62])
    assert task.variant_names == ("red", "blue")
    assert task.bin_color_layout == {
        "red": "target_bin_red",
        "blue": "target_bin",
    }

    source_levels = [
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"source_shelf_level_{index}"
        )
        for index in range(1, 4)
    ]
    assert all(geom_id >= 0 for geom_id in source_levels)
    assert np.all(np.diff(model.geom_pos[source_levels, 2]) > 0.3)
    assert (
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "source_shelf_lip_2"
        )
        == -1
    )
    assert np.isclose(data.qpos[task.can_qpos + 2], 0.835)

    for prefix in ("target_bin", "target_bin_red"):
        floor = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_floor"
        )
        wall = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_front"
        )
        assert np.allclose(model.geom_size[floor], [0.14, 0.14, 0.0075])
        assert np.allclose(model.geom_size[wall], [0.14, 0.005, 0.08])
    assert np.allclose(task.inner_half_extents, [0.13, 0.13])


def test_base_polygon_collision_fits_between_side_tables():
    model, data, _task = _shelf_fixture(seed=5)
    base_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "base_chassis_collision"
    )
    base_cbf_proxy = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "base_chassis_cbf_proxy"
    )
    red_top = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "side_table_red_top"
    )
    blue_top = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "side_table_blue_top"
    )
    assert model.geom_type[base_geom] == mujoco.mjtGeom.mjGEOM_MESH
    assert model.geom_group[base_geom] == 3
    assert model.geom_contype[base_geom] == 1
    assert model.geom_contype[base_cbf_proxy] == 0

    mesh_id = model.geom_dataid[base_geom]
    vertex_start = model.mesh_vertadr[mesh_id]
    vertex_end = vertex_start + model.mesh_vertnum[mesh_id]
    vertices = model.mesh_vert[vertex_start:vertex_end]
    world_vertices = data.geom_xpos[base_geom] + (
        vertices @ data.geom_xmat[base_geom].reshape(3, 3).T
    )
    base_width = np.ptp(world_vertices[:, 1])
    red_inner_edge = data.geom_xpos[red_top, 1] - model.geom_size[red_top, 1]
    blue_inner_edge = data.geom_xpos[blue_top, 1] + model.geom_size[blue_top, 1]
    aisle_width = red_inner_edge - blue_inner_edge
    assert np.isclose(base_width, 0.62, atol=1e-6)
    assert np.isclose(aisle_width, 0.70, atol=1e-6)
    assert aisle_width - base_width >= 0.079

    base_cbf_pairs = [
        pair
        for pair in default_collision_pairs(model)
        if pair.name.startswith("body:") and pair.name.endswith("/base_link")
    ]
    assert len(base_cbf_pairs) == 12
    assert all(base_cbf_proxy in (pair.geom_a, pair.geom_b) for pair in base_cbf_pairs)
    assert all(base_geom not in (pair.geom_a, pair.geom_b) for pair in base_cbf_pairs)

    contact_pairs = {
        frozenset((int(data.contact[index].geom1), int(data.contact[index].geom2)))
        for index in range(data.ncon)
    }
    assert frozenset((base_geom, red_top)) not in contact_pairs
    assert frozenset((base_geom, blue_top)) not in contact_pairs


def test_shelf_support_and_success_metric():
    model, data, task = _shelf_fixture(seed=4)
    initial = data.qpos[task.can_qpos : task.can_qpos + 3].copy()
    for _ in range(1_500):
        mujoco.mj_step(model, data)
    settled = data.qpos[task.can_qpos : task.can_qpos + 3].copy()
    assert np.linalg.norm(settled - initial) < 0.001

    data.qpos[task.can_qpos : task.can_qpos + 3] = data.site_xpos[task.target_site]
    data.qvel[task.can_dof : task.can_dof + 6] = 0.0
    for _ in range(500):
        mujoco.mj_step(model, data)
    assert task.metrics(data).success


def test_shelf_side_proxies_have_physical_collision():
    model, data, task = _shelf_fixture(seed=0)
    can_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "can_geom")
    for suffix in ("left", "right"):
        side_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"source_shelf_side_{suffix}"
        )
        data.qpos[task.can_qpos : task.can_qpos + 3] = model.geom_pos[side_geom]
        data.qpos[task.can_qpos] += model.body_pos[
            model.geom_bodyid[side_geom], 0
        ]
        data.qpos[task.can_qpos + 1] += model.body_pos[
            model.geom_bodyid[side_geom], 1
        ]
        data.qpos[task.can_qpos + 1] += np.copysign(
            model.geom_size[side_geom, 1] + model.geom_size[can_geom, 0] - 0.002,
            model.geom_pos[side_geom, 1],
        )
        data.qpos[task.can_qpos + 3 : task.can_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[task.can_dof : task.can_dof + 6] = 0.0
        mujoco.mj_forward(model, data)
        contact_pairs = {
            frozenset((int(data.contact[index].geom1), int(data.contact[index].geom2)))
            for index in range(data.ncon)
        }
        assert frozenset((can_geom, side_geom)) in contact_pairs


def test_shelf_scene_selection_and_collision_catalog():
    model, _data, task = _shelf_fixture(seed=2)
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    assert model.geom_rgba[table, 3] == 0.0
    assert model.geom_contype[table] == 0

    shelf_pairs = [
        pair
        for pair in default_collision_pairs(model)
        if pair.name.startswith("shelf:")
    ]
    assert len(shelf_pairs) == 2 * 5
    assert {task.object_variant, task.target_label} <= {"red", "blue"}


def test_v_collision_group_contains_complete_shelf_tables_and_bins():
    model, data, task = _shelf_fixture(seed=3)
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    teleop_render._set_collision_view_groups(option, True)
    assert not option.geomgroup[2]
    assert option.geomgroup[3]
    scene = mujoco.MjvScene(model, maxgeom=1_000)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    mujoco.mjv_updateScene(
        model,
        data,
        option,
        mujoco.MjvPerturb(),
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    names = []
    for index in range(scene.ngeom):
        geom = scene.geoms[index]
        if (
            int(geom.objtype) == int(mujoco.mjtObj.mjOBJ_GEOM)
            and int(model.geom_group[int(geom.objid)]) == 3
        ):
            names.append(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom.objid))
                or ""
            )
    assert "base_chassis_collision" in names
    assert sum(name.startswith("source_shelf_") for name in names) == 6
    assert sum(name.startswith("side_table_") for name in names) == 2

    teleop_render._append_collision_overlay(
        SimpleNamespace(
            model=model,
            scene=scene,
            task_collision_geom_ids=task.collision_geom_ids,
            whole_body_solver=SimpleNamespace(collision_safe_distance=0.01),
        ),
        (),
    )
    blue_floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_bin_floor")
    rendered_blue_floor = next(
        scene.geoms[index]
        for index in range(scene.ngeom)
        if int(scene.geoms[index].objid) == blue_floor
    )
    assert np.allclose(rendered_blue_floor.rgba, teleop_render.COLLISION_GEOMETRY_RGBA)
    assert not rendered_blue_floor.transparent
    rendered_base = next(
        scene.geoms[index]
        for index in range(scene.ngeom)
        if int(scene.geoms[index].objid)
        == mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "base_chassis_collision"
        )
    )
    assert np.allclose(rendered_base.rgba, teleop_render.COLLISION_GEOMETRY_RGBA)
    assert not rendered_base.transparent
