"""Configurable can-placement scenarios, reset logic, and success metrics."""

from dataclasses import dataclass

import mujoco
import numpy as np

from ...config import SETTINGS

TASK_NAMES = ("can_to_box", "can_color_sort", "shelf_color_sort")
_KNOWN_BIN_BODIES = (
    "target_bin",
    "target_bin_red",
    "source_shelf",
    "side_table_red",
    "side_table_blue",
)


def _required_id(model, kind, name):
    object_id = mujoco.mj_name2id(model, kind, name)
    if object_id < 0:
        raise ValueError(f"task entity is missing from the model: {name}")
    return object_id


@dataclass(frozen=True)
class CanVariant:
    """One visual object variant and the bin that is correct for it."""

    name: str
    material_name: str
    cap_material_name: str
    target_site_name: str
    target_label: str
    rgba: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class CanTaskScenario:
    """Declarative scene selection used by the shared can task implementation."""

    name: str
    description: str
    variants: tuple[CanVariant, ...]
    bin_body_names: tuple[str, ...]
    spawn_jitter_radius: float
    can_z: float
    spawn_site_name: str | None = None
    spawn_anchor_offset: tuple[float, float] = (0.0, 0.0)
    bin_positions: tuple[tuple[float, float, float], ...] | None = None
    table_position_y: float | None = None
    table_half_width_x: float | None = None
    table_half_width_y: float | None = None
    bin_outer_half_extent: float | None = None
    success_inner_half_extents: tuple[float, float] | None = None
    bin_wall_half_height: float | None = None
    right_arm_start_position: tuple[float, ...] | None = None
    table_visible: bool = True
    fixture_body_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskMetrics:
    success: bool
    object_position_error: float
    object_speed: float
    can_position: np.ndarray
    target_position: np.ndarray
    object_variant: str = "green"
    target_label: str = "blue"


def _legacy_scenario():
    return CanTaskScenario(
        name="can_to_box",
        description="green can -> blue box",
        variants=(
            CanVariant(
                "green",
                "can_mat",
                "can_cap_mat",
                SETTINGS.get("imitation.task.target_site"),
                "blue",
            ),
        ),
        bin_body_names=("target_bin",),
        spawn_jitter_radius=SETTINGS.number(
            "imitation.reset.can_spawn_jitter_radius_m", minimum=0.0
        ),
        can_z=SETTINGS.number("imitation.reset.can_z_m"),
        spawn_anchor_offset=tuple(
            SETTINGS.get("imitation.reset.can_anchor_offset_from_target_xy_m")
        ),
    )


def _color_sort_scenario():
    name = "can_color_sort"
    values = SETTINGS.get("imitation.scenarios.can_color_sort")
    variant_fields = (
        values["variant_names"],
        values["variant_materials"],
        values["variant_cap_materials"],
        values["target_sites"],
        values["target_labels"],
        values["variant_rgba"],
    )
    if len({len(field) for field in variant_fields}) != 1 or not variant_fields[0]:
        raise ValueError(f"{name} variant settings must have equal non-zero lengths")
    variants = tuple(
        CanVariant(*items[:-1], tuple(float(value) for value in items[-1]))
        for items in zip(*variant_fields)
    )
    if any(len(variant.rgba) != 4 for variant in variants):
        raise ValueError(f"{name} variant RGBA values must have length 4")
    return CanTaskScenario(
        name=name,
        description=("green/blue can -> blue box; red/orange can -> red box"),
        variants=variants,
        bin_body_names=tuple(values["bin_bodies"]),
        spawn_jitter_radius=float(values["spawn_jitter_radius_m"]),
        can_z=float(values["can_z_m"]),
        spawn_site_name=values["spawn_site"],
        bin_positions=tuple(
            tuple(float(coordinate) for coordinate in position)
            for position in values["bin_positions_m"]
        ),
        table_position_y=float(values["table_position_y_m"]),
        table_half_width_x=float(values["table_half_width_x_m"]),
        table_half_width_y=float(values["table_half_width_y_m"]),
        bin_wall_half_height=float(values["bin_wall_half_height_m"]),
        right_arm_start_position=tuple(
            float(value) for value in values["right_arm_start_position_rad"]
        ),
    )


def _shelf_color_sort_scenario():
    name = "shelf_color_sort"
    values = SETTINGS.get("imitation.scenarios.shelf_color_sort")
    variant_fields = (
        values["variant_names"],
        values["variant_materials"],
        values["variant_cap_materials"],
        values["target_sites"],
        values["target_labels"],
        values["variant_rgba"],
    )
    if len({len(field) for field in variant_fields}) != 1 or not variant_fields[0]:
        raise ValueError(f"{name} variant settings must have equal non-zero lengths")
    variants = tuple(
        CanVariant(*items[:-1], tuple(float(value) for value in items[-1]))
        for items in zip(*variant_fields)
    )
    return CanTaskScenario(
        name=name,
        description="red can -> left table box; blue can -> right table box",
        variants=variants,
        bin_body_names=tuple(values["bin_bodies"]),
        bin_positions=tuple(
            tuple(float(coordinate) for coordinate in position)
            for position in values["bin_positions_m"]
        ),
        spawn_jitter_radius=float(values["spawn_jitter_radius_m"]),
        can_z=float(values["can_z_m"]),
        spawn_site_name=values["spawn_site"],
        bin_outer_half_extent=float(values["bin_outer_half_extent_m"]),
        success_inner_half_extents=tuple(
            float(value) for value in values["success_inner_half_extents_m"]
        ),
        table_visible=False,
        fixture_body_names=tuple(values["fixture_bodies"]),
    )


_SCENARIO_BUILDERS = {
    "can_to_box": _legacy_scenario,
    "can_color_sort": _color_sort_scenario,
    "shelf_color_sort": _shelf_color_sort_scenario,
}


def scenario_for_name(name):
    """Build a validated scenario without coupling callers to task classes."""
    try:
        scenario = _SCENARIO_BUILDERS[str(name)]()
    except KeyError as error:
        raise ValueError(
            f"unknown imitation task {name!r}; choose one of {TASK_NAMES}"
        ) from error
    if scenario.spawn_jitter_radius < 0.0:
        raise ValueError("task spawn jitter radius must be non-negative")
    if len(scenario.spawn_anchor_offset) != 2:
        raise ValueError("task spawn anchor offset must contain 2 values")
    if not scenario.variants:
        raise ValueError("task scenario must define at least one can variant")
    if scenario.bin_positions is not None and len(scenario.bin_positions) != len(
        scenario.bin_body_names
    ):
        raise ValueError("task bin positions must match task bin bodies")
    if scenario.bin_positions is not None and any(
        len(position) != 3 for position in scenario.bin_positions
    ):
        raise ValueError("each task bin position must contain xyz")
    if scenario.table_half_width_y is not None and scenario.table_half_width_y <= 0.0:
        raise ValueError("task table half-width must be positive")
    if scenario.table_half_width_x is not None and scenario.table_half_width_x <= 0.0:
        raise ValueError("task table half-depth must be positive")
    if (
        scenario.bin_outer_half_extent is not None
        and scenario.bin_outer_half_extent <= 0.0
    ):
        raise ValueError("task bin outer half-extent must be positive")
    if scenario.success_inner_half_extents is not None and (
        len(scenario.success_inner_half_extents) != 2
        or any(value <= 0.0 for value in scenario.success_inner_half_extents)
    ):
        raise ValueError("task success inner half-extents must contain 2 positive values")
    if (
        scenario.bin_wall_half_height is not None
        and scenario.bin_wall_half_height <= 0.0
    ):
        raise ValueError("task bin wall half-height must be positive")
    if (
        scenario.right_arm_start_position is not None
        and len(scenario.right_arm_start_position) != 7
    ):
        raise ValueError("task right-arm start position must contain 7 values")
    return scenario


class CanInBoxTask:
    """Run either the legacy placement task or a configured color-sort variant.

    A single physical can body is shared by all variants. Reset changes only its
    visual material, free-joint state, and desired target site, so color cannot
    accidentally change mass, inertia, friction, or collision geometry.
    """

    def __init__(self, model, scenario=None):
        self.model = model
        self.scenario = _legacy_scenario() if scenario is None else scenario
        self.name = self.scenario.name
        self.description = self.scenario.description
        self.bin_body_names = self.scenario.bin_body_names
        self.collision_body_names = tuple(
            dict.fromkeys(
                self.scenario.bin_body_names + self.scenario.fixture_body_names
            )
        )
        self.can_joint = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, "can_free")
        self.can_body = _required_id(model, mujoco.mjtObj.mjOBJ_BODY, "can")
        self.can_visual_geom = _required_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "can_side_visual"
        )
        self.can_cap_visual_geom = _required_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "can_cap_visual"
        )
        self.can_qpos = int(model.jnt_qposadr[self.can_joint])
        self.can_dof = int(model.jnt_dofadr[self.can_joint])
        self.spawn_anchor_offset = np.asarray(
            self.scenario.spawn_anchor_offset, dtype=float
        )
        self.spawn_jitter_radius = float(self.scenario.spawn_jitter_radius)
        self.can_z = float(self.scenario.can_z)
        inner_half_extents = (
            SETTINGS.get("imitation.task.inner_half_extents_m")
            if self.scenario.success_inner_half_extents is None
            else self.scenario.success_inner_half_extents
        )
        self.inner_half_extents = np.asarray(inner_half_extents, dtype=float)
        self.height_range = tuple(
            float(v) for v in SETTINGS.get("imitation.task.success_height_range_m")
        )
        self.settle_speed = SETTINGS.number(
            "imitation.task.settle_speed_m_s", minimum=0.0
        )
        if self.inner_half_extents.shape != (2,):
            raise ValueError("imitation task inner half extents must contain 2 values")
        if not self.height_range[0] < self.height_range[1]:
            raise ValueError("imitation task height range must be increasing")

        self._variants = tuple(
            self._bind_variant(variant) for variant in self.scenario.variants
        )
        self.spawn_site = (
            _required_id(model, mujoco.mjtObj.mjOBJ_SITE, self.scenario.spawn_site_name)
            if self.scenario.spawn_site_name is not None
            else None
        )
        self.current_variant = self._variants[0]
        self.target_site = self.current_variant[1]
        self._variant_index = 0
        self._bin_colors_swapped = False
        self._configure_scene()
        self._target_sites_by_label = {
            variant.target_label: target_site
            for variant, target_site, _material, _cap_material in self._variants
        }
        self._base_target_sites_by_label = dict(self._target_sites_by_label)
        self._apply_variant(self.current_variant)

    def _bind_variant(self, variant):
        material_id = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_MATERIAL, variant.material_name
        )
        cap_material_id = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_MATERIAL, variant.cap_material_name
        )
        target_site = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, variant.target_site_name
        )
        return variant, target_site, material_id, cap_material_id

    def _configure_scene(self):
        """Apply the selected scenario's visibility, layout, and geometry."""
        active = set(self.collision_body_names)
        positions = (
            {}
            if self.scenario.bin_positions is None
            else dict(zip(self.bin_body_names, self.scenario.bin_positions))
        )
        self._bin_geom_ids = {}
        for body_name in _KNOWN_BIN_BODIES:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                if body_name in active:
                    raise ValueError(f"active task bin is missing: {body_name}")
                continue
            geom_ids = np.flatnonzero(self.model.geom_bodyid == body_id).astype(int)
            site_ids = np.flatnonzero(self.model.site_bodyid == body_id).astype(int)
            if body_name not in active:
                self.model.geom_rgba[geom_ids, 3] = 0.0
                self.model.site_rgba[site_ids, 3] = 0.0
                self.model.geom_contype[geom_ids] = 0
                self.model.geom_conaffinity[geom_ids] = 0
                self.model.body_contype[body_id] = 0
                self.model.body_conaffinity[body_id] = 0
                continue
            self._bin_geom_ids[body_name] = geom_ids
            if body_name in positions:
                self.model.body_pos[body_id] = positions[body_name]
            if not body_name.startswith("target_bin"):
                collision_geom_ids = geom_ids[self.model.geom_group[geom_ids] == 3]
                self.model.geom_rgba[collision_geom_ids] = [0.05, 0.75, 1.0, 0.18]
                continue
            floor_ids = [
                geom_id
                for geom_id in geom_ids
                if (
                    mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)
                    )
                    or ""
                ).endswith("floor")
            ]
            if len(floor_ids) != 1:
                raise ValueError(
                    f"{body_name} must contain exactly one named floor geom"
                )
            floor_id = floor_ids[0]
            if self.scenario.bin_outer_half_extent is not None:
                self._resize_bin_xy(geom_ids, self.scenario.bin_outer_half_extent)
            floor_top = float(
                self.model.geom_pos[floor_id, 2] + self.model.geom_size[floor_id, 2]
            )
            for geom_id in geom_ids:
                geom_name = (
                    mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)
                    )
                    or ""
                )
                self.model.geom_rgba[geom_id, 3] = (
                    1.0 if geom_name.endswith("floor") else 0.85
                )
                if (
                    not geom_name.endswith("floor")
                    and self.scenario.bin_wall_half_height is not None
                ):
                    half_height = self.scenario.bin_wall_half_height
                    self.model.geom_size[geom_id, 2] = half_height
                    self.model.geom_pos[geom_id, 2] = floor_top + half_height
        table_geom = _required_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        if not self.scenario.table_visible:
            self.model.geom_rgba[table_geom, 3] = 0.0
            self.model.geom_contype[table_geom] = 0
            self.model.geom_conaffinity[table_geom] = 0
        elif (
            self.scenario.table_position_y is not None
            or self.scenario.table_half_width_x is not None
            or self.scenario.table_half_width_y is not None
        ):
            if self.scenario.table_position_y is not None:
                self.model.geom_pos[table_geom, 1] = self.scenario.table_position_y
            if self.scenario.table_half_width_x is not None:
                self.model.geom_size[table_geom, 0] = self.scenario.table_half_width_x
            if self.scenario.table_half_width_y is not None:
                self.model.geom_size[table_geom, 1] = self.scenario.table_half_width_y
        self._base_bin_rgba = {
            body_name: self.model.geom_rgba[geom_ids].copy()
            for body_name, geom_ids in self._bin_geom_ids.items()
            if body_name in self.bin_body_names
        }

    def _resize_bin_xy(self, geom_ids, outer_half_extent):
        """Resize an open box in XY while preserving every vertical dimension."""
        wall_half_thickness = 0.005
        wall_center = outer_half_extent - wall_half_thickness
        inner_half_extent = outer_half_extent - 2.0 * wall_half_thickness
        for geom_id in geom_ids:
            geom_name = (
                mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)
                )
                or ""
            )
            if geom_name.endswith("floor"):
                self.model.geom_size[geom_id, :2] = outer_half_extent
            elif geom_name.endswith(("front", "back")):
                self.model.geom_pos[geom_id, 1] = np.copysign(
                    wall_center, self.model.geom_pos[geom_id, 1]
                )
                self.model.geom_size[geom_id, :2] = (
                    outer_half_extent,
                    wall_half_thickness,
                )
            elif geom_name.endswith(("left", "right")):
                self.model.geom_pos[geom_id, 0] = np.copysign(
                    wall_center, self.model.geom_pos[geom_id, 0]
                )
                self.model.geom_size[geom_id, :2] = (
                    wall_half_thickness,
                    inner_half_extent,
                )
            self.model.geom_rbound[geom_id] = np.linalg.norm(
                self.model.geom_size[geom_id]
            )

    def _set_bin_color_layout(self, swapped):
        """Swap only bin appearance and matching target-site assignments."""
        swapped = bool(swapped)
        if swapped == self._bin_colors_swapped:
            return
        if len(self._base_bin_rgba) != 2:
            if swapped:
                raise ValueError("bin color swapping requires exactly two bins")
            return
        body_names = tuple(self._base_bin_rgba)
        source_names = body_names[::-1] if swapped else body_names
        for destination, source in zip(body_names, source_names):
            self.model.geom_rgba[self._bin_geom_ids[destination]] = self._base_bin_rgba[
                source
            ]

        labels = tuple(self._base_target_sites_by_label)
        if len(labels) != 2:
            raise ValueError("bin color swapping requires exactly two targets")
        source_labels = labels[::-1] if swapped else labels
        self._target_sites_by_label = {
            destination: self._base_target_sites_by_label[source]
            for destination, source in zip(labels, source_labels)
        }
        self._bin_colors_swapped = swapped

    def _apply_variant(self, bound_variant):
        variant, target_site, material_id, cap_material_id = bound_variant
        self.current_variant = bound_variant
        self._variant_index = self._variants.index(bound_variant)
        self.target_site = self._target_sites_by_label.get(
            variant.target_label, target_site
        )
        self.model.geom_matid[self.can_visual_geom] = material_id
        self.model.geom_matid[self.can_cap_visual_geom] = cap_material_id
        if variant.rgba is not None:
            self.model.mat_rgba[material_id] = variant.rgba
            self.model.mat_rgba[cap_material_id] = variant.rgba
        return variant

    @property
    def object_variant(self):
        return self.current_variant[0].name

    @property
    def target_label(self):
        return self.current_variant[0].target_label

    @property
    def variant_names(self):
        return tuple(bound[0].name for bound in self._variants)

    @property
    def bin_colors_swapped(self):
        return self._bin_colors_swapped

    @property
    def bin_color_layout(self):
        """Map semantic bin colors to their current physical body names."""
        return {
            label: mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.site_bodyid[site_id]),
            )
            for label, site_id in self._target_sites_by_label.items()
        }

    def reset(
        self,
        data,
        rng,
        *,
        cycle_variant=False,
        swap_bin_colors=False,
        randomize_bin_colors=False,
        allowed_variants=None,
    ):
        """Reset only the free task object and sample its visual/target pairing."""
        if swap_bin_colors and randomize_bin_colors:
            raise ValueError("bin colors cannot be swapped and randomized together")
        mujoco.mj_forward(self.model, data)
        if allowed_variants is None:
            variant_indices = tuple(range(len(self._variants)))
        else:
            requested = tuple(dict.fromkeys(str(name) for name in allowed_variants))
            unknown = set(requested) - set(self.variant_names)
            if unknown:
                raise ValueError(f"unknown variants for {self.name}: {sorted(unknown)}")
            if not requested:
                raise ValueError("allowed_variants must not be empty")
            variant_indices = tuple(
                self.variant_names.index(name) for name in requested
            )
        # Preserve the exact legacy seeded distribution by consuming no random
        # draw when a scenario has only one visual variant.
        variant_index = (
            variant_indices[
                (variant_indices.index(self._variant_index) + 1) % len(variant_indices)
            ]
            if (
                cycle_variant
                and len(variant_indices) > 1
                and self._variant_index in variant_indices
            )
            else variant_indices[0]
            if len(variant_indices) == 1
            else variant_indices[int(rng.integers(len(variant_indices)))]
        )
        if randomize_bin_colors:
            self._set_bin_color_layout(bool(rng.integers(2)))
        elif swap_bin_colors:
            self._set_bin_color_layout(not self._bin_colors_swapped)
        self._apply_variant(self._variants[variant_index])

        radius = self.spawn_jitter_radius * np.sqrt(rng.uniform())
        angle = rng.uniform(0.0, 2.0 * np.pi)
        jitter = radius * np.asarray([np.cos(angle), np.sin(angle)])
        anchor_site = self.target_site if self.spawn_site is None else self.spawn_site
        anchor_xy = np.asarray(data.site_xpos[anchor_site, :2])
        can_xy = anchor_xy + self.spawn_anchor_offset + jitter
        data.qpos[self.can_qpos : self.can_qpos + 3] = [
            can_xy[0],
            can_xy[1],
            self.can_z,
        ]
        data.qpos[self.can_qpos + 3 : self.can_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[self.can_dof : self.can_dof + 6] = 0.0
        mujoco.mj_forward(self.model, data)
        return np.asarray(data.qpos[self.can_qpos : self.can_qpos + 3]).copy()

    def metrics(self, data):
        can_position = np.asarray(data.xpos[self.can_body], dtype=float).copy()
        target_position = np.asarray(
            data.site_xpos[self.target_site], dtype=float
        ).copy()
        delta = can_position - target_position
        speed = float(np.linalg.norm(data.qvel[self.can_dof : self.can_dof + 3]))
        inside_xy = bool(np.all(np.abs(delta[:2]) <= self.inner_half_extents))
        inside_height = self.height_range[0] <= can_position[2] <= self.height_range[1]
        return TaskMetrics(
            success=inside_xy and inside_height and speed <= self.settle_speed,
            object_position_error=float(np.linalg.norm(delta)),
            object_speed=speed,
            can_position=can_position,
            target_position=target_position,
            object_variant=self.object_variant,
            target_label=self.target_label,
        )

    def episode_metadata(self):
        return {
            "scenario_name": self.name,
            "object_variant": self.object_variant,
            "target_label": self.target_label,
            "object_target_mapping": {
                variant.name: variant.target_label
                for variant, _site, _side_material, _cap_material in self._variants
            },
            "bin_color_layout": self.bin_color_layout,
            # Runtime scenario positions are configuration-driven and therefore
            # are not covered by the MJCF file hash. Store them per episode so
            # mixed-layout color-sort data remains auditable.
            "bin_body_positions": {
                body_name: self.model.body_pos[
                    _required_id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                ].tolist()
                for body_name in self.bin_body_names
            },
        }


def create_task(model, name="can_to_box"):
    return CanInBoxTask(model, scenario_for_name(name))


__all__ = [
    "TASK_NAMES",
    "CanInBoxTask",
    "CanTaskScenario",
    "CanVariant",
    "TaskMetrics",
    "create_task",
    "scenario_for_name",
]
