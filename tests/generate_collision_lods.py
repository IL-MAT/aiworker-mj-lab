"""Generate lightweight robot collision proxies for the debug collision view.

Install the optional generator dependency before running this script:
``pip install trimesh scipy fast-simplification``.
"""

import re
from pathlib import Path

import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "full_scene.xml"
MESH_ROOT = REPO_ROOT / "assets" / "robotis_ffw" / "assets"
# Keep the existing asset directory name for MJCF compatibility. These files
# are collision-only proxies despite the historical directory name.
OUTPUT_DIR = REPO_ROOT / "assets" / "visual_lod"
MIN_SOURCE_FACES = 4_000
MIN_TARGET_FACES = 500
MAX_TARGET_FACES = 2_500
TARGET_RATIO = 0.05


def _visual_mesh_assets():
    source = MODEL_PATH.read_text(encoding="utf-8")
    assets = {
        match.group(1): match.group(2)
        for match in re.finditer(
            r'<mesh name="([^"]+)" file="([^"]+)"(?: scale="[^"]+")?\s*/>',
            source,
        )
    }
    visual_names = {
        name
        for match in re.finditer(r"<geom\b[^>]*/>", source)
        if 'class="visual"' in match.group(0)
        for name in re.findall(r'mesh="([^"]+)"', match.group(0))
    }
    source_names = {
        name.removesuffix("_visual_lod") for name in visual_names
    }
    return {
        name: MESH_ROOT / assets[name]
        for name in sorted(source_names)
        if assets[name].lower().endswith(".stl") or name == "can_cap_detail"
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_source = 0
    total_lod = 0
    for name, source_path in _visual_mesh_assets().items():
        source_mesh = trimesh.load_mesh(source_path, process=True)
        source_faces = len(source_mesh.faces)
        if source_faces < MIN_SOURCE_FACES:
            continue
        target_faces = max(
            MIN_TARGET_FACES,
            min(MAX_TARGET_FACES, round(source_faces * TARGET_RATIO)),
        )
        # MuJoCo mesh collision already operates on a convex hull. Store that
        # representation explicitly so V mode shows a clean collision proxy
        # instead of the vendor STL's disconnected CAD shells. These assets are
        # never used by the normal visual geoms, which retain the original mesh.
        lod = source_mesh.convex_hull
        if len(lod.faces) > target_faces:
            lod = lod.simplify_quadric_decimation(
                face_count=target_faces,
                aggression=7,
            )
        # OBJ carries shared vertices and normals, producing a legible V-mode
        # overlay. The normal operator view does not reference this asset.
        output_path = OUTPUT_DIR / f"{name}.obj"
        lod.export(output_path, include_normals=True)
        lod_faces = len(lod.faces)
        total_source += source_faces
        total_lod += lod_faces
        print(f"{name}: {source_faces} -> {lod_faces} faces")
    print(f"total: {total_source} -> {total_lod} faces")


if __name__ == "__main__":
    main()
