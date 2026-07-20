"""
Regenerates ``tools/asset_manifest.json`` for fetch_assets.py.

Run this from a checkout containing the fetched meca500/owi535 assets. It
downloads the pristine upstream files and re-derives every input pin (download
SHA-256s and material colors). The OWI simplification targets below are part
of the pipeline and are intentionally not inferred from whichever output
meshes happen to be installed.

Usage:
    python tools/generate_asset_manifest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

import fetch_assets as fa  # noqa: E402

OWI_BASE = ("https://www.paris.inria.fr/archive_ylabbeprojectsdata/robopose/"
            "deps/owi-description/owi535_description")
MECA_RAW_BASE = "https://raw.githubusercontent.com/Vanderbilt-Applied-Robotics-Lab/meca500_ros2"
MECA_URDF_PATH = "src/meca500_description/urdf/meca500.xacro"
MECA_MESH_PATH = "src/meca500_description/meshes/visual"
MECA_COMMIT = "d26066e88274bf81e87b7bfd6591d8db190abd9a"
MECA_PARTS = [f"meca_500_r3_{p}" for p in ("base", "j1", "j2", "j3", "j4", "j5", "j6")]

# Simplifier requests, not promised output face counts (meshoptimizer may stop
# just below a requested count, and its output is not bit-deterministic).
OWI_DECIMATE_TARGETS = {
    "Untitled_001-mesh.0.obj": 56728,
    "Untitled_001-mesh.2.obj": 9617,
    "Untitled_040-mesh.0.obj": 5000,
    "Untitled_060-mesh.0.obj": 5000,
    "Untitled_063-mesh.0.obj": 5000,
    "Untitled_064-mesh.0.obj": 5000,
}

OWI_REPO = fa.ROBOTS_DIR / "owi535"
MECA_REPO = fa.ROBOTS_DIR / "meca500/meshes/visual"


def _parse_mtl_colors(path: Path) -> dict:
    d: dict = {}
    for ln in path.read_text().splitlines():
        parts = ln.split()
        if not parts:
            continue
        if parts[0] in ("Ka", "Kd", "Ks"):
            d[parts[0]] = [float(x) for x in parts[1:4]]
        elif parts[0] in ("Tr", "Ns"):
            d[parts[0]] = float(parts[1])
        elif parts[0] == "illum":
            d[parts[0]] = int(parts[1])
    return d


def main() -> int:
    manifest: dict = {}

    # ------------------------------------------------------------- owi535 --
    dl = fa.DOWNLOADS_DIR / "owi535"
    urdf_local = fa._download(f"{OWI_BASE}/owi535.urdf", dl / "owi535.urdf", None, True)
    expected_meshes = sorted(fa._parse_visual_origins(urdf_local))

    files: dict = {}
    for name in expected_meshes:
        repo_obj = OWI_REPO / name
        if not repo_obj.is_file():
            raise FileNotFoundError(f"Missing fetched OWI mesh: {repo_obj}")
        src = fa._download(f"{OWI_BASE}/{name}", dl / name, None, True)
        colors = _parse_mtl_colors(OWI_REPO / f"{name}.mtl")

        # sanity: our MTL template must reproduce the repo MTL byte-exactly
        mtl_text = fa._write_owi_mtl(colors)
        assert mtl_text == (OWI_REPO / f"{name}.mtl").read_text(), f"MTL template mismatch: {name}"

        files[name] = {
            "sha256": fa._sha256(src),
            "colors": colors,
            "decimate_target": OWI_DECIMATE_TARGETS.get(name),
        }

    manifest["owi535"] = {
        "base_url": OWI_BASE,
        "urdf": {"name": "owi535.urdf", "sha256": fa._sha256(urdf_local)},
        "files": files,
    }
    n_dec = sum(1 for e in files.values() if e["decimate_target"])
    print(f"owi535: {len(files)} meshes pinned ({n_dec} decimated)")

    # ------------------------------------------------------------ meca500 --
    dlm = fa.DOWNLOADS_DIR / "meca500"
    meca_urdf = fa._download(
        f"{MECA_RAW_BASE}/{MECA_COMMIT}/{MECA_URDF_PATH}",
        dlm / Path(MECA_URDF_PATH).name,
        None,
        True,
    )
    mfiles: dict = {}
    for part in MECA_PARTS:
        url = f"{MECA_RAW_BASE}/{MECA_COMMIT}/{MECA_MESH_PATH}/{part}.dae"
        dae = fa._download(url, dlm / f"{part}.dae", None, True)
        mfiles[part] = {"sha256": fa._sha256(dae)}
    manifest["meca500"] = {
        "raw_base": MECA_RAW_BASE,
        "commit": MECA_COMMIT,
        "urdf": {
            "source_path": MECA_URDF_PATH,
            "name": "meca500.urdf",
            "sha256": fa._sha256(meca_urdf),
        },
        "mesh_path": MECA_MESH_PATH,
        "files": mfiles,
    }
    print(f"meca500: {len(mfiles)} meshes pinned at commit {MECA_COMMIT[:10]}")

    fa._atomic_write_text(fa.MANIFEST_PATH, json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {fa.MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
