"""
Regenerate the shipped xArm7 visual meshes from EasyHeC's textured GLBs.

The OBJs under ``robots/xarm7/meshes/visual/`` are conversions of the
MIT-licensed textured GLB meshes in ootts/EasyHeC (whose geometry matches the
official xArm-Developer/xarm_ros2 STLs: each link's bounding box matches the
official mesh to 0.000 mm, so the geometry is the same, only the materials
differ).  They replaced an earlier mesh set of unknown provenance after an
A/B evaluation on the hydra_xarm benchmark showed better mean ADD and AUC;
see MESH_LICENSES/README.md, statement of changes for "xarm7".

Since the source is MIT, the converted meshes are shipped directly with the
package, so you normally never need this tool. It exists so the conversion
stays documented and reproducible:

    python tools/convert_xarm_easyhec.py     # download (SHA-pinned) + convert + validate

The GLBs' 'white'/'silver' material primitives are mapped onto the repo's
``white``/``dark`` two-material convention; validation compares each link's
bounding box against the official xarm_ros2 STL.

Requires: numpy, trimesh.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
OUT_ROOT = REPO_ROOT / "src/robot_renderer/robots/xarm7/meshes/visual"
DOWNLOADS_DIR = TOOLS_DIR / "_downloads/xarm7_easyhec"

EASYHEC_BASE = ("https://media.githubusercontent.com/media/ootts/EasyHeC/main/"
                "assets/xarm_description/meshes/xarm7/visual")
EASYHEC_BASE_FALLBACK = ("https://raw.githubusercontent.com/ootts/EasyHeC/main/"
                         "assets/xarm_description/meshes/xarm7/visual")
XARM_ROS2_STL_BASE = ("https://raw.githubusercontent.com/xArm-Developer/xarm_ros2/"
                      "humble/xarm_description/meshes/xarm7/visual")

# link name -> (glb filename, pinned sha256)
GLBS: Dict[str, Tuple[str, str]] = {
    "link_base": ("link_base_textured.glb", "bdccb694dd55ef68c8b120d14f464a56c911ff4a55397b43bc19313b6184356b"),
    "link1": ("link1_textured.glb", "fc699576efc528fa469ce6136f72939f6b9c99615e1fca705da88416f9bfda0b"),
    "link2": ("link2_textured.glb", "83c89dbf7f1e264a6e47a00f757a30666aac405e4b4b478d84efc9dc73563cdb"),
    "link3": ("link3_textured.glb", "ae0ce44cd2531b6b5e41f760ec681474b0f437b586038497c0cd5e3ae710047a"),
    "link4": ("link4_textured.glb", "94af657f4434478d1f0db4bc0994a4f42c3659f15eb452a9f6306bf63161669d"),
    "link5": ("link5_textured.glb", "48630b4b7d1162d8a2f47995d05a09263296aa2ae50970ac893c5f95e8838661"),
    "link6": ("link6_textured.glb", "ed865e4c91983dd194d2a9e2de08d5eba2a3ffe2a35cf70901f7609fa4aafd28"),
    "link7": ("link7_silver.glb", "484e3ee74e345f1e6ef24c37e684fd1a4328a56af558ee057f539c177f60c1cb"),
}

# GLB material name -> repo material name.
MATERIAL_MAP = {"white": "white", "silver": "dark"}

# Same material definitions as the rest of the robots (our colors).
MATERIAL_LIB = """newmtl dark
Ka 0 0 0
Kd 0.5500 0.5500 0.5500
Ks 0.1000 0.1000 0.1000
Ns 20
d 1
illum 2

newmtl white
Ka 0 0 0
Kd 1.0000 1.0000 1.0000
Ks 0.0300 0.0300 0.0300
Ns 25
d 1
illum 2
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(fname: str, expected_sha: Optional[str]) -> Path:
    dest = DOWNLOADS_DIR / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        last_err: Optional[Exception] = None
        for base in (EASYHEC_BASE, EASYHEC_BASE_FALLBACK):
            try:
                with urllib.request.urlopen(f"{base}/{fname}", timeout=60) as resp, \
                        tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
                    shutil.copyfileobj(resp, tmp)
                Path(tmp.name).rename(dest)
                last_err = None
                break
            except urllib.error.URLError as e:
                last_err = e
        if last_err is not None:
            raise RuntimeError(f"Download failed: {fname} ({last_err})")
    if expected_sha:
        got = _sha256(dest)
        if got != expected_sha:
            raise RuntimeError(
                f"SHA-256 mismatch for {fname}: expected {expected_sha}, got {got}.\n"
                f"Upstream changed. Inspect the file, then update the pin in this script."
            )
    return dest


def _convert_link(link: str, glb_path: Path) -> Tuple[str, Dict[str, int]]:
    """GLB -> repo-format OBJ text. Returns (obj_text, faces-per-material)."""
    import trimesh

    scene = trimesh.load(glb_path)
    groups: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry[geom_name]
        mat_name = getattr(geom.visual.material, "name", None) or "silver"
        repo_mat = MATERIAL_MAP.get(mat_name)
        if repo_mat is None:
            print(f"  WARNING: unknown material {mat_name!r} in {link}, mapping it to 'dark'")
            repo_mat = "dark"
        verts = trimesh.transformations.transform_points(geom.vertices, transform)
        groups.setdefault(repo_mat, []).append((np.asarray(verts, dtype=np.float64),
                                                np.asarray(geom.faces, dtype=np.int64)))

    lines: List[str] = [f"# xArm7 {link}", "mtllib material.lib"]
    all_verts: List[np.ndarray] = []
    face_blocks: List[Tuple[str, np.ndarray]] = []
    offset = 0
    for mat in ("dark", "white"):
        if mat not in groups:
            continue
        mat_faces: List[np.ndarray] = []
        for verts, faces in groups[mat]:
            all_verts.append(verts)
            mat_faces.append(faces + offset)
            offset += len(verts)
        face_blocks.append((mat, np.vstack(mat_faces)))

    for v in np.vstack(all_verts):
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    counts: Dict[str, int] = {}
    for mat, faces in face_blocks:
        lines.append(f"usemtl {mat}")
        counts[mat] = len(faces)
        for f in faces:
            lines.append(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}")
    return "\n".join(lines) + "\n", counts


def _validate_link(link: str, obj_text: str) -> str:
    """Compare bbox against the official xarm_ros2 STL for this link."""
    import trimesh

    dest = DOWNLOADS_DIR / f"ros2_{link}.stl"
    if not dest.exists():
        try:
            with urllib.request.urlopen(f"{XARM_ROS2_STL_BASE}/{link}.stl", timeout=60) as resp, \
                    tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
                shutil.copyfileobj(resp, tmp)
            Path(tmp.name).rename(dest)
        except urllib.error.URLError:
            return "validation skipped (STL download failed)"
    ref = trimesh.load(dest, force="mesh")
    verts = np.array([[float(x) for x in ln.split()[1:4]]
                      for ln in obj_text.splitlines() if ln.startswith("v ")])
    delta = max(np.abs(ref.vertices.min(0) - verts.min(0)).max(),
                np.abs(ref.vertices.max(0) - verts.max(0)).max())
    status = "ok" if delta < 1e-3 else "MISMATCH"
    return f"bbox vs official STL: {delta * 1000:.3f} mm [{status}]"


def main(argv: Optional[List[str]] = None) -> int:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args(argv)

    print(f"Converting EasyHeC GLBs -> {OUT_ROOT}")
    failures = 0
    for link, (fname, sha) in GLBS.items():
        glb = _download(fname, sha)
        obj_text, counts = _convert_link(link, glb)
        out_dir = OUT_ROOT / link
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{link}.obj").write_text(obj_text)
        (out_dir / "material.lib").write_text(MATERIAL_LIB)
        val = _validate_link(link, obj_text)
        if "MISMATCH" in val or "skipped" in val:
            failures += 1
        parts = ", ".join(f"{m}:{n}" for m, n in counts.items())
        print(f"  {link:9s} faces {parts:24s} {val}")
    print("done." if not failures else f"done with {failures} validation issue(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
