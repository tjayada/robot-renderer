"""
Fetch-and-transform pipeline for robot assets robot-renderer cannot redistribute.

Why this exists
---------------
Two robots' asset sources carry no usable redistribution license:

  * ``meca500``: the robot description and meshes come from
    Vanderbilt-Applied-Robotics-Lab/meca500_ros2, which declares no license
    (``package.xml``: "TODO: License declaration").
  * ``owi535``: the description and meshes come from the RoboPose dependency
    archive hosted by INRIA (no license statement; upstream of that is the
    CRAVES project).

Instead of shipping those assets, this tool downloads the originals from
their public sources and re-applies our (fully scripted) modifications
locally. All other robots' assets are permissively licensed (Apache-2.0,
BSD-3-Clause, or MIT) and are shipped directly with the package. See
``MESH_LICENSES/README.md`` for the full provenance table.

What it reproduces
------------------
owi535 (from the INRIA archive):
  1. Bake each mesh's URDF ``<visual>`` origin translation into the vertices
     and convert mm -> m:  ``v' = (v + xyz) * 0.001``, written with 8 decimals.
     (The archive URDF's rpy values are <= 3.2e-07 rad, which is numerical noise
     from the original DAE conversion. They are deliberately ignored, matching the
     original preparation of these meshes.)
  2. Insert ``mtllib``/``usemtl`` header lines referencing our material files.
  3. Write our custom (non-textured) material colors as ``.mtl`` files.
     The color values are our own and live in the manifest.
  4. Decimate the over-tessellated meshes to their recorded face targets
     (see ``decimate_owi535_meshes.py`` for the rationale).

meca500 (from the meca500_ros2 repo, pinned commit):
  1. Reduce the source description to visual and kinematic elements, replace
     its mesh references with paths to the generated OBJ files, and write a
     plain URDF.
  2. Load each ``.dae`` visual mesh (trimesh + pycollada), merge duplicate
     vertices, and export as OBJ referencing our in-repo ``material.lib``
     (the metallic material definitions are ours and stay in the package).

Verification
------------
Every download is checked against a pinned SHA-256, so the pipeline always
starts from exactly the recorded upstream files. Generated outputs are not
hash-checked because mesh-processing libraries are not bit-deterministic across
platforms or versions. They are checked for valid structure and recorded face
limits before installation.

Usage (from repo root)
----------------------
    python tools/fetch_assets.py --robot all
    python tools/fetch_assets.py --robot owi535 --dest-root /tmp/check
    make assets            # equivalent of --robot all

The tool is safe to re-run: robots whose assets are already present are skipped.
Pass --force to regenerate them anyway.

Requirements
------------
    pip install -e ".[assets]"      # run from the repo root; installs everything below

    meshoptimizer   (owi535 decimation)
    trimesh         (package dependency)
    pycollada       (meca500 DAE conversion)

Missing dependencies are reported up front, before anything is downloaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
ROBOTS_DIR = REPO_ROOT / "src/robot_renderer/robots"
MANIFEST_PATH = TOOLS_DIR / "asset_manifest.json"
DOWNLOADS_DIR = TOOLS_DIR / "_downloads"  # cache; safe to delete
MECA500_GENERATED_MARKER = "<!-- Generated locally by tools/fetch_assets.py. -->"

sys.path.insert(0, str(TOOLS_DIR))
import decimate_owi535_meshes as decimator  # noqa: E402  (sibling tool, reused)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _require_modules(robot: str, mods: List[Tuple[str, str]]) -> None:
    """Fail fast (before any download) if importable *mods* are missing.

    mods: list of (import_name, pip_name).
    """
    import importlib.util
    missing = [pip for mod, pip in mods if importlib.util.find_spec(mod) is None]
    if missing:
        raise RuntimeError(
            f"Missing dependencies for {robot}: {', '.join(missing)}\n"
            f"  Install with:   pip install {' '.join(missing)}\n"
            f"  (or, from the repo root:   pip install -e \".[assets]\")"
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, expected_sha: Optional[str], skip_hash: bool) -> Path:
    """Download ``url`` to ``dest`` (cached) and verify its SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp_path: Optional[Path] = None
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, tempfile.NamedTemporaryFile(
                dir=dest.parent, delete=False
            ) as tmp:
                shutil.copyfileobj(resp, tmp)
                tmp_path = Path(tmp.name)
            tmp_path.replace(dest)
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Download failed: {url}\n  ({e})\n"
                f"  The upstream host may be unavailable. See MESH_LICENSES/README.md for the\n"
                f"  provenance of this file and how to obtain it manually; place it at\n"
                f"  {dest} and re-run."
            ) from e
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
    if expected_sha and not skip_hash:
        got = _sha256(dest)
        if got != expected_sha:
            raise RuntimeError(
                f"SHA-256 mismatch for {dest.name}:\n  expected {expected_sha}\n  got      {got}\n"
                f"  The upstream file at {url} has changed since this manifest was pinned.\n"
                f"  Inspect the file, then either update tools/asset_manifest.json or\n"
                f"  re-run with --skip-hash-check if you accept the new upstream state."
            )
    return dest


def _atomic_write_text(dest: Path, text: str) -> None:
    """Write text to a temporary sibling, then replace the destination."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=dest.parent, delete=False
        ) as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)
        tmp_path.replace(dest)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _validate_mesh_arrays(
    label: str,
    verts: np.ndarray,
    faces: np.ndarray,
    max_faces: Optional[int] = None,
) -> None:
    """Raise when mesh arrays are empty, non-finite, or structurally invalid."""
    if verts.ndim != 2 or verts.shape[1:] != (3,) or len(verts) == 0:
        raise ValueError(f"{label}: expected a nonempty (N, 3) vertex array")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
        raise ValueError(f"{label}: expected a nonempty (M, 3) triangle array")
    if not np.isfinite(verts).all():
        raise ValueError(f"{label}: vertices contain non-finite values")
    if int(faces.min()) < 0 or int(faces.max()) >= len(verts):
        raise ValueError(f"{label}: triangle indices fall outside the vertex array")
    if max_faces is not None and len(faces) > max_faces:
        raise ValueError(
            f"{label}: simplification produced {len(faces)} faces, above target {max_faces}"
        )


def _validate_obj_file(path: Path, max_faces: Optional[int] = None) -> None:
    data = decimator._read_obj(path)
    _validate_mesh_arrays(str(path), data.verts, data.faces, max_faces=max_faces)


def _validate_mtl_file(path: Path) -> None:
    text = path.read_text()
    if "newmtl " not in text or "\nKd " not in text:
        raise ValueError(f"{path}: missing material name or diffuse color")


def _atomic_write_obj(path: Path, data: decimator.ObjData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        decimator._write_obj(tmp_path, data)
        tmp_path.replace(path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _dedup_mesh(verts: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Merge exactly-equal vertices, remap faces, drop degenerate triangles."""
    uniq, inverse = np.unique(verts, axis=0, return_inverse=True)
    f = inverse[faces]
    good = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return uniq, f[good]


def _decimate_meshopt(verts: np.ndarray, faces: np.ndarray, target_faces: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Quadric decimation to ``target_faces`` via meshoptimizer.

    These CAD-derived meshes are triangle soup (every vertex duplicated 2-3x,
    with many boundary edges). meshoptimizer handles this topology and is
    available for the supported Python environment.
    """
    try:
        import meshoptimizer as mo
    except ImportError as e:
        raise RuntimeError(
            "owi535 decimation needs meshoptimizer:  pip install meshoptimizer"
        ) from e
    if target_faces <= 0:
        raise ValueError(f"target_faces must be positive, got {target_faces}")
    _validate_mesh_arrays("meshoptimizer input", verts, faces)
    v, f = _dedup_mesh(verts, faces)
    _validate_mesh_arrays("meshoptimizer deduplicated input", v, f)
    v32 = np.ascontiguousarray(v, dtype=np.float32)
    idx = np.ascontiguousarray(f, dtype=np.uint32).ravel()
    dest = np.empty(len(idx), dtype=np.uint32)
    n = mo.simplify(dest, idx, v32, target_index_count=target_faces * 3,
                    target_error=0.02)
    if n <= 0 or n % 3:
        raise RuntimeError(f"meshoptimizer returned an invalid index count: {n}")
    new_f = dest[:n].reshape(-1, 3).astype(np.int64)
    # compact unused vertices
    used = np.unique(new_f)
    remap = np.full(len(v32), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    new_v = v[used].astype(np.float64)
    new_f = remap[new_f]
    _validate_mesh_arrays(
        "meshoptimizer output", new_v, new_f, max_faces=target_faces
    )
    return new_v, new_f


class Progress:
    """Single-line in-place progress counter. Only active in a terminal; when
    output is piped to a file, nothing is printed (the per-robot summary
    line carries the result)."""

    def __init__(self, total: int) -> None:
        self._total = total
        self._n = 0
        self._tty = sys.stdout.isatty()

    def step(self, name: str) -> None:
        self._n += 1
        if self._tty:
            print(f"\r  [{self._n}/{self._total}] {name[:50]:<50}", end="", flush=True)

    def close(self) -> None:
        if self._tty:
            print("\r" + " " * 60 + "\r", end="", flush=True)


# ---------------------------------------------------------------------------
# OWI-535
# ---------------------------------------------------------------------------


def _parse_visual_origins(urdf_path: Path) -> Dict[str, np.ndarray]:
    """mesh filename -> <visual> origin xyz (mm).  rpy is noise (<=3.2e-7 rad) and ignored."""
    origins: Dict[str, np.ndarray] = {}
    root = ET.parse(urdf_path).getroot()
    for link in root.iter("link"):
        for vis in link.findall("visual"):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            o = vis.find("origin")
            xyz = [float(x) for x in (o.get("xyz") if o is not None else "0 0 0").split()]
            origins[mesh.get("filename")] = np.asarray(xyz, dtype=np.float64)
    return origins


def _transform_owi_obj(src_text: str, obj_name: str, origin_xyz_mm: np.ndarray) -> str:
    """Bake visual-origin translation, convert mm->m, inject mtllib/usemtl header."""
    out: List[str] = []
    header_done = False
    for ln in src_text.splitlines():
        if not header_done and ln.startswith("#"):
            out.append(ln)
            continue
        if not header_done:
            out.append(f"mtllib {obj_name}.mtl")
            out.append("usemtl material_0")
            header_done = True
        if ln.startswith("v "):
            v = np.array([float(x) for x in ln.split()[1:4]])
            v = (v + origin_xyz_mm) * 0.001
            out.append("v " + " ".join(f"{x:.8f}" for x in v))
        else:
            out.append(ln)
    return "\n".join(out) + "\n"


_OWI_JOINT_EFFORT = "100"
_OWI_JOINT_VELOCITY = "3.14"


def _inject_owi_joint_limits(urdf_text: str) -> str:
    """Add effort/velocity to joint <limit> elements that lack them.

    roboticstoolbox requires both attributes; the upstream OWI-535 URDF omits
    them. The values only bound torque/speed, which fkine (the only kinematics
    we use) ignores, so they do not change the rendered pose. Safe to re-run:
    a <limit> that already has an attribute is left as-is.
    """
    out: List[str] = []
    for ln in urdf_text.splitlines():
        stripped = ln.strip()
        if stripped.startswith("<limit") and stripped.endswith("/>"):
            extra = ""
            if "effort=" not in ln:
                extra += f' effort="{_OWI_JOINT_EFFORT}"'
            if "velocity=" not in ln:
                extra += f' velocity="{_OWI_JOINT_VELOCITY}"'
            if extra:
                ln = ln[: ln.rfind("/>")].rstrip() + extra + "/>"
        out.append(ln)
    return "\n".join(out) + "\n"


def _write_owi_mtl(colors: dict) -> str:
    ka, kd, ks = colors["Ka"], colors["Kd"], colors["Ks"]
    return (
        "#\n# Wavefront material file\n# Converted by Meshlab Group\n#\n\n"
        "newmtl material_0\n"
        f"Ka {ka[0]:.6f} {ka[1]:.6f} {ka[2]:.6f}\n"
        f"Kd {kd[0]:.6f} {kd[1]:.6f} {kd[2]:.6f}\n"
        f"Ks {ks[0]:.6f} {ks[1]:.6f} {ks[2]:.6f}\n"
        f"Tr {colors['Tr']:.6f}\n"
        f"illum {int(colors['illum'])}\n"
        f"Ns {colors['Ns']:.6f}\n\n"
    )


def _validate_owi535_outputs(m: dict, dest: Path) -> None:
    urdf_path = dest / m["urdf"]["name"]
    root = ET.parse(urdf_path).getroot()
    for limit in root.iter("limit"):
        if limit.get("effort") is None or limit.get("velocity") is None:
            raise ValueError(
                f"{urdf_path}: a joint <limit> is missing effort/velocity "
                "(roboticstoolbox requires both)"
            )
    for name, info in m["files"].items():
        _validate_obj_file(dest / name, max_faces=info.get("decimate_target"))
        _validate_mtl_file(dest / f"{name}.mtl")


def owi535_outputs_present(m: dict, dest_root: Path) -> bool:
    """True if every OWI output exists and is structurally valid."""
    dest = dest_root / "owi535"
    expected = [dest / m["urdf"]["name"]]
    expected.extend(dest / name for name in m["files"])
    expected.extend(dest / f"{name}.mtl" for name in m["files"])
    if not all(path.is_file() for path in expected):
        return False
    try:
        _validate_owi535_outputs(m, dest)
    except (OSError, ValueError, IndexError, ET.ParseError) as exc:
        print(f"== owi535 ==  existing assets are invalid; regenerating ({exc})")
        return False
    return True


def fetch_owi535(manifest: dict, dest_root: Path, skip_hash: bool) -> None:
    m = manifest["owi535"]
    if any(info.get("decimate_target") for info in m["files"].values()):
        _require_modules("owi535", [("meshoptimizer", "meshoptimizer")])
    base = m["base_url"].rstrip("/")
    dest = dest_root / "owi535"
    dest.mkdir(parents=True, exist_ok=True)
    dl = DOWNLOADS_DIR / "owi535"

    print(f"\n== owi535 ==  fetching URDF and {len(m['files'])} meshes  ->  {dest}")

    urdf_local = _download(f"{base}/{m['urdf']['name']}", dl / m["urdf"]["name"],
                           m["urdf"]["sha256"], skip_hash)
    origins = _parse_visual_origins(urdf_local)
    urdf_text = _inject_owi_joint_limits(urdf_local.read_text())
    _atomic_write_text(dest / m["urdf"]["name"], urdf_text)

    n_decimated = 0
    progress = Progress(len(m["files"]))
    for name, info in m["files"].items():
        progress.step(name)
        src = _download(f"{base}/{name}", dl / name, info["sha256"], skip_hash)
        text = _transform_owi_obj(src.read_text(), name, origins[name])
        out_path = dest / name
        _atomic_write_text(out_path, text)

        # material file (our colors)
        _atomic_write_text(dest / f"{name}.mtl", _write_owi_mtl(info["colors"]))

        target = info.get("decimate_target")
        if target is not None:
            n_decimated += 1
            data = decimator._read_obj(out_path)
            _validate_mesh_arrays(str(out_path), data.verts, data.faces)
            new_v, new_f = _decimate_meshopt(data.verts, data.faces, target)
            _atomic_write_obj(
                out_path, decimator.ObjData(data.header, new_v, new_f)
            )
    progress.close()
    _validate_owi535_outputs(m, dest)

    print(f"  ok: URDF and {len(m['files'])} meshes installed "
          f"({n_decimated} simplified); all outputs structurally valid")


# ---------------------------------------------------------------------------
# Meca500
# ---------------------------------------------------------------------------


def _transform_meca500_urdf(source_text: str, m: dict) -> str:
    """Create the renderer URDF from the pinned upstream description."""
    root = ET.fromstring(source_text)
    if root.tag != "robot":
        raise ValueError("Meca500 description root must be <robot>")
    root.set("name", "meca500")

    expected_parts = set(m["files"])
    seen_parts: List[str] = []
    for link in root.findall("link"):
        for collision in link.findall("collision"):
            link.remove(collision)
        for mesh in link.findall("visual/geometry/mesh"):
            filename = mesh.get("filename")
            if not filename or not filename.endswith(".dae"):
                raise ValueError(f"Meca500 visual has an unexpected mesh path: {filename}")
            part = filename.rsplit("/", 1)[-1][:-4]
            if part not in expected_parts:
                raise ValueError(f"Meca500 description contains an unknown visual mesh: {part}")
            mesh.set("filename", f"meshes/visual/{part}/{part}.obj")
            seen_parts.append(part)

    if len(seen_parts) != len(expected_parts) or set(seen_parts) != expected_parts:
        raise ValueError(
            "Meca500 description visual meshes do not match the asset manifest"
        )

    ET.indent(root, space="  ")
    return (
        '<?xml version="1.0"?>\n'
        + MECA500_GENERATED_MARKER
        + "\n"
        + ET.tostring(root, encoding="unicode")
        + "\n"
    )


def _validate_meca500_outputs(m: dict, dest: Path) -> None:
    urdf_path = dest / m["urdf"]["name"]
    if MECA500_GENERATED_MARKER not in urdf_path.read_text():
        raise ValueError(f"{urdf_path}: not generated by the current asset pipeline")
    root = ET.parse(urdf_path).getroot()
    if root.tag != "robot" or root.get("name") != "meca500":
        raise ValueError(f"{urdf_path}: expected the generated Meca500 robot description")
    if root.findall(".//collision"):
        raise ValueError(f"{urdf_path}: generated URDF still contains collision elements")

    expected_meshes = {
        f"meshes/visual/{part}/{part}.obj" for part in m["files"]
    }
    actual_meshes = {
        mesh.get("filename") for mesh in root.findall(".//visual/geometry/mesh")
    }
    if actual_meshes != expected_meshes:
        raise ValueError(f"{urdf_path}: visual meshes do not match the asset manifest")

    visual_dest = dest / "meshes/visual"
    for part in m["files"]:
        _validate_obj_file(visual_dest / part / f"{part}.obj")


def meca500_outputs_present(m: dict, dest_root: Path) -> bool:
    """True if every Meca500 output exists and is structurally valid."""
    dest = dest_root / "meca500"
    visual_dest = dest / "meshes/visual"
    expected = [dest / m["urdf"]["name"]]
    expected.extend(visual_dest / part / f"{part}.obj" for part in m["files"])
    if not all(path.is_file() for path in expected):
        return False
    try:
        _validate_meca500_outputs(m, dest)
    except (OSError, ValueError, IndexError, ET.ParseError) as exc:
        print(f"== meca500 ==  existing assets are invalid; regenerating ({exc})")
        return False
    return True


def fetch_meca500(manifest: dict, dest_root: Path, skip_hash: bool) -> None:
    _require_modules("meca500", [("trimesh", "trimesh"), ("collada", "pycollada")])
    import trimesh

    m = manifest["meca500"]
    base = f"{m['raw_base']}/{m['commit']}"
    mesh_base = f"{base}/{m['mesh_path']}"
    dest = dest_root / "meca500"
    visual_dest = dest / "meshes/visual"
    dl = DOWNLOADS_DIR / "meca500"

    print(f"\n== meca500 ==  fetching URDF source and {len(m['files'])} meshes "
          f"(pinned commit {m['commit'][:10]})  ->  {dest}")

    urdf_info = m["urdf"]
    source_name = urdf_info["source_path"].rsplit("/", 1)[-1]
    urdf_source = _download(
        f"{base}/{urdf_info['source_path']}",
        dl / source_name,
        urdf_info["sha256"],
        skip_hash,
    )
    _atomic_write_text(
        dest / urdf_info["name"],
        _transform_meca500_urdf(urdf_source.read_text(), m),
    )

    progress = Progress(len(m["files"]))
    for part, info in m["files"].items():
        progress.step(part)
        dae = _download(
            f"{mesh_base}/{part}.dae",
            dl / f"{part}.dae",
            info["sha256"],
            skip_hash,
        )
        mesh = trimesh.load(dae, force="mesh")
        mesh.merge_vertices()
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        _validate_mesh_arrays(f"{part} converted DAE", verts, faces)

        out_dir = visual_dest / part
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"# Meca500 R3 {part}", "mtllib material.lib", "usemtl mat"]
        lines.extend(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in verts)
        lines.extend(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}" for f in faces)
        out_path = out_dir / f"{part}.obj"
        _atomic_write_text(out_path, "\n".join(lines) + "\n")
        written = decimator._read_obj(out_path)
        _validate_mesh_arrays(str(out_path), written.verts, written.faces)
        if len(written.faces) != len(faces):
            raise RuntimeError(
                f"{out_path}: wrote {len(written.faces)} faces, expected {len(faces)}"
            )
        expected_bbox = np.array([verts.min(0), verts.max(0)])
        written_bbox = np.array([written.verts.min(0), written.verts.max(0)])
        if not np.allclose(written_bbox, expected_bbox, atol=1e-6):
            raise RuntimeError(f"{out_path}: bounding box changed while writing OBJ")
    progress.close()
    _validate_meca500_outputs(m, dest)

    print(f"  ok: URDF and {len(m['files'])} meshes generated and structurally valid")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--robot", choices=["owi535", "meca500", "all"], default="all")
    parser.add_argument("--dest-root", type=Path, default=ROBOTS_DIR,
                        help="Install root (default: the package's robots/ directory). "
                             "Assets land below <dest-root>/owi535 and <dest-root>/meca500.")
    parser.add_argument("--skip-hash-check", action="store_true",
                        help="Accept upstream files whose SHA-256 no longer matches the manifest.")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if a robot's meshes are already present "
                             "(default: already-complete robots are skipped).")
    parser.add_argument("--clear-cache", action="store_true",
                        help=f"Delete the download cache ({DOWNLOADS_DIR}) first.")
    args = parser.parse_args(argv)

    if args.clear_cache and DOWNLOADS_DIR.exists():
        shutil.rmtree(DOWNLOADS_DIR)

    manifest = json.loads(MANIFEST_PATH.read_text())
    n_fetched = 0

    if args.robot in ("owi535", "all"):
        if not args.force and owi535_outputs_present(manifest["owi535"], args.dest_root):
            print("== owi535 ==  assets already present, skipping (use --force to regenerate)")
        else:
            fetch_owi535(manifest, args.dest_root, args.skip_hash_check)
            n_fetched += 1
    if args.robot in ("meca500", "all"):
        if not args.force and meca500_outputs_present(manifest["meca500"], args.dest_root):
            print("== meca500 ==  assets already present, skipping (use --force to regenerate)")
        else:
            fetch_meca500(manifest, args.dest_root, args.skip_hash_check)
            n_fetched += 1

    print()
    if n_fetched:
        print(f"--- done: {n_fetched} robot(s) fetched and converted ---")
    else:
        print("--- nothing to do, all assets already present ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
