"""
Wavefront OBJ + companion MTL (including *.lib) to mesh with per-face diffuse colors.

Fallback MTL parser used by RobotPyTorch3DRenderer when PyTorch3D's built-in loader
fails to parse non-standard MTL filenames (e.g. material.lib instead of material.mtl).
Trimesh also often mis-handles these exports (broken TextureVisuals / single grey fallback
instead of usemtl + Kd assignments).  We parse mtllib/usemtl/Kd ourselves and expand
vertices per triangle so each face corner can carry its material colour.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_SPLIT_RE = re.compile(r"\s+")

# Module-level cache. Key = (abs_path, fallback_rgb_tuple) so different color_mapping
# values for the same file don't collide. Mesh files don't change at runtime.
_PARSE_CACHE: Dict[tuple, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}


def _parse_mtl_file(path: Path) -> Dict[str, np.ndarray]:
    """Return material names mapped to diffuse RGB in [0, 1] (float64)."""
    materials: Dict[str, np.ndarray] = {}
    current: Optional[str] = None
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for raw in text:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("newmtl"):
            current = line[6:].strip()
            continue
        if current is None:
            continue
        if line.startswith("Kd"):
            parts = _SPLIT_RE.split(line)
            if len(parts) >= 4:
                materials[current] = np.clip(
                    np.array(
                        [float(parts[1]), float(parts[2]), float(parts[3])],
                        dtype=np.float64,
                    ),
                    0.0,
                    1.0,
                )
    return materials


def _load_mtl_libraries(obj_dir: Path, mtllib_names: List[str]) -> Dict[str, np.ndarray]:
    merged: Dict[str, np.ndarray] = {}
    for name in mtllib_names:
        mpath = obj_dir / name
        if mpath.is_file():
            merged.update(_parse_mtl_file(mpath))
    return merged


def load_obj_with_mtl_vertex_colors(
    obj_path: str,
    default_rgb: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load a triangle mesh from OBJ, applying MTL Kd per face (via usemtl).

    Vertices are duplicated per triangle so each corner can carry the face material colour.
    The exploded topology is then mapped back onto the shared PT3D vertex layout by the caller.

    Results are cached by absolute path, since mesh files do not change at runtime.

    Args:
        obj_path:    Path to the .obj file.
        default_rgb: (3,) diffuse RGB in [0, 1] when material is missing.

    Returns:
        (verts, faces, vertex_colors) or None if the file is not a usable OBJ.
    """
    path = Path(obj_path)
    cache_key = (str(path.resolve()), tuple(float(v) for v in default_rgb))
    if cache_key in _PARSE_CACHE:
        return _PARSE_CACHE[cache_key]
    if not path.is_file():
        return None

    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    mtllib_names: List[str] = []
    vertices: List[List[float]] = []
    # (v0, v1, v2) 0-based indices into *vertices*, plus material name (or None)
    faces_mtl: List[Tuple[int, int, int, Optional[str]]] = []
    current_mtl: Optional[str] = None

    for raw in text:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = _SPLIT_RE.split(line)
        cmd = parts[0]

        if cmd == "mtllib":
            mtllib_names.extend(parts[1:])
        elif cmd == "v" and len(parts) >= 4:
            vertices.append(
                [float(parts[1]), float(parts[2]), float(parts[3])]
            )
        elif cmd == "usemtl":
            current_mtl = " ".join(parts[1:]).strip() or None
        elif cmd == "f":
            corners = parts[1:]
            if len(corners) < 3:
                continue
            idxs: List[int] = []
            for c in corners:
                vi_str = c.split("/")[0]
                if not vi_str:
                    continue
                vi = int(vi_str)
                if vi < 0:
                    vi = len(vertices) + vi + 1
                idxs.append(vi - 1)
            if len(idxs) < 3:
                continue
            if len(idxs) == 3:
                faces_mtl.append((idxs[0], idxs[1], idxs[2], current_mtl))
            elif len(idxs) == 4:
                faces_mtl.append((idxs[0], idxs[1], idxs[2], current_mtl))
                faces_mtl.append((idxs[0], idxs[2], idxs[3], current_mtl))
            else:
                for k in range(1, len(idxs) - 1):
                    faces_mtl.append((idxs[0], idxs[k], idxs[k + 1], current_mtl))

    if not vertices or not faces_mtl:
        return None

    v_count = len(vertices)
    materials = _load_mtl_libraries(path.parent, mtllib_names)
    default_rgb = np.asarray(default_rgb, dtype=np.float64).reshape(3)
    default_rgb = np.clip(default_rgb, 0.0, 1.0)

    verts_np = np.asarray(vertices, dtype=np.float64)

    new_verts: List[np.ndarray] = []
    new_faces: List[List[int]] = []
    new_colors: List[np.ndarray] = []
    offset = 0

    for i0, i1, i2, mtl in faces_mtl:
        for ix in (i0, i1, i2):
            if ix < 0 or ix >= v_count:
                return None
        if mtl is not None and mtl in materials:
            kd = materials[mtl]
        else:
            kd = default_rgb
        new_verts.append(verts_np[i0])
        new_verts.append(verts_np[i1])
        new_verts.append(verts_np[i2])
        for _ in range(3):
            new_colors.append(kd.astype(np.float64, copy=False))
        new_faces.append([offset, offset + 1, offset + 2])
        offset += 3

    result = (
        np.asarray(new_verts, dtype=np.float64),
        np.asarray(new_faces, dtype=np.int32),
        np.asarray(new_colors, dtype=np.float64),
    )
    _PARSE_CACHE[cache_key] = result
    return result
