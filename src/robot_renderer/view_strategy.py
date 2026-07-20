from __future__ import annotations
import math
from typing import List, Optional, Tuple

import torch


def build_viewset(
    name: str,
    num_views: int,
    elevation_range: Tuple[float, float] = (-90.0, 90.0),
) -> List[Tuple[float, float]]:
    """
    Return a list of (azimuth_deg, elevation_deg) view directions.

    Supports named presets and Fibonacci sphere sampling:
      "fibonacci_N":  N near-uniform directions filtered to elevation_range.
      "four_sides":   4 azimuths at 0/90/180/270 degrees, elevation 0.
      "eight_sides":  8 azimuths at 45 degree increments, elevation 0.
      "ring_8":       alias for eight_sides.
      "hemisphere_8": 4 equatorial + 4 at 30 degrees elevation.

    Raises:
        ValueError: for any other name (misspelled viewsets must not silently
        fall back to a different sampling strategy).
    """
    views: List[Tuple[float, float]] = []
    name_lower = name.lower()

    if name_lower == "four_sides":
        for az in [0.0, 90.0, 180.0, 270.0]:
            views.append((az, 0.0))
    elif name_lower in ("eight_sides", "ring_8"):
        for az in [45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 0.0]:
            views.append((az, 0.0))
    elif name_lower == "hemisphere_8":
        for az in [0.0, 90.0, 180.0, 270.0]:
            views.append((az, 0.0))
        for az in [0.0, 90.0, 180.0, 270.0]:
            views.append((az, 30.0))
    elif name_lower.startswith("fibonacci"):
        try:
            n_target = int(name_lower.split("_")[1])
        except (IndexError, ValueError):
            n_target = num_views
        el_span = elevation_range[1] - elevation_range[0]
        sphere_fraction = max(el_span / 180.0, 0.05)
        n_generate = int(math.ceil(n_target / sphere_fraction))
        dirs = fibonacci_sphere_directions(n_generate, elevation_range=elevation_range)
        if len(dirs) > n_target:
            idx = torch.linspace(0, len(dirs) - 1, n_target).long()
            dirs = dirs[idx]
        views = [direction_to_azim_elev(d) for d in dirs]
    else:
        raise ValueError(
            f"Unknown viewset {name!r}. Choose 'fibonacci_N' (e.g. 'fibonacci_256'), "
            "'four_sides', 'eight_sides', 'ring_8', or 'hemisphere_8'."
        )

    return views


def fibonacci_sphere_directions(
    n: int,
    elevation_range: Tuple[float, float] = (-90.0, 90.0),
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Generate n near-uniform directions on a unit sphere using the golden-angle
    (Fibonacci) spiral, optionally filtered to an elevation band.

    Returns:
        directions: (M, 3) unit vectors where M <= n after elevation filtering.
    """
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    indices = torch.arange(0, n, dtype=torch.float64)
    theta = 2.0 * math.pi * indices / golden_ratio
    z = 1.0 - 2.0 * (indices + 0.5) / n
    r_xy = torch.sqrt(1.0 - z * z)
    x = r_xy * torch.cos(theta)
    y = r_xy * torch.sin(theta)
    dirs = torch.stack([x, y, z], dim=-1).to(torch.float32)

    if elevation_range != (-90.0, 90.0):
        el_min_rad = math.radians(elevation_range[0])
        el_max_rad = math.radians(elevation_range[1])
        elev = torch.asin(dirs[:, 1].clamp(-1.0, 1.0))
        mask = (elev >= el_min_rad) & (elev <= el_max_rad)
        dirs = dirs[mask]

    if device is not None:
        dirs = dirs.to(device)
    return dirs


def direction_to_azim_elev(d: torch.Tensor) -> Tuple[float, float]:
    """Convert a (3,) unit direction to (azimuth_deg, elevation_deg)."""
    x, y, z = float(d[0]), float(d[1]), float(d[2])
    elev = math.degrees(math.asin(max(-1.0, min(1.0, y))))
    azim = math.degrees(math.atan2(x, z))
    return azim, elev


def score_views_by_visible_area(
    verts: torch.Tensor,
    faces: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    """
    Score each view direction by Lambert cosine-weighted visible surface area
    combined with a depth-range compactness penalty.

    Args:
        verts:      (V, 3) float32 mesh vertices.
        faces:      (F, 3) int64/int32 face indices.
        directions: (N, 3) unit-length candidate view directions.

    Returns:
        scores: (N,); higher is better.
    """
    faces = faces.to(dtype=torch.long, device=verts.device)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    cross = torch.linalg.cross(v1 - v0, v2 - v0)
    cross_norm = cross.norm(dim=-1, keepdim=True)  # compute once, reuse for area and normals
    area = cross_norm * 0.5
    normals = cross / (cross_norm + 1e-8)

    dirs = directions.to(dtype=normals.dtype, device=normals.device)
    dots = (normals @ dirs.T).clamp(min=0.0)
    scores = (area * dots).sum(dim=0)

    projections = verts @ dirs.T
    depth_ranges = projections.max(dim=0).values - projections.min(dim=0).values
    sigma = (verts - verts.mean(dim=0)).norm(dim=1).max().clamp(min=1e-6)
    return scores * torch.exp(-depth_ranges / sigma)


def _views_to_dir_tensor(
    views: List[Tuple[float, float]],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert a list of (azimuth_deg, elevation_deg) pairs to (N, 3) unit directions."""
    dirs = [
        [math.cos(math.radians(el)) * math.sin(math.radians(az)),
         math.sin(math.radians(el)),
         math.cos(math.radians(el)) * math.cos(math.radians(az))]
        for az, el in views
    ]
    t = torch.tensor(dirs, dtype=torch.float32)
    return t.to(device) if device is not None else t


def score_and_sort_views(
    views: List[Tuple[float, float]],
    verts: torch.Tensor,
    faces: torch.Tensor,
) -> List[Tuple[float, float]]:
    """
    Score a list of (azimuth_deg, elevation_deg) views against a mesh and
    return them sorted best-first (descending score).  O((F+V)*N), no rendering.

    Args:
        views: List of (azimuth_deg, elevation_deg) pairs.
        verts: (V, 3) float32 mesh vertices.
        faces: (F, 3) int mesh face indices.
    """
    dirs_t = _views_to_dir_tensor(views, device=verts.device)
    scores = score_views_by_visible_area(verts, faces, dirs_t)
    order = scores.argsort(descending=True).tolist()
    return [views[i] for i in order]


def select_diverse_views(
    views: List[Tuple[float, float]],
    num_views: int,
    anchor_elevation_range: Optional[Tuple[float, float]] = None,
    device: Optional[torch.device] = None,
) -> List[Tuple[float, float]]:
    """
    Select num_views from a pre-scored (best-first) list using anchor + FPS:
      - The anchor (index 0 in output) is the highest-scored view, optionally
        constrained to anchor_elevation_range. If the range is set and no view
        falls within it, the overall best-scored view is used as fallback.
      - Remaining slots are filled by Farthest Point Sampling on the unit
        sphere, maximising minimum angular distance between selected views.
    """
    if len(views) <= num_views:
        return views

    dirs_t = _views_to_dir_tensor(views, device=device)

    # Pick anchor: best-scored view within anchor_elevation_range, else views[0].
    anchor_idx = 0
    if anchor_elevation_range is not None:
        el_min, el_max = anchor_elevation_range
        for i, (_, el) in enumerate(views):
            if el_min <= el <= el_max:
                anchor_idx = i
                break

    selected = [anchor_idx]
    cos_to_anchor = dirs_t @ dirs_t[anchor_idx]
    min_cos_dist = 1.0 - cos_to_anchor
    min_cos_dist[anchor_idx] = float("-inf")

    for _ in range(num_views - 1):
        next_idx = int(min_cos_dist.argmax().item())
        selected.append(next_idx)
        min_cos_dist[next_idx] = float("-inf")  # exclude from future selection
        cos_sim = dirs_t @ dirs_t[next_idx]
        min_cos_dist = torch.minimum(min_cos_dist, 1.0 - cos_sim)

    return [views[i] for i in selected]
