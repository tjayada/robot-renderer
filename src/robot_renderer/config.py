from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ViewConfig:
    """
    Parameters controlling template rendering.

    render_size:
        Square output resolution (H=W) for rendered templates.
    viewset:
        Name of the view-generation strategy.
        Options: "fibonacci_N" (e.g. "fibonacci_256"), "four_sides",
        "eight_sides", "ring_8", "hemisphere_8".
    num_views:
        Final number of template views to return.  For fibonacci viewsets
        this is the target after scoring + diverse selection.  For fixed
        presets (four_sides etc.) it is ignored.
    sphere_distance_factor:
        Camera distance = bounding_sphere_radius * this factor.
        Increase for more context, decrease to fill the frame more.
    elevation_range:
        (min_deg, max_deg) elevation band for fibonacci sphere sampling.
        Restricting this avoids degenerate top-down / bottom-up views.
    orientation:
        How to orient the robot mesh before rendering.
        "simple_upright":                    align robot Z-axis to Y-up.
        "align_mesh_longest_axis_upright":   PCA longest axis to Y-up.
        "align_mesh_longest_axis_top_right": PCA longest axis to upper-right (45 degree diagonal).
        "end_effector_frame":                rotate into the EE orientation
                                              (pivot = EE origin; no re-centring).
    diverse_selection:
        When True: anchor = highest-scored view (or best within
        anchor_elevation_range if set), remaining slots filled by Farthest
        Point Sampling on the sphere for angular diversity.
        When False: plain top-N by Lambert score.
    anchor_elevation_range:
        Optional (min_deg, max_deg) band that constrains which view can be
        picked as the anchor (index 0 in the final list). The anchor will be
        the highest-scored view whose elevation falls within this range; all
        other views are still drawn from the full elevation_range. Useful
        when a robot has a large base that tends to dominate bottom-up views
        and get selected as anchor (e.g. OWI-535). Set e.g. (-5.0, 5.0) to
        force a front-on anchor. Has no effect when diverse_selection=False.
    backgrounds:
        List of [R,G,B] background colours (values in [0,1]).
        Cycled across views.
    fill_frame:
        When True, each rendered view is independently cropped to its object
        silhouette (square, padded, shifted to stay in-frame) and resized back
        to render_size, so every template fills the frame to a consistent
        fraction regardless of viewing angle or joint configuration.
        This changes the image framing without changing the returned camera
        poses or depth maps. Default off.
    fill_frame_pad:
        Relative padding around the silhouette bbox before the square crop
        (0.0 is a tight crop; 0.1 to 0.2 adds visible context).
    fill_frame_min_px:
        Minimum silhouette pixel count for a view to be cropped; below this the
        view is left full-frame (degenerate edge-on / barely-visible views).
    """
    render_size: int = 448
    viewset: str = "fibonacci_256"
    num_views: int = 8
    sphere_distance_factor: float = 2.5
    elevation_range: Tuple[float, float] = (-5.0, 5.0)
    orientation: str = "simple_upright"
    diverse_selection: bool = True
    anchor_elevation_range: Optional[Tuple[float, float]] = None
    backgrounds: Optional[List[List[float]]] = None
    fill_frame: bool = False
    fill_frame_pad: float = 0.1
    fill_frame_min_px: int = 64

    def __post_init__(self):
        if self.backgrounds is None:
            self.backgrounds = [
                [0.25, 0.25, 0.25], [0.35, 0.35, 0.35], [0.20, 0.20, 0.25],
                [0.30, 0.28, 0.25], [0.28, 0.28, 0.28], [0.32, 0.32, 0.38],
                [0.22, 0.25, 0.22], [0.30, 0.24, 0.24],
            ]

    @property
    def candidate_views(self) -> List[Tuple[float, float]]:
        """Candidate (azimuth, elevation) pairs for the configured viewset.

        Built on each access (cheap, pure trigonometry), so edits to
        ``viewset`` / ``elevation_range`` always take effect.
        """
        from .view_strategy import build_viewset
        return build_viewset(
            self.viewset, self.num_views, elevation_range=self.elevation_range
        )
