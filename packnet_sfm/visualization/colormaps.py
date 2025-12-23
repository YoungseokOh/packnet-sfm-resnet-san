"""Shared colormap helpers.

The project historically had multiple ad-hoc implementations of a depth colormap
("red=near → blue=far") spread across visualization scripts. This module
centralizes that logic so all scripts can import the same function.

Usage:
    from packnet_sfm.visualization.colormaps import create_custom_depth_colormap
    cmap = create_custom_depth_colormap(min_depth=0.1, max_depth=10.0)

Notes:
- Returned object is a Matplotlib `LinearSegmentedColormap`.
- Inputs are *metric depth* values (meters).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from matplotlib.colors import LinearSegmentedColormap


RGB = Tuple[float, float, float]
DepthPoint = Tuple[float, RGB]


@dataclass(frozen=True)
class DepthColormapSpec:
    """Specification for a depth colormap defined by absolute depth control points."""

    name: str
    points: Sequence[DepthPoint]


_DEFAULT_DEPTH_POINTS: Sequence[DepthPoint] = (
    (0.1, (1.0, 0.0, 0.0)),  # Pure red
    (0.3, (1.0, 0.0, 0.0)),
    (0.4, (1.0, 0.15, 0.0)),
    (0.5, (1.0, 0.35, 0.0)),
    (0.6, (1.0, 0.5, 0.0)),
    (0.8, (1.0, 0.55, 0.0)),
    (1.0, (1.0, 0.6, 0.0)),
    (1.1, (1.0, 0.7, 0.0)),
    (1.25, (1.0, 0.85, 0.0)),
    (1.4, (1.0, 1.0, 0.0)),
    (1.8, (1.0, 1.0, 0.0)),
    (2.2, (0.9, 1.0, 0.0)),
    (2.4, (0.7, 1.0, 0.1)),
    (2.5, (0.5, 1.0, 0.2)),
    (2.7, (0.3, 1.0, 0.3)),
    (3.0, (0.1, 1.0, 0.4)),
    (3.3, (0.0, 1.0, 0.5)),
    (3.5, (0.0, 1.0, 0.7)),
    (3.8, (0.0, 1.0, 0.85)),
    (4.5, (0.0, 1.0, 1.0)),
    (5.5, (0.0, 0.9, 1.0)),
    (6.5, (0.0, 0.7, 1.0)),
    (7.0, (0.0, 0.5, 1.0)),
    (8.0, (0.0, 0.3, 1.0)),
    (10.0, (0.0, 0.15, 1.0)),
    (12.0, (0.0, 0.05, 1.0)),
    (15.0, (0.0, 0.0, 1.0)),  # Pure blue
)

DEFAULT_DEPTH_CMAP_SPEC = DepthColormapSpec(
    name="depth_custom",
    points=_DEFAULT_DEPTH_POINTS,
)


def _validate_points(points: Sequence[DepthPoint]) -> None:
    if len(points) < 2:
        raise ValueError("Colormap spec must have at least 2 points")
    depths = [d for d, _ in points]
    if any(depths[i] > depths[i + 1] for i in range(len(depths) - 1)):
        raise ValueError("Colormap depth points must be sorted ascending")


def create_custom_depth_colormap(
    min_depth: float = 0.1,
    max_depth: float = 15.0,
    *,
    spec: DepthColormapSpec = DEFAULT_DEPTH_CMAP_SPEC,
    n: int = 512,
) -> LinearSegmentedColormap:
    """Create a custom progressive colormap (red=near, blue=far).

    Args:
        min_depth: Minimum depth (meters) that maps to position 0.
        max_depth: Maximum depth (meters) that maps to position 1.
        spec: Absolute-depth control points.
        n: Number of discrete colors in the returned colormap.

    Returns:
        A Matplotlib `LinearSegmentedColormap`.

    Behavior:
    - Control points outside [min_depth, max_depth] are dropped.
    - If `min_depth`/`max_depth` are not explicitly present in the remaining
      control points, boundary points are inserted using the nearest color
      on each side (consistent with the previous script implementations).
    """

    if max_depth <= min_depth:
        raise ValueError(f"max_depth must be > min_depth (got {min_depth}..{max_depth})")

    _validate_points(spec.points)

    depth_range = max_depth - min_depth

    # Keep only points inside the requested range
    depth_points: List[DepthPoint] = [(d, c) for d, c in spec.points if min_depth <= d <= max_depth]

    # Ensure start point
    if len(depth_points) == 0 or depth_points[0][0] > min_depth:
        # Find the first point in the absolute spec that is >= min_depth
        inserted = False
        for d, c in spec.points:
            if d >= min_depth:
                depth_points.insert(0, (min_depth, c))
                inserted = True
                break
        if not inserted:
            # If min_depth is beyond the last control point, reuse the last color
            depth_points.insert(0, (min_depth, spec.points[-1][1]))

    # Ensure end point
    if depth_points[-1][0] < max_depth:
        depth_points.append((max_depth, spec.points[-1][1]))

    positions = [(d - min_depth) / depth_range for d, _ in depth_points]
    colors = [c for _, c in depth_points]

    # Numerical safety: clamp boundaries exactly to [0,1]
    if positions[0] != 0.0:
        positions[0] = 0.0
    if positions[-1] != 1.0:
        positions[-1] = 1.0

    return LinearSegmentedColormap.from_list(spec.name, list(zip(positions, colors)), N=n)
