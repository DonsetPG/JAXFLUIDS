from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class BezierShape:
    control_points: np.ndarray
    contour_points: np.ndarray
    levelset: np.ndarray


def _cubic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    omt = 1.0 - t
    return (
        (omt ** 3)[:, None] * p0[None, :]
        + (3.0 * omt ** 2 * t)[:, None] * p1[None, :]
        + (3.0 * omt * t ** 2)[:, None] * p2[None, :]
        + (t ** 3)[:, None] * p3[None, :]
    )


def bezier_closed_from_four_points(
    control_points: np.ndarray,
    samples_per_segment: int = 128,
    tension: float = 1.0,
) -> np.ndarray:
    """Builds a smooth closed Bezier contour through 4 points.

    The contour is composed of 4 cubic Bezier segments, where the tangent
    direction at each control point is inferred from neighboring points.
    """
    points = np.asarray(control_points, dtype=np.float64)
    assert points.shape == (4, 2), "Expected 4 control points of shape (4, 2)."
    t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)

    segments = []
    for i in range(4):
        p_im1 = points[(i - 1) % 4]
        p_i = points[i]
        p_ip1 = points[(i + 1) % 4]
        p_ip2 = points[(i + 2) % 4]

        tangent_i = 0.5 * (p_ip1 - p_im1)
        tangent_ip1 = 0.5 * (p_ip2 - p_i)
        handle_1 = p_i + (tension / 3.0) * tangent_i
        handle_2 = p_ip1 - (tension / 3.0) * tangent_ip1

        segment = _cubic_bezier(p_i, handle_1, handle_2, p_ip1, t)
        segments.append(segment)

    contour = np.vstack(segments)
    contour = np.vstack([contour, contour[:1]])
    return contour


def _point_segment_distance(
    px: np.ndarray,
    py: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> np.ndarray:
    vx = x1 - x0
    vy = y1 - y0
    denom = vx * vx + vy * vy + 1.0e-14
    t = ((px - x0) * vx + (py - y0) * vy) / denom
    t = np.clip(t, 0.0, 1.0)
    cx = x0 + t * vx
    cy = y0 + t * vy
    return np.sqrt((px - cx) ** 2 + (py - cy) ** 2)


def _inside_polygon(px: np.ndarray, py: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x = polygon[:, 0]
    y = polygon[:, 1]
    x0 = x[:-1]
    y0 = y[:-1]
    x1 = x[1:]
    y1 = y[1:]

    cond = ((y0 <= py[..., None]) & (y1 > py[..., None])) | (
        (y0 > py[..., None]) & (y1 <= py[..., None])
    )
    xints = x0 + (py[..., None] - y0) * (x1 - x0) / (y1 - y0 + 1.0e-14)
    crossings = np.sum(cond & (px[..., None] < xints), axis=-1)
    return (crossings % 2) == 1


def signed_distance_from_contour(
    contour_points: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    xx, yy = np.meshgrid(x, y, indexing="ij")
    min_distance = np.full_like(xx, np.inf, dtype=np.float64)
    for i in range(contour_points.shape[0] - 1):
        x0, y0 = contour_points[i]
        x1, y1 = contour_points[i + 1]
        dist = _point_segment_distance(xx, yy, x0, y0, x1, y1)
        min_distance = np.minimum(min_distance, dist)

    inside = _inside_polygon(xx, yy, contour_points)
    sign = np.where(inside, -1.0, 1.0)
    return sign * min_distance


def write_levelset_h5(levelset_xy: np.ndarray, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    levelset_xyz = levelset_xy[..., None].astype(np.float64)
    with h5py.File(out_path, "w") as h5file:
        h5file.create_dataset("levelset", data=levelset_xyz)
    return out_path


def render_shape_preview(
    contour_points: np.ndarray,
    control_points: np.ndarray,
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(contour_points[:, 0], contour_points[:, 1], color="tab:blue", lw=2.0)
    ax.scatter(control_points[:, 0], control_points[:, 1], color="tab:orange", zorder=5)
    for i, (xc, yc) in enumerate(control_points):
        ax.text(xc, yc, f"P{i}", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("4-point closed Bezier shape")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def clamp_points_to_ring(
    points: np.ndarray,
    r_min: float = 0.7,
    r_max: float = 1.3,
) -> np.ndarray:
    """Constrains points to the ring used in the paper-style setup."""
    p = np.asarray(points, dtype=np.float64).copy()
    radii = np.linalg.norm(p, axis=1) + 1.0e-14
    clamped_radii = np.clip(radii, r_min, r_max)
    p *= (clamped_radii / radii)[:, None]
    return p
