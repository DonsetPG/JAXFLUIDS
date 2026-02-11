from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from jaxfluids import InputManager

from bezier_shape import (
    bezier_closed_from_four_points,
    clamp_points_to_ring,
    render_shape_preview,
    signed_distance_from_contour,
    write_levelset_h5,
)


def get_cell_centers(case_setup: str, numerical_setup: str) -> tuple[np.ndarray, np.ndarray]:
    input_manager = InputManager(case_setup, numerical_setup)
    x_raw, y_raw, _ = input_manager.domain_information.get_global_cell_centers_unsplit()
    x = np.asarray(x_raw).reshape(-1)
    y = np.asarray(y_raw).reshape(-1)
    return x, y


def sample_four_points(rng: np.random.Generator) -> np.ndarray:
    base_angles = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi], dtype=np.float64)
    angle_noise = rng.uniform(-0.20, 0.20, size=4)
    radii = rng.uniform(0.78, 1.22, size=4)
    angles = base_angles + angle_noise
    points = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=-1)
    return clamp_points_to_ring(points, r_min=0.7, r_max=1.3)


def main() -> None:
    case_setup = "case_setup_viq_re200.json"
    numerical_setup = "numerical_setup.json"

    out_dir = Path("generated_shapes_4pt_bezier")
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y = get_cell_centers(case_setup, numerical_setup)
    rng = np.random.default_rng(7)

    rows = []
    for i in range(10):
        control_points = sample_four_points(rng)
        contour = bezier_closed_from_four_points(control_points, samples_per_segment=200, tension=1.0)
        levelset = signed_distance_from_contour(contour, x, y)

        tag = f"shape_{i:02d}"
        h5_path = out_dir / f"{tag}_levelset.h5"
        png_path = out_dir / f"{tag}.png"
        json_path = out_dir / f"{tag}.json"

        write_levelset_h5(levelset, h5_path)
        render_shape_preview(contour, control_points, png_path)

        shape_data = {
            "shape_id": tag,
            "control_points": control_points.tolist(),
            "levelset_h5": str(h5_path.resolve()),
            "preview_png": str(png_path.resolve()),
        }
        json_path.write_text(json.dumps(shape_data, indent=2))

        flat_points = control_points.reshape(-1)
        rows.append([tag, str(h5_path.resolve()), str(png_path.resolve())] + flat_points.tolist())

    csv_path = out_dir / "manifest.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "shape_id",
                "levelset_h5",
                "preview_png",
                "p0_x",
                "p0_y",
                "p1_x",
                "p1_y",
                "p2_x",
                "p2_y",
                "p3_x",
                "p3_y",
            ]
        )
        writer.writerows(rows)

    print(f"Generated 10 shapes in: {out_dir.resolve()}")
    print(f"Manifest: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
