"""CSV export utilities."""
from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.models import DataPoint


class CSVExporter:
    """Export data to CSV format."""

    @staticmethod
    def export_image(image: np.ndarray, filepath: str) -> None:
        """Export image as CSV."""
        np.savetxt(filepath, image, delimiter=",", fmt="%.6f")

    @staticmethod
    def export_data_points(data_points: list[DataPoint], filepath: str) -> None:
        """Export data points as CSV."""
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                "timestamp", "x_pos", "y_pos", "z_pos", "r_pos", "current"
            ])

            # Write data
            for dp in data_points:
                writer.writerow([
                    dp.timestamp,
                    dp.x_pos,
                    dp.y_pos,
                    dp.z_pos if dp.z_pos is not None else "",
                    dp.r_pos if dp.r_pos is not None else "",
                    dp.current
                ])
