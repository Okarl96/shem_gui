"""Image reconstruction from scan data."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtCore import QMutex, QMutexLocker, QObject, pyqtSignal, pyqtSlot

if TYPE_CHECKING:
    from core.models import DataPoint, ScanParameters


class ImageReconstructor(QObject):
    """Real-time image reconstruction."""

    # Signals
    image_updated = pyqtSignal(object)  # numpy array
    pixel_updated = pyqtSignal(int, int, float)  # x, y, value

    def __init__(self) -> None:
        super().__init__()
        self.scan_params: ScanParameters | None = None
        self.image: np.ndarray | None = None
        self.pixel_counts: np.ndarray | None = None
        self.mutex = QMutex()

        # Reconstruction settings
        self.min_samples_per_pixel = 1
        self.interpolation_method = "nearest"

    def initialize_scan(self, scan_params: ScanParameters) -> None:
        """Initialize for new scan."""
        with QMutexLocker(self.mutex):
            self.scan_params = scan_params
            # Initialize with zeros
            self.image = np.zeros((scan_params.y_pixels, scan_params.x_pixels), dtype=np.float64)
            self.pixel_counts = np.zeros((scan_params.y_pixels, scan_params.x_pixels))

            # Emit initial image
            self.image_updated.emit(self.image.copy())

    @pyqtSlot(object)
    def add_data_point(self, data_point: DataPoint) -> None:
        """Add data point to image reconstruction."""
        if not self.scan_params or data_point.x_pos is None or data_point.y_pos is None:
            return

        try:
            with QMutexLocker(self.mutex):
                # Convert position to pixel coordinates
                x_pixel = self._position_to_pixel_x(data_point.x_pos)
                y_pixel = self._position_to_pixel_y(data_point.y_pos)

                # Check bounds
                if (0 <= x_pixel < self.scan_params.x_pixels and
                    0 <= y_pixel < self.scan_params.y_pixels):

                    # Update pixel value (running average)
                    current_count = self.pixel_counts[y_pixel, x_pixel]
                    if current_count == 0:
                        # First measurement
                        self.image[y_pixel, x_pixel] = data_point.current
                    else:
                        # Running average
                        old_value = self.image[y_pixel, x_pixel]
                        new_value = (old_value * current_count + data_point.current) / (current_count + 1)
                        self.image[y_pixel, x_pixel] = new_value

                    self.pixel_counts[y_pixel, x_pixel] += 1

                    # Emit updates
                    self.pixel_updated.emit(x_pixel, y_pixel, self.image[y_pixel, x_pixel])

        except Exception as e:
            print(f"Error adding data point to image: {e}")

    @pyqtSlot()
    def emit_full_image(self) -> None:
        """Emit complete current image."""
        with QMutexLocker(self.mutex):
            if self.image is not None:
                self.image_updated.emit(self.image.copy())

    def get_completion_percentage(self) -> float:
        """Get scan completion percentage."""
        with QMutexLocker(self.mutex):
            if self.pixel_counts is None:
                return 0.0

            filled_pixels = np.sum(self.pixel_counts > 0)
            total_pixels = self.pixel_counts.size
            return (filled_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    def get_current_image(self) -> np.ndarray | None:
        """Get current image."""
        with QMutexLocker(self.mutex):
            return self.image.copy() if self.image is not None else None

    def _position_to_pixel_x(self, position: float) -> int:
        """Convert X position to pixel coordinate."""
        if not self.scan_params:
            return 0
        x_range = self.scan_params.x_end - self.scan_params.x_start
        if x_range == 0:
            return 0
        normalized = (position - self.scan_params.x_start) / x_range
        return int(normalized * (self.scan_params.x_pixels - 1))

    def _position_to_pixel_y(self, position: float) -> int:
        """Convert Y position to pixel coordinate."""
        if not self.scan_params:
            return 0
        y_range = self.scan_params.y_end - self.scan_params.y_start
        if y_range == 0:
            return 0
        normalized = (position - self.scan_params.y_start) / y_range
        return int(normalized * (self.scan_params.y_pixels - 1))

    def reset(self) -> None:
        """Reset the reconstructor."""
        with QMutexLocker(self.mutex):
            self.scan_params = None
            self.image = None
            self.pixel_counts = None
