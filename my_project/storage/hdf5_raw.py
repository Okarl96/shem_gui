from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from core.models import DataPoint, ScanParameters


class HDF5RawStorage:
    """HDF5 storage for raw scan data with proper resource management."""

    def __init__(self, base_path: str) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.current_file: h5py.File | None = None
        self.current_file_path: Path | None = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure file is closed."""
        self.close()
        return False

    def create_file(self, scan_id: int, scan_params: ScanParameters) -> str:
        """Create new HDF5 file for scan."""
        filename = f"scan_{scan_id:06d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        filepath = self.base_path / filename

        # Close current file if open
        if self.current_file:
            self.current_file.close()

        # Create new file
        self.current_file = h5py.File(filepath, "w")
        self.current_file_path = filepath

        # Store scan parameters as attributes
        self.current_file.attrs["scan_id"] = scan_id
        self.current_file.attrs["x_start"] = scan_params.x_start
        self.current_file.attrs["x_end"] = scan_params.x_end
        self.current_file.attrs["y_start"] = scan_params.y_start
        self.current_file.attrs["y_end"] = scan_params.y_end
        self.current_file.attrs["x_pixels"] = scan_params.x_pixels
        self.current_file.attrs["y_pixels"] = scan_params.y_pixels
        self.current_file.attrs["mode"] = scan_params.mode.value
        self.current_file.attrs["dwell_time"] = scan_params.dwell_time
        self.current_file.attrs["scan_speed"] = scan_params.scan_speed

        # Create datasets
        grp = self.current_file.create_group("raw_data")

        # Pre-allocate datasets with compression
        grp.create_dataset("timestamps", (0,), maxshape=(None,),
                          dtype="f8", chunks=True,
                          compression="gzip", compression_opts=6)
        grp.create_dataset("x_positions", (0,), maxshape=(None,),
                          dtype="f8", chunks=True,
                          compression="gzip", compression_opts=6)
        grp.create_dataset("y_positions", (0,), maxshape=(None,),
                          dtype="f8", chunks=True,
                          compression="gzip", compression_opts=6)
        grp.create_dataset("z_positions", (0,), maxshape=(None,),
                          dtype="f8", chunks=True,
                          compression="gzip", compression_opts=6)
        grp.create_dataset("r_positions", (0,), maxshape=(None,),
                          dtype="f8", chunks=True,
                          compression="gzip", compression_opts=6)
        grp.create_dataset("currents", (0,), maxshape=(None,),
                          dtype="f8", chunks=True,
                          compression="gzip", compression_opts=6)

        return str(filepath)

    def save_batch(self, data_points: list[DataPoint]) -> None:
        """Save batch of data points with error handling."""
        if not self.current_file or not data_points:
            return

        try:
            grp = self.current_file["raw_data"]

            # Extract data
            timestamps = [dp.timestamp for dp in data_points]
            x_pos = [dp.x_pos for dp in data_points]
            y_pos = [dp.y_pos for dp in data_points]
            z_pos = [dp.z_pos if dp.z_pos is not None else np.nan for dp in data_points]
            r_pos = [dp.r_pos if dp.r_pos is not None else np.nan for dp in data_points]
            currents = [dp.current for dp in data_points]

            # Append to datasets
            for name, data in [
                ("timestamps", timestamps),
                ("x_positions", x_pos),
                ("y_positions", y_pos),
                ("z_positions", z_pos),
                ("r_positions", r_pos),
                ("currents", currents)
            ]:
                dataset = grp[name]
                old_size = dataset.shape[0]
                new_size = old_size + len(data)
                dataset.resize(new_size, axis=0)
                dataset[old_size:new_size] = data

            # Flush to disk
            self.current_file.flush()

        except Exception as e:
            msg = f"Failed to save batch to HDF5: {e}"
            raise OSError(msg)

    def save_image(self, image: np.ndarray, name: str = "reconstructed_image") -> None:
        """Save reconstructed image."""
        if not self.current_file:
            return

        if "images" not in self.current_file:
            self.current_file.create_group("images")

        grp = self.current_file["images"]

        if name in grp:
            del grp[name]

        grp.create_dataset(name, data=image,
                          compression="gzip", compression_opts=6)
        self.current_file.flush()

    def close(self) -> None:
        """Close current file."""
        if self.current_file:
            try:
                self.current_file.close()
            except:
                pass  # Ignore errors during close
            finally:
                self.current_file = None
                self.current_file_path = None
