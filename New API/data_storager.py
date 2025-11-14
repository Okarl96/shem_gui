"""
DataStorager - Persistent Data Storage

Modular system for saving scan results in multiple formats:
- HDF5: Complete data (metadata + raw + reconstructed)
- CSV: Summary data (metadata + reconstructed)
- PNG: Visualization (reconstructed images)

Manages automatic scan ID generation with prefixes:
- 2D scans: S0000, S0001, ...
- 1D scans: L0000, L0001, ...
- Z scans: Z0000, Z0001, ...

All formats use consistent metadata structure.
"""

import h5py
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re


# ============================================================================
# Data Formatting Utilities
# ============================================================================

class MetadataFormatter:
    """
    Standardizes metadata structure across all export formats.

    Ensures consistent key names, types, and organization.
    """

    @staticmethod
    def flatten_metadata(metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested metadata structure for storage.

        Converts nested dicts into flat key-value pairs using dot notation.
        Example: {'z_compensation': {'x_ratio': 1.0}} -> {'z_compensation.x_ratio': 1.0}

        Args:
            metadata_dict: Nested metadata dictionary

        Returns:
            Flattened dictionary with string-serialized values
        """
        flat = {}

        for key, value in metadata_dict.items():
            if isinstance(value, dict):
                # Flatten nested dicts
                for sub_key, sub_value in value.items():
                    flat_key = f"{key}.{sub_key}"
                    flat[flat_key] = MetadataFormatter._serialize_value(sub_value)
            elif isinstance(value, (list, tuple)):
                # Convert sequences to JSON strings
                flat[key] = json.dumps(value)
            else:
                flat[key] = MetadataFormatter._serialize_value(value)

        return flat

    @staticmethod
    def _serialize_value(value: Any) -> str:
        """Convert value to string for storage"""
        if value is None:
            return 'None'
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            return str(value)

    @staticmethod
    def prepare_for_hdf5(metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metadata for HDF5 attributes.

        HDF5 attributes have restrictions - this ensures compatibility.
        """
        prepared = {}

        for key, value in metadata_dict.items():
            if isinstance(value, dict):
                # Store nested dicts as JSON strings
                prepared[key] = json.dumps(value)
            elif isinstance(value, (list, tuple)):
                # Store sequences as JSON strings
                prepared[key] = json.dumps(value)
            elif value is None:
                prepared[key] = 'None'
            else:
                prepared[key] = value

        return prepared

    @staticmethod
    def prepare_for_csv(metadata_dict: Dict[str, Any]) -> Dict[str, str]:
        """
        Prepare metadata for CSV format.

        Flattens all nested structures into simple key-value pairs.
        """
        return MetadataFormatter.flatten_metadata(metadata_dict)


# ============================================================================
# Data Packing Utilities
# ============================================================================

class DataPacker:
    """
    Organize raw and reconstructed data into consistent structures.

    Handles different scan types (2D, 1D, Z-scan) uniformly.
    """

    @staticmethod
    def pack_raw_data(datapoints: List) -> Dict[str, np.ndarray]:
        """
        Pack raw datapoints into arrays.

        Args:
            datapoints: List of DataPoint objects

        Returns:
            Dictionary with arrays for each field
        """
        if not datapoints:
            return {}

        # Extract fields from first datapoint to determine structure
        first_dp = datapoints[0]

        packed = {
            't_start': np.array([dp.t_start if dp.t_start is not None else np.nan for dp in datapoints]),
            't_end': np.array([dp.t_end if dp.t_end is not None else np.nan for dp in datapoints]),
            'avg_signal': np.array([dp.avg_signal if dp.avg_signal is not None else np.nan for dp in datapoints]),
            'std_signal': np.array([dp.std_signal if dp.std_signal is not None else np.nan for dp in datapoints]),
            'n_samples': np.array([dp.n_samples for dp in datapoints]),
        }

        # Add position data if available
        if hasattr(first_dp, 'position') and first_dp.position is not None:
            # Position is dict with keys like 'X_nm', 'Y_nm', etc.
            if isinstance(first_dp.position, dict):
                for axis_key in ['X_nm', 'Y_nm', 'Z_nm', 'R_udeg']:
                    if axis_key in first_dp.position:
                        packed[f'position_{axis_key}'] = np.array([
                            dp.position.get(axis_key, np.nan) if dp.position else np.nan
                            for dp in datapoints
                        ])

        return packed

    @staticmethod
    def pack_reconstructed_2d(image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Pack 2D reconstructed image.

        Args:
            image: 2D numpy array

        Returns:
            Dictionary with image data
        """
        return {
            'image': image,
            'shape': np.array(image.shape),
        }

    @staticmethod
    def pack_reconstructed_1d(profile: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Pack 1D reconstructed profile.

        Args:
            profile: 1D numpy array

        Returns:
            Dictionary with profile data
        """
        return {
            'profile': profile,
            'length': len(profile),
        }


# ============================================================================
# Format-Specific Writers
# ============================================================================

class HDF5Writer:
    """
    Write scan data to HDF5 format.

    Structure:
        /metadata (attributes)
        /raw_data/
            timestamps
            avg_signal
            std_signal
            ...
        /reconstructed/
            image (2D) or profile (1D)
            statistics (attributes)
    """

    @staticmethod
    def write(filepath: Path, scan_result) -> bool:
        """
        Write scan result to HDF5 file.

        Args:
            filepath: Output file path
            scan_result: ScanResult object

        Returns:
            True if successful
        """
        try:
            with h5py.File(filepath, 'w') as f:
                # Write metadata as root attributes
                metadata_dict = scan_result.metadata.to_dict()
                prepared_meta = MetadataFormatter.prepare_for_hdf5(metadata_dict)

                for key, value in prepared_meta.items():
                    f.attrs[key] = value

                # Write raw data
                raw_grp = f.create_group('raw_data')
                raw_data = DataPacker.pack_raw_data(scan_result.raw_datapoints)

                for key, array in raw_data.items():
                    raw_grp.create_dataset(key, data=array, compression='gzip')

                # Write reconstructed data
                recon_grp = f.create_group('reconstructed')

                if scan_result.reconstructed_data is not None:
                    if scan_result.metadata.scan_type == '2D':
                        packed = DataPacker.pack_reconstructed_2d(
                            scan_result.reconstructed_data
                        )
                        recon_grp.create_dataset(
                            'image',
                            data=packed['image'],
                            compression='gzip'
                        )
                    else:  # 1D or Z
                        packed = DataPacker.pack_reconstructed_1d(
                            scan_result.reconstructed_data
                        )
                        recon_grp.create_dataset(
                            'profile',
                            data=packed['profile'],
                            compression='gzip'
                        )

                # Write statistics as attributes
                if scan_result.statistics:
                    for key, value in scan_result.statistics.items():
                        recon_grp.attrs[key] = value

            return True

        except Exception as e:
            print(f"HDF5 write failed: {e}")
            return False


class CSVWriter:
    """
    Write scan metadata and reconstructed data to CSV format.

    Structure:
        - Metadata rows (# prefix)
        - Blank line
        - Data header row
        - Data rows (reconstructed values)
    """

    @staticmethod
    def write(filepath: Path, scan_result) -> bool:
        """
        Write scan result to CSV.

        Args:
            filepath: Output file path
            scan_result: ScanResult object

        Returns:
            True if successful
        """
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)

                # Write metadata section (commented lines)
                metadata_dict = scan_result.metadata.to_dict()
                flat_meta = MetadataFormatter.prepare_for_csv(metadata_dict)

                writer.writerow(['# METADATA'])
                for key, value in flat_meta.items():
                    writer.writerow([f'# {key}', value])

                # Statistics
                if scan_result.statistics:
                    writer.writerow(['# STATISTICS'])
                    for key, value in scan_result.statistics.items():
                        writer.writerow([f'# {key}', value])

                writer.writerow([])  # Blank line

                # Write reconstructed data
                if scan_result.reconstructed_data is not None:
                    if scan_result.metadata.scan_type == '2D':
                        CSVWriter._write_2d_data(writer, scan_result)
                    else:
                        CSVWriter._write_1d_data(writer, scan_result)

            return True

        except Exception as e:
            print(f"CSV write failed: {e}")
            return False

    @staticmethod
    def _write_2d_data(writer, scan_result):
        """Write 2D image data to CSV"""
        image = scan_result.reconstructed_data

        # Header
        writer.writerow(['# 2D IMAGE DATA'])
        writer.writerow(['# Row', 'Column', 'Signal (nA)'])

        # Data rows
        rows, cols = image.shape
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(image[i, j]):
                    writer.writerow([i, j, image[i, j]])

    @staticmethod
    def _write_1d_data(writer, scan_result):
        """Write 1D profile data to CSV"""
        profile = scan_result.reconstructed_data

        # Header
        writer.writerow(['# 1D PROFILE DATA'])
        writer.writerow(['# Index', 'Signal (nA)'])

        # Data rows - use numpy's isnan element-wise check properly
        for i, value in enumerate(profile):
            # Check if scalar is nan - this works for single values
            if np.isfinite(value):  # More robust than 'not np.isnan'
                writer.writerow([i, value])


class PNGWriter:
    """
    Create visualization images from scan data.

    Generates publication-quality plots with metadata annotations.
    """

    @staticmethod
    def write(filepath: Path, scan_result) -> bool:
        """
        Create PNG visualization.

        Args:
            filepath: Output file path
            scan_result: ScanResult object

        Returns:
            True if successful
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            if scan_result.metadata.scan_type == '2D':
                PNGWriter._plot_2d(ax, scan_result)
            else:
                PNGWriter._plot_1d(ax, scan_result)

            # Add title with scan info
            title = f"Scan: {scan_result.metadata.scan_id}"
            if scan_result.status != 'completed':
                title += f" ({scan_result.status})"
            ax.set_title(title, fontsize=14, fontweight='bold')

            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)

            return True

        except Exception as e:
            print(f"PNG write failed: {e}")
            return False

    @staticmethod
    def _plot_2d(ax, scan_result):
        """Plot 2D image with colorbar using real positions"""
        image = scan_result.reconstructed_data

        # Ensure image is float type for proper plotting
        if image is not None:
            image = np.asarray(image, dtype=np.float64)
        else:
            print("Warning: No reconstructed data available")
            return

        # Get real position ranges from metadata
        x_range = scan_result.metadata.x_range_nm
        y_range = scan_result.metadata.y_range_nm

        # Set extent for real position axes
        # extent = [left, right, bottom, top]
        if x_range and y_range:
            extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
        else:
            extent = None

        # Create image plot with real positions
        im = ax.imshow(image, cmap='viridis', aspect='auto', origin='lower', extent=extent)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Signal (nA)', rotation=270, labelpad=20)

        # Labels with units
        ax.set_xlabel('X position (nm)')
        ax.set_ylabel('Y position (nm)')

        # Add statistics text
        stats = scan_result.statistics
        if stats:
            text = f"Mean: {stats.get('signal_mean', 0):.2f} nA\n"
            text += f"Std: {stats.get('signal_std', 0):.2f} nA\n"
            text += f"Range: [{stats.get('signal_min', 0):.2f}, "
            text += f"{stats.get('signal_max', 0):.2f}] nA"
            ax.text(0.02, 0.98, text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9)

    @staticmethod
    def _plot_1d(ax, scan_result):
        """Plot 1D profile"""
        profile = scan_result.reconstructed_data

        x = np.arange(len(profile))
        mask = ~np.isnan(profile)

        ax.plot(x[mask], profile[mask], 'o-', linewidth=2, markersize=4)
        ax.set_xlabel('Point index')
        ax.set_ylabel('Signal (nA)')
        ax.grid(True, alpha=0.3)

        # Add statistics
        stats = scan_result.statistics
        if stats:
            text = f"Mean: {stats.get('signal_mean', 0):.2f} nA\n"
            text += f"Std: {stats.get('signal_std', 0):.2f} nA"
            ax.text(0.02, 0.98, text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9)


# ============================================================================
# Main DataStorager Class
# ============================================================================

class DataStorager:
    """
    Unified interface for saving scan data in multiple formats.

    Handles automatic scan ID generation with prefixes:
    - 2D scans: S0000, S0001, S0002, ...
    - 1D scans: L0000, L0001, L0002, ...
    - Z scans: Z0000, Z0001, Z0002, ...

    Usage:
        storager = DataStorager(base_path='./data')

        # Save in all formats (ID auto-generated)
        storager.save_scan(
            scan_result,
            formats=['hdf5', 'csv', 'png']
        )

        # Or save individually
        storager.save_hdf5(scan_result, 'scan.h5')
        storager.save_csv(scan_result, 'scan.csv')
        storager.save_png(scan_result, 'scan.png')
    """

    def __init__(self, base_path: str = './scan_data'):
        """
        Initialize data storager.

        Args:
            base_path: Base directory for all saved files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        print(f"DataStorager initialized: {self.base_path}")

    def get_next_scan_id(self, scan_type: str) -> str:
        """
        Generate next scan ID based on scan type and existing files.

        Args:
            scan_type: '2D', '1D_line', or 'Z_scan'

        Returns:
            Next scan ID with appropriate prefix and index
            - 2D: S0000, S0001, ...
            - 1D_line: L0000, L0001, ...
            - Z_scan: Z0000, Z0001, ...
        """
        # Determine prefix based on scan type
        prefix_map = {
            '2D': 'S',
            '1D_line': 'L',
            'Z_scan': 'Z'
        }

        prefix = prefix_map.get(scan_type, 'S')

        # Find all files matching the prefix pattern
        pattern = re.compile(rf'^{prefix}(\d+)')
        max_idx = -1

        if self.base_path.exists():
            for file in self.base_path.iterdir():
                match = pattern.match(file.name)
                if match:
                    idx = int(match.group(1))
                    max_idx = max(max_idx, idx)

        # Return next index
        next_idx = max_idx + 1
        return f"{prefix}{next_idx:04d}"

    def save_scan(self,
                  scan_result,
                  formats: List[str] = ['hdf5', 'csv', 'png'],
                  filename_prefix: Optional[str] = None) -> Dict[str, Path]:
        """
        Save scan result in multiple formats.

        Automatically assigns scan ID based on scan type if not provided.

        Args:
            scan_result: ScanResult object
            formats: List of format strings: 'hdf5', 'csv', 'png'
            filename_prefix: Optional custom filename (without extension)
                           If None, auto-generates ID with prefix (S/L/Z)

        Returns:
            Dictionary mapping format to filepath
        """
        # Auto-generate scan ID if not provided
        if filename_prefix is None:
            # Generate ID based on scan type
            scan_id = self.get_next_scan_id(scan_result.metadata.scan_type)
            # Update metadata with final scan ID
            scan_result.metadata.scan_id = scan_id
            filename_prefix = scan_id

        saved_files = {}

        for fmt in formats:
            if fmt == 'hdf5':
                filepath = self.save_hdf5(scan_result, f"{filename_prefix}.h5")
                if filepath:
                    saved_files['hdf5'] = filepath

            elif fmt == 'csv':
                filepath = self.save_csv(scan_result, f"{filename_prefix}.csv")
                if filepath:
                    saved_files['csv'] = filepath

            elif fmt == 'png':
                filepath = self.save_png(scan_result, f"{filename_prefix}.png")
                if filepath:
                    saved_files['png'] = filepath

            else:
                print(f"Unknown format: {fmt}")

        return saved_files

    def save_hdf5(self, scan_result, filename: str) -> Optional[Path]:
        """Save to HDF5 format"""
        filepath = self.base_path / filename

        if HDF5Writer.write(filepath, scan_result):
            print(f"Saved HDF5: {filepath}")
            return filepath
        return None

    def save_csv(self, scan_result, filename: str) -> Optional[Path]:
        """Save to CSV format"""
        filepath = self.base_path / filename

        if CSVWriter.write(filepath, scan_result):
            print(f"Saved CSV: {filepath}")
            return filepath
        return None

    def save_png(self, scan_result, filename: str) -> Optional[Path]:
        """Save to PNG format"""
        filepath = self.base_path / filename

        if PNGWriter.write(filepath, scan_result):
            print(f"Saved PNG: {filepath}")
            return filepath
        return None