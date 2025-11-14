"""
ScanController - Orchestrate Scanning Workflow

FEATURES:
1. Separate methods for rectangular and vertices-based 2D scans
2. Optimized 2D scanning (Z/R set once at start)
3. Data saving on cancel/interrupt
4. Scan ID managed by DataStorager
5. ASCII-only output

Connects pattern generation, stage movement, and data acquisition.
Executes scans and provides callbacks for progress/results.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any, Tuple, Union
from datetime import datetime
import time
import numpy as np


@dataclass
class ScanMetadata:
    """
    Standardized metadata structure for all scan types.
    This ensures consistent metadata across HDF5, CSV, and PNG exports.

    Each scan type can add its own specific fields via the extra_fields dict.
    """
    # Scan identification
    scan_id: str
    scan_type: str  # '2D', '1D_line', 'Z_scan', 'Multi_Z'
    timestamp: str  # ISO format

    # Spatial parameters
    x_range_nm: Optional[tuple] = None  # (min, max)
    y_range_nm: Optional[tuple] = None  # (min, max)
    z_range_nm: Optional[tuple] = None  # (min, max)
    z_fixed_nm: Optional[float] = None   # Fixed Z for 2D scans
    r_fixed_udeg: Optional[float] = None # Fixed R for 2D scans

    # Discretization
    num_points_x: Optional[int] = None
    num_points_y: Optional[int] = None
    num_points_z: Optional[int] = None
    num_points_total: int = 0

    # Pattern
    pattern_type: str = 'raster'  # 'raster', 'snake', 'single', 'bidirectional', 'z_outer'
    fast_axis: Optional[str] = None
    vertices_nm: Optional[List[Tuple[float, float]]] = None  # For custom area

    # Acquisition parameters
    settle_tolerance_nm: Dict[str, float] = field(default_factory=dict)
    dwell_time_s: float = 1.0
    detector_lag_s: float = 0.0
    sample_interval_s: float = 0.025

    # Compensation
    z_compensation: Dict[str, float] = field(default_factory=dict)

    # Timing
    scan_start_time: Optional[str] = None
    scan_end_time: Optional[str] = None
    scan_duration_s: Optional[float] = None

    # Hardware info
    mqtt_broker: Optional[str] = None
    stage_controller: str = "ECC100"
    detector: str = "Picoammeter"

    # Additional notes
    notes: str = ""

    # Scan-type-specific additional fields
    # This allows each scan type to store its own metadata without modifying the base class
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        base_dict = {
            'scan_id': self.scan_id,
            'scan_type': self.scan_type,
            'timestamp': self.timestamp,
            'x_range_nm': self.x_range_nm,
            'y_range_nm': self.y_range_nm,
            'z_range_nm': self.z_range_nm,
            'z_fixed_nm': self.z_fixed_nm,
            'r_fixed_udeg': self.r_fixed_udeg,
            'num_points_x': self.num_points_x,
            'num_points_y': self.num_points_y,
            'num_points_z': self.num_points_z,
            'num_points_total': self.num_points_total,
            'pattern_type': self.pattern_type,
            'fast_axis': self.fast_axis,
            'vertices_nm': self.vertices_nm,
            'settle_tolerance_nm': self.settle_tolerance_nm,
            'dwell_time_s': self.dwell_time_s,
            'detector_lag_s': self.detector_lag_s,
            'sample_interval_s': self.sample_interval_s,
            'z_compensation': self.z_compensation,
            'scan_start_time': self.scan_start_time,
            'scan_end_time': self.scan_end_time,
            'scan_duration_s': self.scan_duration_s,
            'mqtt_broker': self.mqtt_broker,
            'stage_controller': self.stage_controller,
            'detector': self.detector,
            'notes': self.notes,
        }

        # Merge extra fields for scan-type-specific metadata
        base_dict.update(self.extra_fields)

        return base_dict


@dataclass
class ScanResult:
    """Container for scan results"""
    status: str  # 'completed', 'cancelled', 'error'
    metadata: ScanMetadata
    raw_datapoints: List[Any]
    reconstructed_data: Optional[np.ndarray] = None
    statistics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class ScanController:
    """
    Orchestrates scanning workflow.

    Workflow:
    1. Configure scan (2D rectangular, 2D vertices, 1D, or Z)
    2. Execute scan loop with callbacks
    3. Return ScanResult with data and metadata

    Note: Scan ID is assigned by DataStorager during save, not here.
    """

    def __init__(self, commander, reader):
        """
        Args:
            commander: CommandSender instance
            reader: PositionSignalReader instance
        """
        self.commander = commander
        self.reader = reader

        # Scan configuration
        self.pattern = None
        self.metadata = None

        # Acquisition parameters
        self.settle_params = {
            'check_interval': 0.05,
            'timeout': 10.0,
            'required_consecutive': 3,
        }
        self.dwell_params = {
            'dwell_time': 1.0,
            'detector_lag': 0.0,
            'sample_interval': 0.025,
        }

        # Control flags
        self._paused = False
        self._cancelled = False

        # Fixed axis storage (for 2D scans)
        self._z_fixed = None
        self._r_fixed = None
        self._fixed_axes_set = False

    def configure_2d_scan(self,
                         x_range: Tuple[float, float],
                         y_range: Tuple[float, float],
                         num_points: Tuple[int, int],
                         pattern: str = 'raster',
                         fast_axis: str = 'X',
                         z_fixed_nm: Optional[float] = None,
                         r_fixed_udeg: Optional[float] = None,
                         settle_tolerance: Optional[Dict[str, float]] = None,
                         dwell_time: float = 1.0,
                         detector_lag: float = 0.0,
                         **kwargs):
        """
        Configure 2D rectangular area scan.

        Args:
            x_range: (min, max) in nm
            y_range: (min, max) in nm
            num_points: (nx, ny) grid size
            pattern: 'raster' or 'snake'
            fast_axis: 'X' or 'Y'
            z_fixed_nm: Fixed Z height (set once at start)
            r_fixed_udeg: Fixed rotation (set once at start)
            settle_tolerance: dict with axis tolerances
            dwell_time: seconds per point
            detector_lag: seconds
        """
        from core.scan_patterns import generate_2d_rectangular_grid
        from core.scan_patterns import apply_raster_pattern_2d, apply_snake_pattern_2d

        # Generate rectangular grid
        grid = generate_2d_rectangular_grid(
            x_min_nm=x_range[0],
            x_max_nm=x_range[1],
            x_pixels=num_points[0],
            y_min_nm=y_range[0],
            y_max_nm=y_range[1],
            y_pixels=num_points[1]
        )

        # Add row/col information for proper reconstruction
        num_x, num_y = num_points
        for point in grid:
            point['row'] = point['idx'] // num_x
            point['col'] = point['idx'] % num_x

        # Apply movement pattern (snake or raster)
        if pattern == 'snake':
            self.pattern = apply_snake_pattern_2d(grid, fast_axis=fast_axis)
        elif pattern == 'raster':
            self.pattern = apply_raster_pattern_2d(grid, fast_axis=fast_axis)
        else:
            raise ValueError(f"Pattern must be 'raster' or 'snake', got '{pattern}'")

        # Store fixed axis values for optimization
        self._z_fixed = z_fixed_nm
        self._r_fixed = r_fixed_udeg
        self._fixed_axes_set = False  # Will be set on first move

        # Setup metadata (scan_id will be assigned by DataStorager)
        self.metadata = ScanMetadata(
            scan_id='2D_scanning',  # Temporary ID for display
            scan_type='2D',
            timestamp=datetime.now().isoformat(),
            x_range_nm=x_range,
            y_range_nm=y_range,
            z_fixed_nm=z_fixed_nm,
            r_fixed_udeg=r_fixed_udeg,
            num_points_x=num_points[0],
            num_points_y=num_points[1],
            num_points_total=len(self.pattern),
            pattern_type=pattern,
            fast_axis=fast_axis,
            vertices_nm=None,  # Not used for rectangular
            settle_tolerance_nm=settle_tolerance or {'X': 5, 'Y': 5, 'Z': 5},
            dwell_time_s=dwell_time,
            detector_lag_s=detector_lag,
            **kwargs
        )

        # Update acquisition params
        self.dwell_params['dwell_time'] = dwell_time
        self.dwell_params['detector_lag'] = detector_lag

    def configure_2d_vertices_scan(self,
                                   vertices_nm: List[Tuple[float, float]],
                                   num_points: Tuple[int, int],
                                   pattern: str = 'raster',
                                   fast_axis: str = 'X',
                                   z_fixed_nm: Optional[float] = None,
                                   r_fixed_udeg: Optional[float] = None,
                                   settle_tolerance: Optional[Dict[str, float]] = None,
                                   dwell_time: float = 1.0,
                                   detector_lag: float = 0.0,
                                   **kwargs):
        """
        Configure 2D scan with custom vertices defining the area.

        Args:
            vertices_nm: List of (x, y) vertices defining polygon boundary
            num_points: (nx, ny) grid size
            pattern: 'raster' or 'snake' (movement order within area)
            fast_axis: 'X' or 'Y'
            z_fixed_nm: Fixed Z height (set once at start)
            r_fixed_udeg: Fixed rotation (set once at start)
            settle_tolerance: dict with axis tolerances
            dwell_time: seconds per point
            detector_lag: seconds
        """
        from core.scan_patterns import generate_2d_custom_grid
        from core.scan_patterns import apply_raster_pattern_2d, apply_snake_pattern_2d

        # Generate grid within custom vertices
        grid = generate_2d_custom_grid(
            vertices_nm=vertices_nm,
            x_pixels=num_points[0],
            y_pixels=num_points[1]
        )
        # Note: grid points now include 'row' and 'col' for bounding box position

        # Apply movement pattern (snake or raster)
        # These functions preserve the row/col fields
        if pattern == 'snake':
            self.pattern = apply_snake_pattern_2d(grid, fast_axis=fast_axis)
        elif pattern == 'raster':
            self.pattern = apply_raster_pattern_2d(grid, fast_axis=fast_axis)
        else:
            raise ValueError(f"Pattern must be 'raster' or 'snake', got '{pattern}'")

        # Store fixed axis values for optimization
        self._z_fixed = z_fixed_nm
        self._r_fixed = r_fixed_udeg
        self._fixed_axes_set = False

        # Calculate bounding box for metadata
        xs = [v[0] for v in vertices_nm]
        ys = [v[1] for v in vertices_nm]
        x_range = (min(xs), max(xs))
        y_range = (min(ys), max(ys))

        # Setup metadata (scan_id will be assigned by DataStorager)
        self.metadata = ScanMetadata(
            scan_id='2D_scanning',  # Temporary ID
            scan_type='2D',
            timestamp=datetime.now().isoformat(),
            x_range_nm=x_range,
            y_range_nm=y_range,
            z_fixed_nm=z_fixed_nm,
            r_fixed_udeg=r_fixed_udeg,
            num_points_x=num_points[0],
            num_points_y=num_points[1],
            num_points_total=len(self.pattern),
            pattern_type=pattern,
            fast_axis=fast_axis,
            vertices_nm=vertices_nm,  # Store vertices
            settle_tolerance_nm=settle_tolerance or {'X': 5, 'Y': 5, 'Z': 5},
            dwell_time_s=dwell_time,
            detector_lag_s=detector_lag,
            **kwargs
        )

        # Update acquisition params
        self.dwell_params['dwell_time'] = dwell_time
        self.dwell_params['detector_lag'] = detector_lag

    def configure_1d_line_scan(self,
                              fixed_axis: str,
                              fixed_value: float,
                              scan_axis: str,
                              start: float,
                              end: float,
                              num_points: int,
                              bidirectional: bool = False,
                              settle_tolerance: Optional[Dict[str, float]] = None,
                              dwell_time: float = 1.0,
                              detector_lag: float = 0.0,
                              **kwargs):
        """
        Configure 1D line scan.

        Scan ID will be assigned by DataStorager with 'L' prefix.
        """
        from core.scan_patterns import generate_1d_line
        from core.scan_patterns import apply_single_direction_1d, apply_bidirectional_1d

        # Generate line
        line = generate_1d_line(
            fixed_axis=fixed_axis,
            fixed_value_nm=fixed_value,
            scan_axis=scan_axis,
            start_nm=start,
            end_nm=end,
            num_points=num_points
        )

        # Apply pattern
        if bidirectional:
            self.pattern = apply_bidirectional_1d(line)
        else:
            self.pattern = apply_single_direction_1d(line)

        # Store fixed axis info
        self._fixed_axis = fixed_axis
        self._fixed_value = fixed_value

        # Determine ranges
        if scan_axis == 'X':
            x_range = (start, end)
            y_range = (fixed_value, fixed_value)
        else:
            x_range = (fixed_value, fixed_value)
            y_range = (start, end)

        # Setup metadata (scan_id will be assigned by DataStorager)
        self.metadata = ScanMetadata(
            scan_id='1D_scanning',  # Temporary ID
            scan_type='1D_line',
            timestamp=datetime.now().isoformat(),
            x_range_nm=x_range,
            y_range_nm=y_range,
            num_points_total=len(self.pattern),
            pattern_type='bidirectional' if bidirectional else 'single',
            settle_tolerance_nm=settle_tolerance or {scan_axis: 5},
            dwell_time_s=dwell_time,
            detector_lag_s=detector_lag,
            **kwargs
        )

        # Update acquisition params
        self.dwell_params['dwell_time'] = dwell_time
        self.dwell_params['detector_lag'] = detector_lag

    def configure_z_scan(self,
                        x_nominal: float,
                        y_nominal: float,
                        z_start: float,
                        z_end: float,
                        num_points: int,
                        z_compensation: Optional[Dict[str, float]] = None,
                        settle_tolerance: Optional[Dict[str, float]] = None,
                        dwell_time: float = 1.0,
                        detector_lag: float = 0.0,
                        **kwargs):
        """
        Configure Z-scan with optional XY compensation.

        Scan ID will be assigned by DataStorager with 'Z' prefix.
        """
        from core.scan_patterns import generate_z_scan
        from core.scan_patterns import apply_single_direction_1d

        # Default compensation
        z_comp = z_compensation or {'z_ref': 0, 'x_ratio': 0, 'y_ratio': 0}

        # Generate Z scan with compensation
        z_pattern = generate_z_scan(
            x_nominal_nm=x_nominal,
            y_nominal_nm=y_nominal,
            z_start_nm=z_start,
            z_end_nm=z_end,
            num_points=num_points,
            z_ref_nm=z_comp.get('z_ref', 0),
            x_ratio=z_comp.get('x_ratio', 0),
            y_ratio=z_comp.get('y_ratio', 0)
        )

        # Apply single direction pattern
        self.pattern = apply_single_direction_1d(z_pattern)

        # Setup metadata (scan_id will be assigned by DataStorager)
        self.metadata = ScanMetadata(
            scan_id='Z_scanning',  # Temporary ID
            scan_type='Z_scan',
            timestamp=datetime.now().isoformat(),
            x_range_nm=(x_nominal, x_nominal),
            y_range_nm=(y_nominal, y_nominal),
            z_range_nm=(z_start, z_end),
            num_points_z=num_points,
            num_points_total=len(self.pattern),
            pattern_type='single',
            z_compensation=z_comp,
            settle_tolerance_nm=settle_tolerance or {'X': 5, 'Y': 5, 'Z': 10},
            dwell_time_s=dwell_time,
            detector_lag_s=detector_lag,
            **kwargs
        )

        # Update acquisition params
        self.dwell_params['dwell_time'] = dwell_time
        self.dwell_params['detector_lag'] = detector_lag

    def configure_multi_z_scan(self,
                               xy_points: List[Tuple[float, float]],
                               z_range: Tuple[float, float],
                               num_z_steps: int,
                               z_compensation: Optional[Dict[str, float]] = None,
                               settle_tolerance: Optional[Dict[str, float]] = None,
                               dwell_time: float = 1.0,
                               detector_lag: float = 0.0,
                               notes: str = '',
                               **kwargs):
        """Configure Multi-Z scan: Visit multiple (X,Y) points at each Z height."""
        from core.scan_patterns import generate_multi_z_scan
        from core.scan_patterns import apply_single_direction_1d
        from datetime import datetime

        # Default compensation
        z_comp = z_compensation or {'z_ref': 0, 'x_ratio': 0, 'y_ratio': 0}

        # Generate Z positions
        z_start, z_end = z_range
        if num_z_steps == 1:
            z_positions = [0.5 * (z_start + z_end)]
        else:
            step = (z_end - z_start) / (num_z_steps - 1)
            z_positions = [z_start + i * step for i in range(num_z_steps)]

        # Generate Multi-Z pattern with compensation
        multi_z_pattern = generate_multi_z_scan(
            xy_points=xy_points,
            z_positions=z_positions,
            z_compensation=z_comp
        )

        # Apply single direction pattern
        self.pattern = apply_single_direction_1d(multi_z_pattern)

        # Calculate bounds for metadata
        xs = [x for x, y in xy_points]
        ys = [y for x, y in xy_points]
        x_range_meta = (min(xs), max(xs))
        y_range_meta = (min(ys), max(ys))

        # Setup metadata with Multi-Z specific params in extra_fields
        self.metadata = ScanMetadata(
            scan_id='MZ_scanning',
            scan_type='Multi_Z',
            timestamp=datetime.now().isoformat(),
            x_range_nm=x_range_meta,
            y_range_nm=y_range_meta,
            z_range_nm=z_range,
            num_points_total=len(self.pattern),
            pattern_type='z_outer',
            settle_tolerance_nm=settle_tolerance or {'X': 5, 'Y': 5, 'Z': 10},
            dwell_time_s=dwell_time,
            detector_lag_s=detector_lag,
            z_compensation=z_comp,
            notes=notes,
            # Multi-Z specific parameters stored in extra_fields
            extra_fields={
                'xy_points': xy_points,
                'num_xy_points': len(xy_points),
                'num_z_steps': num_z_steps,
                'z_positions': z_positions,
                **kwargs  # Any additional kwargs passed by user
            }
        )

        # Update acquisition params
        self.dwell_params['dwell_time'] = dwell_time
        self.dwell_params['detector_lag'] = detector_lag

    def _build_move_target(self, point_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Build movement command from pattern point.

        For 2D scans: Z/R set once on first move, then only X/Y
        For other scans: All axes from pattern included
        """
        # Keep uppercase keys for CommandSender
        move_target = {k: v for k, v in point_dict.items()
                      if k in ['X', 'Y', 'Z', 'R']}

        # For 2D scans: Add fixed Z/R on first move only
        if self.metadata.scan_type == '2D':
            if not self._fixed_axes_set:
                # First move: set Z and R if specified
                if self._z_fixed is not None:
                    move_target['Z'] = self._z_fixed
                if self._r_fixed is not None:
                    move_target['R'] = self._r_fixed
                self._fixed_axes_set = True
            # Subsequent moves: Only X, Y (already in move_target)

        return move_target

    def execute_scan(self,
                    on_point_complete: Optional[Callable] = None,
                    on_progress: Optional[Callable] = None) -> ScanResult:
        """
        Execute configured scan.

        Args:
            on_point_complete: callback(index, position, signal_value)
            on_progress: callback(current, total, elapsed_time)

        Returns:
            ScanResult with status, data, and metadata
            Note: scan_id in metadata is temporary; DataStorager assigns final ID
        """
        if self.pattern is None or self.metadata is None:
            raise RuntimeError("Scan not configured. Call configure_*_scan() first.")

        from core.acquisition_primitives import acquire_point

        # Snapshot current Z and R positions if not specified by user
        # This ensures metadata always has the actual scan positions
        if self.metadata.scan_type in ['2D', '1D_line']:
            # Only snapshot for 2D and 1D scans (Z-scan manages its own Z)
            if self.metadata.z_fixed_nm is None or self.metadata.r_fixed_udeg is None:
                # Give MQTT a moment to receive position data if connection is fresh
                import time as time_module
                time_module.sleep(0.1)

                current_pos = self.reader.get_xyzr()
                if current_pos and current_pos.get('t') is not None:  # Check if we have valid data
                    if self.metadata.z_fixed_nm is None and current_pos.get('Z_nm') is not None:
                        self.metadata.z_fixed_nm = current_pos['Z_nm']
                        print(f"Snapshot Z position: {current_pos['Z_nm']:.1f} nm")
                    if self.metadata.r_fixed_udeg is None and current_pos.get('R_udeg') is not None:
                        self.metadata.r_fixed_udeg = current_pos['R_udeg']
                        print(f"Snapshot R position: {current_pos['R_udeg']:.1f} µdeg")
                else:
                    print("Warning: Could not snapshot Z/R positions - no position data available yet")

        # Initialize
        self._paused = False
        self._cancelled = False
        raw_datapoints = []

        total_points = len(self.pattern)
        start_time = time.time()
        self.metadata.scan_start_time = datetime.now().isoformat()

        print(f"\n{'='*60}")
        print(f"Starting {self.metadata.scan_type} scan")
        print(f"Total points: {total_points}")
        print(f"Pattern: {self.metadata.pattern_type}")
        print(f"{'='*60}\n")

        # Scan loop
        for i, point_dict in enumerate(self.pattern):
            # Check cancellation
            if self._cancelled:
                print("\nScan cancelled by user")
                self.commander.stop_all()
                # Still finalize to save partial data
                return self._finalize_scan(
                    raw_datapoints,
                    status='cancelled',
                    start_time=start_time
                )

            # Handle pause
            while self._paused:
                time.sleep(0.1)

            # Prepare movement command
            move_target = self._build_move_target(point_dict)

            # Move to position
            try:
                self.commander.move_to(**move_target)
            except Exception as e:
                print(f"\nError moving to position {i}: {e}")
                self.commander.stop_all()
                return self._finalize_scan(
                    raw_datapoints,
                    status='error',
                    error_message=f"Move error: {str(e)}",
                    start_time=start_time
                )

            # Acquire data point
            try:
                datapoint = acquire_point(
                    reader=self.reader,
                    target=move_target,
                    tolerance=self.metadata.settle_tolerance_nm,
                    settle_params=self.settle_params,
                    dwell_params=self.dwell_params
                )

                raw_datapoints.append(datapoint)

                # Callback: point complete
                if on_point_complete:
                    if self.metadata.scan_type == '2D' and 'row' in point_dict and 'col' in point_dict:
                        index = (point_dict['row'], point_dict['col'])
                    else:
                        index = point_dict['idx']

                    on_point_complete(
                        index,
                        point_dict,
                        datapoint.avg_signal
                    )

                # Callback: progress
                if on_progress and (i % max(1, total_points // 20) == 0 or i == total_points - 1):
                    elapsed = time.time() - start_time
                    on_progress(i + 1, total_points, elapsed)

            except Exception as e:
                print(f"\nError at point {i}: {e}")
                self.commander.stop_all()
                # Still save partial data
                return self._finalize_scan(
                    raw_datapoints,
                    status='error',
                    error_message=str(e),
                    start_time=start_time
                )

        # Scan completed successfully
        print(f"\nScan completed: {total_points} points")
        return self._finalize_scan(raw_datapoints, status='completed', start_time=start_time)

    def pause_scan(self):
        """Pause the current scan"""
        self._paused = True
        print("Scan paused")

    def resume_scan(self):
        """Resume paused scan"""
        self._paused = False
        print("Scan resumed")

    def cancel_scan(self):
        """Cancel the current scan (data will still be saved)"""
        self._cancelled = True

    def _finalize_scan(self,
                      raw_datapoints: List[Any],
                      status: str,
                      start_time: float,
                      error_message: Optional[str] = None) -> ScanResult:
        """
        Finalize scan and prepare results.

        Always saves data, even if cancelled or errored.
        """
        # Update metadata timing
        end_time = time.time()
        self.metadata.scan_end_time = datetime.now().isoformat()
        self.metadata.scan_duration_s = end_time - start_time

        # Reconstruct data
        reconstructed = None
        if raw_datapoints:
            reconstructed = self._reconstruct_data(raw_datapoints)

        # Calculate statistics
        stats = None
        if raw_datapoints:
            signals = [dp.avg_signal for dp in raw_datapoints]
            stats = {
                'num_acquired': len(raw_datapoints),
                'num_valid': sum(1 for s in signals if not np.isnan(s)),
                'signal_mean': np.nanmean(signals),
                'signal_std': np.nanstd(signals),
                'signal_min': np.nanmin(signals),
                'signal_max': np.nanmax(signals),
            }

        return ScanResult(
            status=status,
            metadata=self.metadata,
            raw_datapoints=raw_datapoints,
            reconstructed_data=reconstructed,
            statistics=stats,
            error_message=error_message
        )

    def _reconstruct_data(self, raw_datapoints: List[Any]) -> np.ndarray:
        """
        Reconstruct data into proper shape based on scan type.

        Uses row/col information to handle snake patterns correctly.
        """
        if self.metadata.scan_type == '2D':
            # Use row/col from pattern points
            rows = self.metadata.num_points_y
            cols = self.metadata.num_points_x
            data = np.full((rows, cols), np.nan, dtype=np.float64)

            for dp, point in zip(raw_datapoints, self.pattern):
                if 'row' in point and 'col' in point:
                    # Convert None to nan
                    value = dp.avg_signal if dp.avg_signal is not None else np.nan
                    data[point['row'], point['col']] = float(value)

            return data

        elif self.metadata.scan_type == '1D_line':
            # Simple 1D array, converting None to nan
            values = [float(dp.avg_signal) if dp.avg_signal is not None else np.nan
                     for dp in raw_datapoints]
            return np.array(values, dtype=np.float64)

        elif self.metadata.scan_type == 'Z_scan':
            # 1D Z profile with compensated positions
            positions = []
            signals = []
            for dp in raw_datapoints:
                # Extract Z position
                pos = dp.position.get('Z_nm', np.nan) if dp.position else np.nan
                sig = float(dp.avg_signal) if dp.avg_signal is not None else np.nan
                positions.append(pos)
                signals.append(sig)

            return np.column_stack([
                np.array(positions, dtype=np.float64),
                np.array(signals, dtype=np.float64)
            ])

        return None