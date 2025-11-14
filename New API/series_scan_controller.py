"""
Series Scan Controller

Implements multi-slice scanning capabilities:
- Z-Series: Repeat scans at different Z heights
- R-Series: Repeat scans at different rotation angles

Architecture:
- Z-Series wraps: 2D-rectangular, 2D-vertices
- R-Series wraps: 2D-rectangular, 2D-vertices, Z-scan, Multi-Z

R-Series supports two rotation modes:
- SIMPLE: Only rotate stage, use original coordinates
- COR: Transform coordinates around center of rotation with Z-compensation
"""

from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from scan_controller import ScanController, ScanResult
from core.utils import (
    apply_rotation_with_z_compensation,
    transform_scan_area_aabb,
    transform_scan_area_center_rotate
)


class RotationMode(Enum):
    """Rotation behavior for R-series scans"""
    SIMPLE = 'simple'  # Rotate stage only, no coordinate transform
    COR = 'cor'        # Transform coordinates around center of rotation


class CORTransformMode(Enum):
    """COR transformation strategy"""
    CENTER_ROTATE = 'center_rotate'  # Rotate center, keep size (preferred)
    AABB = 'aabb'                     # Transform corners, find bounding box (legacy)


@dataclass
class ZSeriesResult:
    """Result from Z-series scan"""
    slices: List[ScanResult]
    z_positions: List[float]
    z_compensation: Dict[str, float]
    metadata: Dict
    status: str = 'completed'


@dataclass
class RSeriesResult:
    """Result from R-series scan"""
    slices: List[ScanResult]
    r_angles_udeg: List[float]
    rotation_mode: str
    cor: Optional[Tuple[float, float]]
    cor_base_z: Optional[float]
    metadata: Dict
    status: str = 'completed'


class ZSeriesScanController:
    """
    Multi-slice Z-series scanning.
    
    Repeats 2D scans (rectangular or vertices) at different Z heights
    with automatic X/Y compensation for tilted beam geometry.
    
    Features:
    - Z-outer loop: scan at Z0, then Z1, etc.
    - Automatic XY compensation at each Z
    - Preserves scan pattern at each slice
    
    Usage:
        z_controller = ZSeriesScanController(commander, reader)
        
        # Configure Z-series of rectangular 2D scans
        z_controller.configure_z_series_from_2d(
            x_range=(0, 10000),
            y_range=(0, 10000),
            num_points=(100, 100),
            z_range=(0, 5000),
            z_numbers=10,
            z_compensation={'z_ref': 0, 'x_ratio': 1.0, 'y_ratio': 0.0}
        )
        
        # Execute
        result = z_controller.execute_z_series()
    """
    
    def __init__(self, commander, reader):
        """
        Initialize Z-series controller.
        
        Args:
            commander: CommandSender instance
            reader: PositionSignalReader instance
        """
        self.base_controller = ScanController(commander, reader)
        self.z_positions = []
        self.z_compensation = {}
        self.base_scan_config = {}
        self.slices = []
        
    def configure_z_series_from_2d(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        num_points: Tuple[int, int],
        z_range: Tuple[float, float],
        z_numbers: int,
        z_compensation: Optional[Dict[str, float]] = None,
        pattern: str = 'snake',
        fast_axis: str = 'X',
        r_fixed_udeg: Optional[float] = None,
        settle_tolerance: Optional[Dict[str, float]] = None,
        dwell_time: float = 1.0,
        detector_lag: float = 0.0,
        notes: str = '',
        **kwargs
    ):
        """
        Configure Z-series using rectangular 2D scan.
        
        Args:
            x_range: (x_start, x_end) in nm
            y_range: (y_start, y_end) in nm
            num_points: (nx, ny) pixels
            z_range: (z_start, z_end) in nm
            z_numbers: Number of Z slices
            z_compensation: {'z_ref': 0, 'x_ratio': 1.0, 'y_ratio': 0.0}
            pattern: 'raster' or 'snake'
            fast_axis: 'X' or 'Y'
            r_fixed_udeg: Fixed rotation angle (µdeg)
            settle_tolerance: Per-axis settling tolerances (nm)
            dwell_time: Dwell time per point (s)
            detector_lag: Detector lag compensation (s)
            notes: Scan notes
        """
        # Store base scan configuration
        self.base_scan_config = {
            'type': '2D_rectangular',
            'x_range': x_range,
            'y_range': y_range,
            'num_points': num_points,
            'pattern': pattern,
            'fast_axis': fast_axis,
            'r_fixed_udeg': r_fixed_udeg,
            'settle_tolerance': settle_tolerance,
            'dwell_time': dwell_time,
            'detector_lag': detector_lag,
            'notes': notes,
            **kwargs
        }
        
        # Calculate Z positions
        z_start, z_end = z_range
        if z_numbers == 1:
            self.z_positions = [0.5 * (z_start + z_end)]
        else:
            step = (z_end - z_start) / (z_numbers - 1)
            self.z_positions = [z_start + i * step for i in range(z_numbers)]
        
        # Store compensation
        self.z_compensation = z_compensation or {'z_ref': 0, 'x_ratio': 0, 'y_ratio': 0}
        
    def configure_z_series_from_vertices(
        self,
        vertices_nm: List[Tuple[float, float]],
        num_points: Tuple[int, int],
        z_range: Tuple[float, float],
        z_numbers: int,
        z_compensation: Optional[Dict[str, float]] = None,
        pattern: str = 'snake',
        fast_axis: str = 'X',
        r_fixed_udeg: Optional[float] = None,
        settle_tolerance: Optional[Dict[str, float]] = None,
        dwell_time: float = 1.0,
        detector_lag: float = 0.0,
        notes: str = '',
        **kwargs
    ):
        """
        Configure Z-series using vertices 2D scan.
        
        Args:
            vertices_nm: List of (x, y) polygon vertices
            num_points: (nx, ny) pixels
            z_range: (z_start, z_end) in nm
            z_numbers: Number of Z slices
            z_compensation: {'z_ref': 0, 'x_ratio': 1.0, 'y_ratio': 0.0}
            pattern: 'raster' or 'snake'
            fast_axis: 'X' or 'Y'
            r_fixed_udeg: Fixed rotation angle (µdeg)
            settle_tolerance: Per-axis settling tolerances (nm)
            dwell_time: Dwell time per point (s)
            detector_lag: Detector lag compensation (s)
            notes: Scan notes
        """
        # Store base scan configuration
        self.base_scan_config = {
            'type': '2D_vertices',
            'vertices_nm': vertices_nm,
            'num_points': num_points,
            'pattern': pattern,
            'fast_axis': fast_axis,
            'r_fixed_udeg': r_fixed_udeg,
            'settle_tolerance': settle_tolerance,
            'dwell_time': dwell_time,
            'detector_lag': detector_lag,
            'notes': notes,
            **kwargs
        }
        
        # Calculate Z positions
        z_start, z_end = z_range
        if z_numbers == 1:
            self.z_positions = [0.5 * (z_start + z_end)]
        else:
            step = (z_end - z_start) / (z_numbers - 1)
            self.z_positions = [z_start + i * step for i in range(z_numbers)]
        
        # Store compensation
        self.z_compensation = z_compensation or {'z_ref': 0, 'x_ratio': 0, 'y_ratio': 0}
        
    def execute_z_series(
        self,
        on_slice_complete: Optional[Callable[[int, ScanResult], None]] = None
    ) -> ZSeriesResult:
        """
        Execute Z-series scan.
        
        Args:
            on_slice_complete: Callback(z_index, scan_result) after each slice
        
        Returns:
            ZSeriesResult with all slices
        """
        self.slices = []
        
        for z_idx, z_pos in enumerate(self.z_positions):
            print(f"\nZ-series: slice {z_idx + 1}/{len(self.z_positions)} at Z={z_pos:.0f} nm")
            
            # Configure base scan at this Z with compensation
            if self.base_scan_config['type'] == '2D_rectangular':
                # Apply XY compensation to scan ranges
                x_start, x_end = self.base_scan_config['x_range']
                y_start, y_end = self.base_scan_config['y_range']
                
                z_offset = z_pos - self.z_compensation.get('z_ref', 0)
                x_comp = z_offset * self.z_compensation.get('x_ratio', 0)
                y_comp = z_offset * self.z_compensation.get('y_ratio', 0)
                
                self.base_controller.configure_2d_scan(
                    x_range=(x_start + x_comp, x_end + x_comp),
                    y_range=(y_start + y_comp, y_end + y_comp),
                    num_points=self.base_scan_config['num_points'],
                    pattern=self.base_scan_config['pattern'],
                    fast_axis=self.base_scan_config['fast_axis'],
                    z_fixed_nm=z_pos,
                    r_fixed_udeg=self.base_scan_config.get('r_fixed_udeg'),
                    settle_tolerance=self.base_scan_config.get('settle_tolerance'),
                    dwell_time=self.base_scan_config['dwell_time'],
                    detector_lag=self.base_scan_config['detector_lag'],
                    notes=f"{self.base_scan_config['notes']} | Z-series slice {z_idx + 1}/{len(self.z_positions)}"
                )
            
            elif self.base_scan_config['type'] == '2D_vertices':
                # Apply XY compensation to vertices
                vertices = self.base_scan_config['vertices_nm']
                
                z_offset = z_pos - self.z_compensation.get('z_ref', 0)
                x_comp = z_offset * self.z_compensation.get('x_ratio', 0)
                y_comp = z_offset * self.z_compensation.get('y_ratio', 0)
                
                compensated_vertices = [(x + x_comp, y + y_comp) for x, y in vertices]
                
                self.base_controller.configure_2d_vertices_scan(
                    vertices_nm=compensated_vertices,
                    num_points=self.base_scan_config['num_points'],
                    pattern=self.base_scan_config['pattern'],
                    fast_axis=self.base_scan_config['fast_axis'],
                    z_fixed_nm=z_pos,
                    r_fixed_udeg=self.base_scan_config.get('r_fixed_udeg'),
                    settle_tolerance=self.base_scan_config.get('settle_tolerance'),
                    dwell_time=self.base_scan_config['dwell_time'],
                    detector_lag=self.base_scan_config['detector_lag'],
                    notes=f"{self.base_scan_config['notes']} | Z-series slice {z_idx + 1}/{len(self.z_positions)}"
                )
            
            # Execute slice
            slice_result = self.base_controller.execute_scan()
            self.slices.append(slice_result)
            
            # Callback
            if on_slice_complete:
                on_slice_complete(z_idx, slice_result)
        
        # Build result
        return ZSeriesResult(
            slices=self.slices,
            z_positions=self.z_positions,
            z_compensation=self.z_compensation,
            metadata={
                'series_type': 'Z',
                'base_config': self.base_scan_config,
                'timestamp': datetime.now().isoformat()
            },
            status='completed'
        )


class RSeriesScanController:
    """
    Multi-angle R-series scanning.
    
    Repeats scans at different rotation angles with two modes:
    - SIMPLE: Only rotate stage
    - COR: Transform coordinates around center of rotation
    
    Can wrap:
    - 2D rectangular scans
    - 2D vertices scans
    - 1D Z scans (old "ZR scan")
    - Multi-Z scans (tomography!)
    
    Features:
    - COR with Z-compensation: COR position shifts with Z
    - Two transform modes: center_rotate (preferred) and aabb (legacy)
    - Rotation relative to base R position
    
    Usage:
        r_controller = RSeriesScanController(commander, reader)
        
        # Configure COR mode
        r_controller.set_rotation_mode(
            mode='cor',
            cor_x=5000, cor_y=5000,
            cor_base_z=0,
            cor_x_ratio=1.0, cor_y_ratio=0.0,
            transform_mode='center_rotate'
        )
        
        # Configure R-series of 2D scans
        r_controller.configure_r_series_from_2d(
            x_range=(0, 10000),
            y_range=(0, 10000),
            num_points=(100, 100),
            r_start_udeg=0,
            r_end_udeg=180000,
            r_numbers=4
        )
        
        # Execute
        result = r_controller.execute_r_series()
    """
    
    def __init__(self, commander, reader):
        """
        Initialize R-series controller.
        
        Args:
            commander: CommandSender instance
            reader: PositionSignalReader instance
        """
        self.base_controller = ScanController(commander, reader)
        
        # Rotation configuration
        self.rotation_mode = RotationMode.SIMPLE
        self.cor_x = None
        self.cor_y = None
        self.cor_base_z = 0.0
        self.cor_x_ratio = 0.0
        self.cor_y_ratio = 0.0
        self.cor_transform_mode = CORTransformMode.CENTER_ROTATE
        
        # Series configuration
        self.r_angles_udeg = []
        self.r_base_udeg = 0.0
        self.base_scan_config = {}
        self.slices = []
        
    def set_rotation_mode(
        self,
        mode: str,
        cor_x: Optional[float] = None,
        cor_y: Optional[float] = None,
        cor_base_z: float = 0.0,
        cor_x_ratio: float = 0.0,
        cor_y_ratio: float = 0.0,
        transform_mode: str = 'center_rotate'
    ):
        """
        Set rotation behavior.
        
        Args:
            mode: 'simple' or 'cor'
            cor_x, cor_y: Center of rotation at base Z (nm)
            cor_base_z: Z where COR coordinates are valid (nm)
            cor_x_ratio: COR X shift per Z (1.0 = 45° beam)
            cor_y_ratio: COR Y shift per Z (typically 0.0)
            transform_mode: 'center_rotate' or 'aabb'
        """
        self.rotation_mode = RotationMode(mode)
        
        if self.rotation_mode == RotationMode.COR:
            if cor_x is None or cor_y is None:
                raise ValueError("COR mode requires cor_x and cor_y")
            self.cor_x = cor_x
            self.cor_y = cor_y
            self.cor_base_z = cor_base_z
            self.cor_x_ratio = cor_x_ratio
            self.cor_y_ratio = cor_y_ratio
            self.cor_transform_mode = CORTransformMode(transform_mode)
            
    def configure_r_series_from_2d(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        num_points: Tuple[int, int],
        r_start_udeg: float,
        r_end_udeg: float,
        r_numbers: int,
        r_base_udeg: float = 0.0,
        pattern: str = 'snake',
        fast_axis: str = 'X',
        z_fixed_nm: Optional[float] = None,
        settle_tolerance: Optional[Dict[str, float]] = None,
        dwell_time: float = 1.0,
        detector_lag: float = 0.0,
        notes: str = '',
        **kwargs
    ):
        """
        Configure R-series of rectangular 2D scans.
        
        Args:
            x_range: (x_start, x_end) in nm
            y_range: (y_start, y_end) in nm
            num_points: (nx, ny) pixels
            r_start_udeg: Starting rotation (µdeg)
            r_end_udeg: Ending rotation (µdeg)
            r_numbers: Number of R slices
            r_base_udeg: Base rotation angle (µdeg)
            pattern: 'raster' or 'snake'
            fast_axis: 'X' or 'Y'
            z_fixed_nm: Fixed Z position (nm)
            settle_tolerance: Per-axis settling tolerances (nm)
            dwell_time: Dwell time per point (s)
            detector_lag: Detector lag compensation (s)
            notes: Scan notes
        """
        # Store base scan configuration
        self.base_scan_config = {
            'type': '2D_rectangular',
            'x_range': x_range,
            'y_range': y_range,
            'num_points': num_points,
            'pattern': pattern,
            'fast_axis': fast_axis,
            'z_fixed_nm': z_fixed_nm or 0.0,
            'settle_tolerance': settle_tolerance,
            'dwell_time': dwell_time,
            'detector_lag': detector_lag,
            'notes': notes,
            **kwargs
        }
        
        # Calculate R angles
        self.r_base_udeg = r_base_udeg
        if r_numbers == 1:
            self.r_angles_udeg = [0.5 * (r_start_udeg + r_end_udeg)]
        else:
            step = (r_end_udeg - r_start_udeg) / (r_numbers - 1)
            self.r_angles_udeg = [r_start_udeg + i * step for i in range(r_numbers)]
            
    def configure_r_series_from_vertices(
        self,
        vertices_nm: List[Tuple[float, float]],
        num_points: Tuple[int, int],
        r_start_udeg: float,
        r_end_udeg: float,
        r_numbers: int,
        r_base_udeg: float = 0.0,
        pattern: str = 'snake',
        fast_axis: str = 'X',
        z_fixed_nm: Optional[float] = None,
        settle_tolerance: Optional[Dict[str, float]] = None,
        dwell_time: float = 1.0,
        detector_lag: float = 0.0,
        notes: str = '',
        **kwargs
    ):
        """
        Configure R-series of vertices 2D scans.
        
        Args:
            vertices_nm: List of (x, y) polygon vertices
            num_points: (nx, ny) pixels
            r_start_udeg: Starting rotation (µdeg)
            r_end_udeg: Ending rotation (µdeg)
            r_numbers: Number of R slices
            r_base_udeg: Base rotation angle (µdeg)
            pattern: 'raster' or 'snake'
            fast_axis: 'X' or 'Y'
            z_fixed_nm: Fixed Z position (nm)
            settle_tolerance: Per-axis settling tolerances (nm)
            dwell_time: Dwell time per point (s)
            detector_lag: Detector lag compensation (s)
            notes: Scan notes
        """
        # Store base scan configuration
        self.base_scan_config = {
            'type': '2D_vertices',
            'vertices_nm': vertices_nm,
            'num_points': num_points,
            'pattern': pattern,
            'fast_axis': fast_axis,
            'z_fixed_nm': z_fixed_nm or 0.0,
            'settle_tolerance': settle_tolerance,
            'dwell_time': dwell_time,
            'detector_lag': detector_lag,
            'notes': notes,
            **kwargs
        }
        
        # Calculate R angles
        self.r_base_udeg = r_base_udeg
        if r_numbers == 1:
            self.r_angles_udeg = [0.5 * (r_start_udeg + r_end_udeg)]
        else:
            step = (r_end_udeg - r_start_udeg) / (r_numbers - 1)
            self.r_angles_udeg = [r_start_udeg + i * step for i in range(r_numbers)]
            
    def configure_r_series_from_z_scan(
        self,
        x_nominal: float,
        y_nominal: float,
        z_range: Tuple[float, float],
        num_z_points: int,
        r_start_udeg: float,
        r_end_udeg: float,
        r_numbers: int,
        r_base_udeg: float = 0.0,
        z_compensation: Optional[Dict[str, float]] = None,
        settle_tolerance: Optional[Dict[str, float]] = None,
        dwell_time: float = 1.0,
        detector_lag: float = 0.0,
        notes: str = '',
        **kwargs
    ):
        """
        Configure R-series of Z scans (old "ZR scan").
        
        Note: For Z-scan in R-series, ALWAYS use COR transformation.
        The single (X,Y) point is transformed around COR at each R angle.
        
        Args:
            x_nominal, y_nominal: Base XY position (nm)
            z_range: (z_start, z_end) in nm
            num_z_points: Number of Z points
            r_start_udeg: Starting rotation (µdeg)
            r_end_udeg: Ending rotation (µdeg)
            r_numbers: Number of R slices
            r_base_udeg: Base rotation angle (µdeg)
            z_compensation: {'z_ref': 0, 'x_ratio': 1.0, 'y_ratio': 0.0}
            settle_tolerance: Per-axis settling tolerances (nm)
            dwell_time: Dwell time per point (s)
            detector_lag: Detector lag compensation (s)
            notes: Scan notes
        """
        if self.rotation_mode != RotationMode.COR:
            raise ValueError("Z-scan in R-series requires COR transformation mode")
        
        # Store base scan configuration
        self.base_scan_config = {
            'type': 'Z_scan',
            'x_nominal': x_nominal,
            'y_nominal': y_nominal,
            'z_range': z_range,
            'num_z_points': num_z_points,
            'z_compensation': z_compensation or {'z_ref': 0, 'x_ratio': 0, 'y_ratio': 0},
            'settle_tolerance': settle_tolerance,
            'dwell_time': dwell_time,
            'detector_lag': detector_lag,
            'notes': notes,
            **kwargs
        }
        
        # Calculate R angles
        self.r_base_udeg = r_base_udeg
        if r_numbers == 1:
            self.r_angles_udeg = [0.5 * (r_start_udeg + r_end_udeg)]
        else:
            step = (r_end_udeg - r_start_udeg) / (r_numbers - 1)
            self.r_angles_udeg = [r_start_udeg + i * step for i in range(r_numbers)]
            
    def configure_r_series_from_multi_z(
        self,
        xy_points: List[Tuple[float, float]],
        z_range: Tuple[float, float],
        num_z_steps: int,
        r_start_udeg: float,
        r_end_udeg: float,
        r_numbers: int,
        r_base_udeg: float = 0.0,
        z_compensation: Optional[Dict[str, float]] = None,
        settle_tolerance: Optional[Dict[str, float]] = None,
        dwell_time: float = 1.0,
        detector_lag: float = 0.0,
        notes: str = '',
        **kwargs
    ):
        """
        Configure R-series of Multi-Z scans (tomography!).
        
        This is the ultimate scan: multiple XY points at each Z, repeated at each R.
        Perfect for 3D reconstruction.
        
        Args:
            xy_points: List of (x, y) coordinates
            z_range: (z_start, z_end) in nm
            num_z_steps: Number of Z slices
            r_start_udeg: Starting rotation (µdeg)
            r_end_udeg: Ending rotation (µdeg)
            r_numbers: Number of R angles
            r_base_udeg: Base rotation angle (µdeg)
            z_compensation: {'z_ref': 0, 'x_ratio': 1.0, 'y_ratio': 0.0}
            settle_tolerance: Per-axis settling tolerances (nm)
            dwell_time: Dwell time per point (s)
            detector_lag: Detector lag compensation (s)
            notes: Scan notes
        """
        # Store base scan configuration
        self.base_scan_config = {
            'type': 'Multi_Z',
            'xy_points': xy_points,
            'z_range': z_range,
            'num_z_steps': num_z_steps,
            'z_compensation': z_compensation or {'z_ref': 0, 'x_ratio': 0, 'y_ratio': 0},
            'settle_tolerance': settle_tolerance,
            'dwell_time': dwell_time,
            'detector_lag': detector_lag,
            'notes': notes,
            **kwargs
        }
        
        # Calculate R angles
        self.r_base_udeg = r_base_udeg
        if r_numbers == 1:
            self.r_angles_udeg = [0.5 * (r_start_udeg + r_end_udeg)]
        else:
            step = (r_end_udeg - r_start_udeg) / (r_numbers - 1)
            self.r_angles_udeg = [r_start_udeg + i * step for i in range(r_numbers)]
        
    def _apply_rotation_transform(self, r_angle_udeg: float):
        """
        Apply rotation transformation to scan parameters.
        
        Args:
            r_angle_udeg: Current rotation angle (µdeg)
        
        Returns:
            Transformed scan parameters dict
        """
        config = self.base_scan_config.copy()
        current_z = config.get('z_fixed_nm', self.cor_base_z)
        
        if self.rotation_mode == RotationMode.SIMPLE:
            # Simple mode: just set R angle, no coordinate transform
            config['r_fixed_udeg'] = r_angle_udeg
            return config
            
        elif self.rotation_mode == RotationMode.COR:
            # COR mode: transform coordinates
            
            if config['type'] == '2D_rectangular':
                # Transform scan area
                x_start, x_end = config['x_range']
                y_start, y_end = config['y_range']
                
                if self.cor_transform_mode == CORTransformMode.CENTER_ROTATE:
                    x_s, x_e, y_s, y_e = transform_scan_area_center_rotate(
                        x_start, x_end, y_start, y_end,
                        r_angle_udeg, self.r_base_udeg,
                        self.cor_x, self.cor_y,
                        current_z, self.cor_base_z,
                        self.cor_x_ratio, self.cor_y_ratio
                    )
                else:  # AABB
                    x_s, x_e, y_s, y_e = transform_scan_area_aabb(
                        x_start, x_end, y_start, y_end,
                        r_angle_udeg, self.r_base_udeg,
                        self.cor_x, self.cor_y,
                        current_z, self.cor_base_z,
                        self.cor_x_ratio, self.cor_y_ratio
                    )
                
                config['x_range'] = (x_s, x_e)
                config['y_range'] = (y_s, y_e)
                
            elif config['type'] == '2D_vertices':
                # Transform vertices
                vertices = config['vertices_nm']
                rotated = []
                for x, y in vertices:
                    x_rot, y_rot = apply_rotation_with_z_compensation(
                        x, y, r_angle_udeg, self.r_base_udeg,
                        self.cor_x, self.cor_y,
                        current_z, self.cor_base_z,
                        self.cor_x_ratio, self.cor_y_ratio
                    )
                    rotated.append((x_rot, y_rot))
                config['vertices_nm'] = rotated
                
            elif config['type'] == 'Z_scan':
                # Transform single XY point
                x_nom = config['x_nominal']
                y_nom = config['y_nominal']
                
                x_rot, y_rot = apply_rotation_with_z_compensation(
                    x_nom, y_nom, r_angle_udeg, self.r_base_udeg,
                    self.cor_x, self.cor_y,
                    current_z, self.cor_base_z,
                    self.cor_x_ratio, self.cor_y_ratio
                )
                config['x_nominal'] = x_rot
                config['y_nominal'] = y_rot
                
            elif config['type'] == 'Multi_Z':
                # Transform all XY points
                xy_points = config['xy_points']
                rotated = []
                for x, y in xy_points:
                    # For multi-Z, we rotate around COR but each Z will have
                    # its own compensation applied during pattern generation
                    x_rot, y_rot = apply_rotation_with_z_compensation(
                        x, y, r_angle_udeg, self.r_base_udeg,
                        self.cor_x, self.cor_y,
                        current_z, self.cor_base_z,
                        self.cor_x_ratio, self.cor_y_ratio
                    )
                    rotated.append((x_rot, y_rot))
                config['xy_points'] = rotated
            
            return config
    
    def execute_r_series(
        self,
        on_slice_complete: Optional[Callable[[int, ScanResult], None]] = None
    ) -> RSeriesResult:
        """
        Execute R-series scan.
        
        Args:
            on_slice_complete: Callback(r_index, scan_result) after each slice
        
        Returns:
            RSeriesResult with all slices
        """
        self.slices = []
        
        for r_idx, r_angle in enumerate(self.r_angles_udeg):
            print(f"\nR-series: slice {r_idx + 1}/{len(self.r_angles_udeg)} at R={r_angle/1000:.3f}° ({r_angle} µdeg)")
            
            # Apply rotation transformation
            transformed_config = self._apply_rotation_transform(r_angle)
            
            # Configure base scan with transformed parameters
            if transformed_config['type'] == '2D_rectangular':
                self.base_controller.configure_2d_scan(
                    x_range=transformed_config['x_range'],
                    y_range=transformed_config['y_range'],
                    num_points=transformed_config['num_points'],
                    pattern=transformed_config['pattern'],
                    fast_axis=transformed_config['fast_axis'],
                    z_fixed_nm=transformed_config.get('z_fixed_nm'),
                    r_fixed_udeg=transformed_config.get('r_fixed_udeg'),
                    settle_tolerance=transformed_config.get('settle_tolerance'),
                    dwell_time=transformed_config['dwell_time'],
                    detector_lag=transformed_config['detector_lag'],
                    notes=f"{transformed_config['notes']} | R-series slice {r_idx + 1}/{len(self.r_angles_udeg)}"
                )
                
            elif transformed_config['type'] == '2D_vertices':
                self.base_controller.configure_2d_vertices_scan(
                    vertices_nm=transformed_config['vertices_nm'],
                    num_points=transformed_config['num_points'],
                    pattern=transformed_config['pattern'],
                    fast_axis=transformed_config['fast_axis'],
                    z_fixed_nm=transformed_config.get('z_fixed_nm'),
                    r_fixed_udeg=transformed_config.get('r_fixed_udeg'),
                    settle_tolerance=transformed_config.get('settle_tolerance'),
                    dwell_time=transformed_config['dwell_time'],
                    detector_lag=transformed_config['detector_lag'],
                    notes=f"{transformed_config['notes']} | R-series slice {r_idx + 1}/{len(self.r_angles_udeg)}"
                )
                
            elif transformed_config['type'] == 'Z_scan':
                self.base_controller.configure_z_scan(
                    x_nominal=transformed_config['x_nominal'],
                    y_nominal=transformed_config['y_nominal'],
                    z_start=transformed_config['z_range'][0],
                    z_end=transformed_config['z_range'][1],
                    num_points=transformed_config['num_z_points'],
                    z_compensation=transformed_config['z_compensation'],
                    settle_tolerance=transformed_config.get('settle_tolerance'),
                    dwell_time=transformed_config['dwell_time'],
                    detector_lag=transformed_config['detector_lag']
                )
                
            elif transformed_config['type'] == 'Multi_Z':
                self.base_controller.configure_multi_z_scan(
                    xy_points=transformed_config['xy_points'],
                    z_range=transformed_config['z_range'],
                    num_z_steps=transformed_config['num_z_steps'],
                    z_compensation=transformed_config['z_compensation'],
                    settle_tolerance=transformed_config.get('settle_tolerance'),
                    dwell_time=transformed_config['dwell_time'],
                    detector_lag=transformed_config['detector_lag'],
                    notes=f"{transformed_config['notes']} | R-series slice {r_idx + 1}/{len(self.r_angles_udeg)}"
                )
            
            # Execute slice
            slice_result = self.base_controller.execute_scan()
            self.slices.append(slice_result)
            
            # Callback
            if on_slice_complete:
                on_slice_complete(r_idx, slice_result)
        
        # Build result
        return RSeriesResult(
            slices=self.slices,
            r_angles_udeg=self.r_angles_udeg,
            rotation_mode=self.rotation_mode.value,
            cor=(self.cor_x, self.cor_y) if self.rotation_mode == RotationMode.COR else None,
            cor_base_z=self.cor_base_z if self.rotation_mode == RotationMode.COR else None,
            metadata={
                'series_type': 'R',
                'base_config': self.base_scan_config,
                'cor_transform_mode': self.cor_transform_mode.value if self.rotation_mode == RotationMode.COR else None,
                'r_base_udeg': self.r_base_udeg,
                'timestamp': datetime.now().isoformat()
            },
            status='completed'
        )
