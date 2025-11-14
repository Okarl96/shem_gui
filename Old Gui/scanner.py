#!/usr/bin/env python3
import sys
import time
import json
import numpy as np
import sqlite3
import threading
import queue
from collections import deque
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from enum import Enum
import os
from dataclasses import dataclass, field
import traceback
import re
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import RectangleSelector

# Try to import h5py, provide fallback if not available
try:
    import h5py

    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("Warning: h5py not available. Raw data storage will be disabled.")

import paho.mqtt.client as mqtt
from PyQt5.QtWidgets import (
    QAction,QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QLineEdit, QComboBox,
    QTextEdit, QGroupBox, QSplitter, QDockWidget, QSpinBox,
    QDoubleSpinBox, QMessageBox, QCheckBox, QTabWidget,
    QProgressBar, QSlider, QFrame, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QScrollArea, QFormLayout,QDialog
)
from PyQt5.QtCore import (
    Qt, QTimer, pyqtSignal, pyqtSlot, QThread, QObject,
    QMutex, QMutexLocker, QDateTime, QSettings
)
from PyQt5.QtGui import QFont, QTextCursor, QIcon, QPalette, QColor, QPixmap

import pyqtgraph as pg
import pyqtgraph.exporters  # : Ensure exporters module is available

# Try to import scipy, provide fallback if not available
try:
    from scipy import interpolate, ndimage

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Advanced interpolation features will be disabled.")

from scipy.ndimage import shift as scipy_shift
try:
    from skimage.registration import phase_cross_correlation
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Image registration will use fallback method.")

from numpy.polynomial import Polynomial

# Configure pyqtgraph
pg.setConfigOptions(antialias=True, useOpenGL=True)
pg.setConfigOptions(background='w', foreground='k')
pg.setConfigOptions(imageAxisOrder='row-major')

class ScanMode(Enum):
    STEP_STOP = "step_stop"
    CONTINUOUS = "continuous"

class ScanState(Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    PAUSED = "paused"
    STOPPING = "stopping"

@dataclass
class ZSeriesParameters:
    """Z-series configuration for both 2D and 1D scans"""
    enabled: bool = False
    z_start: float = 0.0  # nm
    z_end: float = 1000.0  # nm
    z_numbers: int = 10
    z_step_input: float = 111.11  # nm
    x_compensation_ratio: float = 1.0  # X movement per Z movement (typically 1.0 for 45°, 0.0 = disabled)
    y_compensation_ratio: float = 0.0  # Y movement per Z movement (typically 0.0, set as needed)

    # For 1D Z-series: fixed position parameters
    fixed_x_position: float = 5000.0  # nm - fixed X position for Z scan
    fixed_y_position: float = 5000.0  # nm - fixed Y position for Z scan
    z_dwell_time: float = 1.0  # seconds - dwell time at each Z position

    @property
    def z_step(self) -> float:
        """Actual step size used"""
        return self.z_step_input

    @property
    def z_effective_range(self) -> float:
        """Effective range in Z direction"""
        return (self.z_numbers - 1) * self.z_step if self.z_numbers > 1 else 0

    @property
    def z_positions(self) -> List[float]:
        """Calculate all Z positions for the series"""
        if not self.enabled or self.z_numbers <= 1:
            return [self.z_start]

        z_forward = self.z_end >= self.z_start
        z_step = self.z_step if z_forward else -self.z_step
        return [self.z_start + i * z_step for i in range(self.z_numbers)]

@dataclass
class RSeriesParameters:
    """R-series (rotation) configuration for both 2D and 1D scans"""
    enabled: bool = False
    r_start: float = 0.0  # μdeg
    r_end: float = 360000000.0  # μdeg (360 degrees)
    r_numbers: int = 10
    r_step_input: float = 40000000.0  # μdeg

    base_r_position: float = 0.0  # μdeg - base offset for entire R-series

    # Center of Rotation for coordinate transformation
    cor_enabled: bool = False
    cor_x: float = 0.0  # nm - X coordinate of rotation center
    cor_y: float = 0.0  # nm - Y coordinate of rotation center
    cor_base_z: float = 0.0  # nm - Z height where COR coordinates are valid
    cor_x_compensation_ratio: float = 1.0  # X movement per Z movement (1.0 = 45° beam, 0.0 = disabled)
    cor_y_compensation_ratio: float = 0.0  # Y movement per Z movement (typically 0.0 for pure X-Z tilt)

    # Transformation mode
    cor_mode: str = "center_rotate"

    # For 1D R-series
    r_dwell_time: float = 1.0  # seconds - dwell time at each R position

    # Combined with Z-series for 1D R+Z scan
    combine_with_z: bool = False  # Enable R+Z combined scan

    # Settling tolerances (NEW)
    r_tol_udeg: float = 1000.0  # μdeg tolerance (0.001°)
    xy_tol_nm: float = 5.0  # nm tolerance for X/Y after COR transform
    settle_required_samples: int = 3
    settle_timeout_s: float = 10.0

    @property
    def r_step(self) -> float:
        """Actual step size used"""
        return self.r_step_input

    @property
    def r_effective_range(self) -> float:
        """Effective range in R direction"""
        return (self.r_numbers - 1) * self.r_step if self.r_numbers > 1 else 0

    @property
    def r_positions(self) -> List[float]:
        """Calculate all R positions for the series (r_start to r_end)"""
        if not self.enabled or self.r_numbers <= 1:
            return [self.r_start]

        r_forward = self.r_end >= self.r_start
        r_step = self.r_step if r_forward else -self.r_step
        # Simple formula: r_start + i * r_step (no base offset)
        return [self.r_start + i * r_step for i in range(self.r_numbers)]

    def transform_coordinates(self, x: float, y: float, r_angle: float,
                              current_z: float = None) -> Tuple[float, float]:
        """Transform X,Y coordinates based on rotation around COR with Z compensation

        The rotation is relative to base_r_position (similar to Z-series X compensation)

        Args:
            x, y: Original coordinates in nm
            r_angle: Absolute rotation angle in µdeg
            current_z: Current Z position in nm (for COR Z compensation)

        Returns:
            Transformed (x, y) coordinates
        """
        if not self.cor_enabled:
            return x, y

            # Calculate Z-compensated COR position
        z_offset = (current_z - self.cor_base_z) if current_z is not None else 0.0

        # X compensation (enabled if ratio != 0)
        if abs(self.cor_x_compensation_ratio) > 1e-9:
            cor_x_actual = self.cor_x + z_offset * self.cor_x_compensation_ratio
        else:
            cor_x_actual = self.cor_x

        # Y compensation (enabled if ratio != 0)
        if abs(self.cor_y_compensation_ratio) > 1e-9:
            cor_y_actual = self.cor_y + z_offset * self.cor_y_compensation_ratio
        else:
            cor_y_actual = self.cor_y

        # Calculate rotation RELATIVE to base position (like Z-series does)
        r_offset = r_angle - self.base_r_position

        # Convert µdeg to radians
        theta = np.radians(r_offset / 1e6)

        # Translate to Z-compensated COR origin
        x_rel = x - cor_x_actual
        y_rel = y - cor_y_actual

        # Apply rotation matrix (clockwise)
        x_new = x_rel * np.cos(theta) + y_rel * np.sin(theta)
        y_new = -x_rel * np.sin(theta) + y_rel * np.cos(theta)

        # Translate back
        return x_new + cor_x_actual, y_new + cor_y_actual

    def transform_scan_area(self, x_start: float, x_end: float,
                            y_start: float, y_end: float,
                            r_angle: float, current_z: float = None) -> Tuple[float, float, float, float]:
        """Transform entire scan area based on rotation relative to base_r_position

        Returns:
            Transformed (x_start, x_end, y_start, y_end)
        """
        if not self.cor_enabled:
            return x_start, x_end, y_start, y_end

        # Transform all four corners
        corners = [
            (x_start, y_start),
            (x_start, y_end),
            (x_end, y_start),
            (x_end, y_end)
        ]

        transformed_corners = [self.transform_coordinates(x, y, r_angle, current_z)
                               for x, y in corners]

        # Find new bounding box
        x_coords = [x for x, y in transformed_corners]
        y_coords = [y for x, y in transformed_corners]

        return min(x_coords), max(x_coords), min(y_coords), max(y_coords)

    def transform_scan_area_center_rotate(self, x_start: float, x_end: float,
                                         y_start: float, y_end: float,
                                         r_angle: float, current_z: float = None) -> Tuple[float, float, float, float]:
        """Transform scan area by rotating center point only, keeping width/height fixed"""
        if not self.cor_enabled:
            return x_start, x_end, y_start, y_end

        # Calculate current center and size
        x_center = (x_start + x_end) / 2
        y_center = (y_start + y_end) / 2
        width = abs(x_end - x_start)
        height = abs(y_end - y_start)

        # Rotate center point around COR
        x_center_new, y_center_new = self.transform_coordinates(x_center, y_center, r_angle, current_z)

        # Rebuild scan area around rotated center with same size
        return (x_center_new - width / 2, x_center_new + width / 2,
                y_center_new - height / 2, y_center_new + height / 2)

@dataclass
class ScanParameters:
    """Scan configuration parameters"""
    x_start: float = 0.0  # nm
    x_end: float = 10000.0  # nm
    y_start: float = 0.0  # nm
    y_end: float = 10000.0  # nm
    x_pixels: int = 100
    y_pixels: int = 100
    x_step_input: float = 101.01  # User-specified step size
    y_step_input: float = 101.01  # User-specified step size
    mode: ScanMode = ScanMode.STEP_STOP
    dwell_time: float = 0.75  # seconds (for step-stop)
    scan_speed: float = 1000.0  # nm/s (for continuous)
    pattern: str = "snake"  # "raster" or "snake"
    base_z_position: float = 0.0  # Initial Z position for non-series scans
    base_r_position: float = 0.0  # Initial R position for non-series scans
    z_series: ZSeriesParameters = field(default_factory=ZSeriesParameters)
    r_series: RSeriesParameters = field(default_factory=RSeriesParameters)

    # --- in-position gating (defaults) ---
    pos_tol_x_nm: float = 5  # allowable X error to accept as "on target"
    pos_tol_y_nm: float = 5  # allowable Y error
    pos_tol_z_nm: float = 5  # allowable Z error for Z-series
    settle_required_samples: int = 3  # consecutive fresh samples within tolerance
    settle_timeout_s: float = 5.0  # give up waiting after this time (seconds)

    @property
    def x_step(self) -> float:
        """Actual step size used (same as input in underscan mode)"""
        return self.x_step_input

    @property
    def y_step(self) -> float:
        """Actual step size used (same as input in underscan mode)"""
        return self.y_step_input

    @property
    def x_effective_fov(self) -> float:
        """Effective FOV in X direction"""
        return (self.x_pixels - 1) * self.x_step if self.x_pixels > 1 else 0

    @property
    def y_effective_fov(self) -> float:
        """Effective FOV in Y direction"""
        return (self.y_pixels - 1) * self.y_step if self.y_pixels > 1 else 0

    @property
    def total_pixels(self) -> int:
        return self.x_pixels * self.y_pixels

    @property
    def total_images(self) -> int:
        """Total number of images in Z-series (1 if Z-series disabled)"""
        return self.z_series.z_numbers if self.z_series.enabled else 1

@dataclass
class LineScanParameters:
    """Line scan configuration parameters"""
    fixed_axis: str = "X"  # "X" or "Y" - which axis is fixed
    fixed_position: float = 0.0  # Position of fixed axis in nm
    scan_axis: str = "Y"  # "X" or "Y" - which axis scans
    scan_start: float = 0.0  # nm
    scan_end: float = 10000.0  # nm
    num_points: int = 100
    step_size: float = 101.01  # nm
    dwell_time: float = 0.1  # seconds
    base_z_position: float = 0.0  # Initial Z position for non-series scans
    base_r_position: float = 0.0  # Initial R position for non-series scans
    z_series: ZSeriesParameters = field(default_factory=ZSeriesParameters)
    r_series: RSeriesParameters = field(default_factory=RSeriesParameters)

    # In-position gating (same as 2D scan)
    pos_tolerance_nm: float = 5.0
    pos_tol_z_nm: float = 5.0  # Z tolerance for Z-series
    settle_required_samples: int = 3
    settle_timeout_s: float = 5.0

    @property
    def scan_range(self) -> float:
        return abs(self.scan_end - self.scan_start)

    @property
    def effective_range(self) -> float:
        return (self.num_points - 1) * self.step_size if self.num_points > 1 else 0

    @property
    def total_measurements(self) -> int:
        """Total number of measurements: lines for regular scan, Z-points for Z-series"""
        if self.z_series.enabled:
            return self.z_series.z_numbers  # Z-series: one measurement per Z position
        else:
            return self.num_points  # Regular line scan: one measurement per point

    @property
    def is_z_series_mode(self) -> bool:
        """Check if this is a Z-series scan (different from regular line scan)"""
        return self.z_series.enabled

class LineScanController(QObject):
    """Controls line scan execution - simplified to use existing data pipeline"""

    # Signals
    scan_started = pyqtSignal()
    scan_completed = pyqtSignal()
    scan_progress = pyqtSignal(int, int)  # current_point, total_points
    movement_command = pyqtSignal(str)
    status_update = pyqtSignal(str)
    current_position_index = pyqtSignal(int)  # Emit current point index for display

    def __init__(self):
        super().__init__()
        self.line_params = None
        self.scan_state = ScanState.IDLE
        self.current_point = 0
        self.scan_positions = []

        #Add line_reconstructor attribute (will be set by MainWindow)
        self.line_reconstructor = None

        # Timers (same as ScanController)
        self.step_timer = QTimer()
        self.step_timer.setSingleShot(True)
        self.step_timer.timeout.connect(self.on_dwell_completed)

        self.settle_timer = QTimer()
        self.settle_timer.timeout.connect(self._on_settle_check)
        self._waiting_to_settle = False
        self._settle_start_time = 0.0
        self._settle_consecutive = 0
        self._current_target_pos = None
        self._data_processor = None

        self._detector_lag_timer = QTimer()
        self._detector_lag_timer.setSingleShot(True)
        self._detector_lag_timer.timeout.connect(self._begin_dwell_after_detector_lag)
        self._in_detector_lag = False
        self._detector_lag_s = 0.350  # Default, will be updated from UI

        # Initialization
        self._initializing = False
        self.init_timer = QTimer()
        self.init_timer.timeout.connect(self._check_init_position)

    def set_data_processor(self, dp):
        """Set reference to data processor for position monitoring"""
        self._data_processor = dp

    def set_parameters(self, params: LineScanParameters):
        """Set line scan parameters"""
        self.line_params = params
        self.generate_scan_positions()

    def generate_scan_positions(self):
        """Generate list of scan positions"""
        if not self.line_params:
            return

        self.scan_positions = []

        # Determine scan direction
        forward = self.line_params.scan_end >= self.line_params.scan_start
        step = self.line_params.step_size if forward else -self.line_params.step_size

        for i in range(self.line_params.num_points):
            pos = self.line_params.scan_start + i * step
            self.scan_positions.append(pos)

    def start_scan(self):
        """Start line scan"""
        if not self.line_params:
            self.status_update.emit("ERROR: No line scan parameters set")
            return

        self.scan_state = ScanState.SCANNING
        self.current_point = 0
        self.scan_started.emit()

        # Move to start position first
        self._move_to_start_position()

    def _move_to_start_position(self):
        """Move to initial scan position"""
        self._initializing = True
        self._settle_start_time = time.monotonic()

        # Move fixed axis to position
        fixed_axis = self.line_params.fixed_axis
        fixed_pos = self.line_params.fixed_position
        self.movement_command.emit(f"MOVE/{fixed_axis}/{fixed_pos:.0f}")

        # Move scan axis to start
        scan_axis = self.line_params.scan_axis
        start_pos = self.line_params.scan_start
        self.movement_command.emit(f"MOVE/{scan_axis}/{start_pos:.0f}")

        self.status_update.emit(f"Moving to start: {fixed_axis}={fixed_pos:.0f}, {scan_axis}={start_pos:.0f}")

        # Start checking position
        self.init_timer.start(50)

    def _check_init_position(self):
        """Check if we've reached start position"""
        if not self._initializing or not self._data_processor:
            self.init_timer.stop()
            return

        x, y, z, r, tx, ty, tz, tr, now = self._data_processor.get_position_snapshot()

        if x is None or y is None:
            return

        # Check freshness
        if abs(now - tx) > 0.5 or abs(now - ty) > 0.5:
            return

        # Get target positions
        if self.line_params.fixed_axis == "X":
            target_x = self.line_params.fixed_position
            target_y = self.line_params.scan_start
        else:
            target_x = self.line_params.scan_start
            target_y = self.line_params.fixed_position

        dx = abs(x - target_x)
        dy = abs(y - target_y)
        tol = self.line_params.pos_tolerance_nm

        if dx <= tol and dy <= tol:
            # Reached start position
            self.init_timer.stop()
            self._initializing = False
            self.status_update.emit("At start position. Beginning line scan...")
            self.execute_next_point()
            return

        # Check timeout
        if time.monotonic() - self._settle_start_time > 30:
            self.init_timer.stop()
            self._initializing = False
            self.status_update.emit("ERROR: Timeout reaching start position")
            self.complete_scan()

    def execute_next_point(self):
        """Move to next scan point"""
        if self.scan_state != ScanState.SCANNING or self.current_point >= len(self.scan_positions):
            self.complete_scan()
            return

        # Get next position
        scan_pos = self.scan_positions[self.current_point]

        # NEW: Set current point index in line reconstructor (similar to 2D scan)
        if self.line_reconstructor:
            self.line_reconstructor.set_current_point(self.current_point)

        # Emit current position index for display tracking
        self.current_position_index.emit(self.current_point)

        # Send movement command
        scan_axis = self.line_params.scan_axis
        self.movement_command.emit(f"MOVE/{scan_axis}/{scan_pos:.0f}")

        self.scan_progress.emit(self.current_point + 1, len(self.scan_positions))

        # Start settle wait
        self._current_target_pos = scan_pos
        self._start_settle_wait()

    def _start_settle_wait(self):
        """Start waiting for position to settle"""
        # Reset any detector lag state from previous point
        self._in_detector_lag = False
        self._detector_lag_timer.stop()
        self._waiting_to_settle = True
        self._settle_consecutive = 0
        self._settle_start_time = time.monotonic()
        self.settle_timer.start(20)  # 50Hz check

    def set_detector_lag(self, lag_seconds: float):
        """Set detector lag time in seconds"""
        self._detector_lag_s = lag_seconds

    def _start_detector_lag_wait(self):
        """Wait for detector lag before starting real dwell."""
        # Fast path for zero lag
        if self._detector_lag_s <= 0:
            self._start_dwell()
            return

        self._in_detector_lag = True
        self.status_update.emit(f"Waiting {self._detector_lag_s:.3f}s for detector stabilization...")
        self._detector_lag_timer.start(int(self._detector_lag_s * 1000))

    def _begin_dwell_after_detector_lag(self):
        """Begin actual dwell after detector lag period."""
        self._in_detector_lag = False
        self._start_dwell()  # Existing method

    def _on_settle_check(self):
        """Check if position has settled"""
        if not self._waiting_to_settle or not self._data_processor:
            self.settle_timer.stop()
            return

        x, y, z, r, tx, ty, tz, tr, now = self._data_processor.get_position_snapshot()

        # Get current position of scan axis
        if self.line_params.scan_axis == "X":
            current_pos = x
            timestamp = tx
        else:
            current_pos = y
            timestamp = ty

        if current_pos is None or timestamp is None:
            return

        # Check freshness
        if abs(now - timestamp) > 0.3:
            return

        # Check if in tolerance
        error = abs(current_pos - self._current_target_pos)

        if error <= self.line_params.pos_tolerance_nm:
            self._settle_consecutive += 1
        else:
            self._settle_consecutive = 0

        # Check if settled
        if self._settle_consecutive >= self.line_params.settle_required_samples:
            self.settle_timer.stop()
            self._waiting_to_settle = False
            self._start_detector_lag_wait()  # Wait for detector first
            return

        # Check timeout
        if time.monotonic() - self._settle_start_time >= self.line_params.settle_timeout_s:
            self.settle_timer.stop()
            self._waiting_to_settle = False
            self.status_update.emit(f"Settle timeout at point {self.current_point}")
            self._start_detector_lag_wait()  # Wait for detector first

    def _start_dwell(self):
        """Start dwell time acquisition"""
        dwell_ms = int(self.line_params.dwell_time * 1000)
        self.step_timer.start(dwell_ms)

    def on_dwell_completed(self):
        """Called when dwell is complete"""
        self.current_point += 1
        self.execute_next_point()

    def stop_scan(self):
        """Stop line scan"""
        self.scan_state = ScanState.IDLE
        self.settle_timer.stop()
        self.step_timer.stop()
        self.init_timer.stop()
        self._detector_lag_timer.stop()
        self._in_detector_lag = False

        # Send stop commands
        self.movement_command.emit(f"STOP/{self.line_params.scan_axis}")
        self.movement_command.emit(f"STOP/{self.line_params.fixed_axis}")

        self.complete_scan()

    def complete_scan(self):
        """Complete the line scan"""
        self.scan_state = ScanState.IDLE

        # NEW: Clear current point in reconstructor
        if self.line_reconstructor:
            self.line_reconstructor.clear_current_point()
            self.line_reconstructor.complete_scan()

        self.scan_completed.emit()
        self.status_update.emit("Line scan completed")

class ZSeriesController(QObject):
    """Controls Z-series scans by coordinating Z movements with existing 2D/1D controllers"""

    # Signals
    z_series_started = pyqtSignal()
    z_series_completed = pyqtSignal()
    z_slice_started = pyqtSignal(int, float)  # z_index, z_position
    z_slice_completed = pyqtSignal(int)  # z_index
    movement_command = pyqtSignal(str)
    status_update = pyqtSignal(str)
    z_series_progress = pyqtSignal(int, int)  # current_z, total_z

    def __init__(self):
        super().__init__()
        self.z_params = None
        self.scan_type = None  # "2D" or "1D"
        self.current_z_index = 0
        self.base_x_position = 0.0  # Store initial X for compensation
        self.inner_scan_controller = None  # Reference to 2D or 1D controller
        self.inner_scan_params = None  # Store original scan parameters
        self.data_storage = None
        self.data_processor = None
        self.z_series_state = ScanState.IDLE

        # Z movement and settling
        self.z_settle_timer = QTimer()
        self.z_settle_timer.timeout.connect(self._on_z_settle_check)
        self._z_waiting_to_settle = False
        self._z_settle_start_time = 0.0
        self._z_settle_consecutive = 0
        self._z_target_position = None
        self._z_compensated_x = None

    def set_data_processor(self, dp):
        """Set reference to data processor for position monitoring"""
        self.data_processor = dp

    def set_data_storage(self, storage):
        """Set reference to data storage"""
        self.data_storage = storage

    def start_z_series(self, scan_type: str, inner_controller, scan_params):
        """Start Z-series scan"""
        self.scan_type = scan_type
        self.inner_scan_controller = inner_controller
        self.inner_scan_params = scan_params
        self.current_z_index = 0
        self.z_series_state = ScanState.SCANNING

        # Get Z-series parameters
        if scan_type == "2D":
            self.z_params = scan_params.z_series
            self.base_x_position = scan_params.x_start
            self.base_y_position = scan_params.y_start  # ADD THIS LINE
        else:  # "1D"
            self.z_params = scan_params.z_series
            # For line scans, use the scan position or fixed position as base X
            if scan_params.scan_axis == "X":
                self.base_x_position = scan_params.scan_start
            else:
                self.base_x_position = scan_params.fixed_position

            # Initialize base_y_position for line scans
            if scan_params.scan_axis == "Y":
                self.base_y_position = scan_params.scan_start
            else:
                self.base_y_position = scan_params.fixed_position

        self.z_series_started.emit()
        self.status_update.emit(f"Starting Z-series: {self.z_params.z_numbers} slices")

        # Move to first Z position
        self._move_to_next_z_slice()

    def _move_to_next_z_slice(self):
        """Move to next Z position and start inner scan"""
        if self.current_z_index >= len(self.z_params.z_positions):
            self.complete_z_series()
            return

        z_pos = self.z_params.z_positions[self.current_z_index]

        # Calculate X and Y compensation for tilted beam
        z_offset = z_pos - self.inner_scan_params.base_z_position

        # X compensation (enabled if ratio != 0)
        if abs(self.z_params.x_compensation_ratio) > 1e-9:
            x_offset = z_offset * self.z_params.x_compensation_ratio
            if self.scan_type == "2D":
                self._z_compensated_x = self.inner_scan_params.x_start + x_offset
            else:  # "1D"
                if self.inner_scan_params.scan_axis == "X":
                    self._z_compensated_x = self.inner_scan_params.scan_start + x_offset
                else:
                    self._z_compensated_x = self.inner_scan_params.fixed_position + x_offset
        else:
            self._z_compensated_x = None

        # Y compensation (enabled if ratio != 0)
        if abs(self.z_params.y_compensation_ratio) > 1e-9:
            y_offset = z_offset * self.z_params.y_compensation_ratio
            if self.scan_type == "2D":
                self._z_compensated_y = self.inner_scan_params.y_start + y_offset
            else:  # "1D"
                if self.inner_scan_params.scan_axis == "Y":
                    self._z_compensated_y = self.inner_scan_params.scan_start + y_offset
                else:
                    self._z_compensated_y = self.inner_scan_params.fixed_position + y_offset
        else:
            self._z_compensated_y = None

        self.z_slice_started.emit(self.current_z_index, z_pos)
        self.z_series_progress.emit(self.current_z_index + 1, len(self.z_params.z_positions))

        self.status_update.emit(
            f"Moving to Z slice {self.current_z_index + 1}/{len(self.z_params.z_positions)} (Z={z_pos:.0f}nm)")

        # Send movement commands
        if self._z_compensated_x is not None:
            self.movement_command.emit(f"MOVE/X/{self._z_compensated_x:.0f}")
        if self._z_compensated_y is not None:
            self.movement_command.emit(f"MOVE/Y/{self._z_compensated_y:.0f}")

        self.movement_command.emit(f"MOVE/Z/{z_pos:.0f}")

        # Start settle wait
        self._z_target_position = z_pos
        self._start_z_settle_wait()

    def _start_z_settle_wait(self):
        """Begin waiting for Z position to settle"""
        self._z_waiting_to_settle = True
        self._z_settle_consecutive = 0
        self._z_settle_start_time = time.monotonic()
        self.z_settle_timer.start(50)  # 20Hz check

    def _on_z_settle_check(self):
        """Check if Z position has settled"""
        if not self._z_waiting_to_settle or not self.data_processor:
            self.z_settle_timer.stop()
            return

        x, y, z, r, tx, ty, tz, tr, now = self.data_processor.get_position_snapshot()

        if z is None or tz is None:
            return

        # Check freshness
        if abs(now - tz) > 0.5:
            return

        # Check Z tolerance
        z_error = abs(z - self._z_target_position)

        # Also check X compensation if enabled
        x_ok = True
        if self._z_compensated_x is not None and x is not None and tx is not None:
            if abs(now - tx) < 0.5:  # X data is fresh
                x_error = abs(x - self._z_compensated_x)
                x_ok = x_error <= self.inner_scan_params.pos_tol_x_nm

        # Also check Y compensation if enabled
        y_ok = True
        if self._z_compensated_y is not None and y is not None and ty is not None:
            if abs(now - ty) < 0.5:  # Y data is fresh
                y_error = abs(y - self._z_compensated_y)
                y_ok = y_error <= self.inner_scan_params.pos_tol_y_nm

        if z_error <= getattr(self.inner_scan_params, 'pos_tol_z_nm', 5.0) and x_ok and y_ok:
            self._z_settle_consecutive += 1
        else:
            self._z_settle_consecutive = 0

        # Check if settled
        if self._z_settle_consecutive >= getattr(self.inner_scan_params, 'settle_required_samples', 3):
            self.z_settle_timer.stop()
            self._z_waiting_to_settle = False
            self._start_inner_scan()
            return

        # Check timeout
        if time.monotonic() - self._z_settle_start_time >= getattr(self.inner_scan_params, 'settle_timeout_s', 10.0):
            self.z_settle_timer.stop()
            self._z_waiting_to_settle = False
            self.status_update.emit(f"Z settle timeout (ΔZ≈{z_error:.0f}nm). Proceeding.")
            self._start_inner_scan()

    def _start_inner_scan(self):
        """Start the inner 2D or 1D scan at current Z position"""
        z_pos = self.z_params.z_positions[self.current_z_index]

        # Update scan parameters with current Z position
        if self.scan_type == "2D":
            # Create a copy of the original parameters to avoid modifying the original
            import copy
            updated_params = copy.deepcopy(self.inner_scan_params)

            # Update Z position
            updated_params.base_z_position = z_pos

            # Update X start position for beam compensation
            if abs(self.z_params.x_compensation_ratio) > 1e-9 and self._z_compensated_x is not None:
                # Calculate the X offset from the original base position
                x_offset = self._z_compensated_x - self.base_x_position

                # Apply offset to both start and end positions
                updated_params.x_start = self.inner_scan_params.x_start + x_offset
                updated_params.x_end = self.inner_scan_params.x_end + x_offset

            # --- ALSO apply Y offset when Y-comp is active ---
            if abs(self.z_params.y_compensation_ratio) > 1e-9 and self._z_compensated_y is not None:
                # Calculate the Y offset from the original base position
                y_offset = self._z_compensated_y - self.base_y_position

                # Apply offset to both start and end positions
                updated_params.y_start = self.inner_scan_params.y_start + y_offset
                updated_params.y_end = self.inner_scan_params.y_end + y_offset

            self.inner_scan_controller.set_scan_parameters(updated_params)

        else:  # "1D"
            # Create a copy for line scan parameters
            import copy
            updated_params = copy.deepcopy(self.inner_scan_params)
            updated_params.base_z_position = z_pos

            if abs(self.z_params.x_compensation_ratio) > 1e-9 and self._z_compensated_x is not None:
                x_offset = self._z_compensated_x - self.base_x_position

                if updated_params.scan_axis == "X":
                    # Compensate scan positions
                    updated_params.scan_start = self.inner_scan_params.scan_start + x_offset
                    updated_params.scan_end = self.inner_scan_params.scan_end + x_offset
                else:
                    # Compensate fixed position
                    updated_params.fixed_position = self.inner_scan_params.fixed_position + x_offset

                # --- ALSO apply Y offset when Y-comp is active ---
            if abs(self.z_params.y_compensation_ratio) > 1e-9 and self._z_compensated_y is not None:
                y_offset = self._z_compensated_y - self.base_y_position

                if updated_params.scan_axis == "Y":
                    # Compensate scan positions
                    updated_params.scan_start = self.inner_scan_params.scan_start + y_offset
                    updated_params.scan_end = self.inner_scan_params.scan_end + y_offset
                else:
                    # Compensate fixed position
                    updated_params.fixed_position = self.inner_scan_params.fixed_position + y_offset

            self.inner_scan_controller.set_parameters(updated_params)

        self.status_update.emit(f"Starting inner {self.scan_type} scan at Z={z_pos:.0f}nm")

        # Connect to inner scan completion
        self.inner_scan_controller.scan_completed.connect(self._on_inner_scan_completed)

        # Start inner scan
        self.inner_scan_controller.start_scan()

    def _on_inner_scan_completed(self):
        """Handle completion of inner scan"""
        # Disconnect completion signal
        try:
            self.inner_scan_controller.scan_completed.disconnect(self._on_inner_scan_completed)
        except:
            pass

        z_pos = self.z_params.z_positions[self.current_z_index]
        self.z_slice_completed.emit(self.current_z_index)
        self.status_update.emit(f"Completed Z slice {self.current_z_index + 1} at Z={z_pos:.0f}nm")

        # Move to next Z slice
        self.current_z_index += 1
        self._move_to_next_z_slice()

    def stop_z_series(self):
        """Stop Z-series scan"""
        self.z_series_state = ScanState.IDLE
        self.z_settle_timer.stop()

        # Stop inner scan if running
        if self.inner_scan_controller:
            if hasattr(self.inner_scan_controller, 'stop_scan'):
                self.inner_scan_controller.stop_scan()

        self.complete_z_series()

    def complete_z_series(self):
        """Complete Z-series scan"""
        self.z_series_state = ScanState.IDLE
        self.z_settle_timer.stop()

        # Disconnect any remaining signals
        try:
            if self.inner_scan_controller:
                self.inner_scan_controller.scan_completed.disconnect(self._on_inner_scan_completed)
        except:
            pass

        self.z_series_completed.emit()
        self.status_update.emit(f"Z-series completed: {self.current_z_index} slices")

class RSeriesController(QObject):
    """Controls R-series scans by coordinating R movements with existing 2D/1D controllers"""

    # Signals
    r_series_started = pyqtSignal()
    r_series_completed = pyqtSignal()
    r_slice_started = pyqtSignal(int, float)  # r_index, r_position
    r_slice_completed = pyqtSignal(int)  # r_index
    movement_command = pyqtSignal(str)
    status_update = pyqtSignal(str)
    r_series_progress = pyqtSignal(int, int)  # current_r, total_r

    def __init__(self):
        super().__init__()
        self.r_params = None
        self.scan_type = None  # "2D" or "1D"
        self.current_r_index = 0
        self.inner_scan_controller = None
        self.inner_scan_params = None
        self.original_scan_area = None  # Store original scan area for 2D
        self.data_storage = None
        self.data_processor = None
        self.r_series_state = ScanState.IDLE

        # R movement and settling
        self.r_settle_timer = QTimer()
        self.r_settle_timer.timeout.connect(self._on_r_settle_check)
        self._r_waiting_to_settle = False
        self._r_settle_start_time = 0.0
        self._r_settle_consecutive = 0
        self._r_target_position = None
        self._r_transformed_x = None
        self._r_transformed_y = None

    def set_data_processor(self, dp):
        """Set reference to data processor for position monitoring"""
        self.data_processor = dp

    def set_data_storage(self, storage):
        """Set reference to data storage"""
        self.data_storage = storage

    def start_r_series(self, scan_type: str, inner_controller, scan_params):
        """Start R-series scan

        Args:
            scan_type: "2D" or "1D"
            inner_controller: The 2D or 1D scan controller
            scan_params: ScanParameters or LineScanParameters
        """
        self.scan_type = scan_type
        self.inner_scan_controller = inner_controller
        self.inner_scan_params = scan_params
        self.current_r_index = 0
        self.r_series_state = ScanState.SCANNING

        # Get R-series parameters
        self.r_params = scan_params.r_series

        # Store original scan area for 2D
        if scan_type == "2D":
            self.original_scan_area = (
                scan_params.x_start, scan_params.x_end,
                scan_params.y_start, scan_params.y_end
            )

        self.r_series_started.emit()
        self.status_update.emit(f"Starting R-series: {self.r_params.r_numbers} angles")

        # Move to first R position
        self._move_to_next_r_slice()

    def _move_to_next_r_slice(self):
        """Move to next R position and start inner scan"""
        if self.current_r_index >= len(self.r_params.r_positions):
            self.complete_r_series()
            return

        r_pos = self.r_params.r_positions[self.current_r_index]

        # Calculate coordinate transformations if COR is enabled
        if self.r_params.cor_enabled:
            if self.scan_type == "2D":
                # Transform the scan area
                x_start, x_end, y_start, y_end = self.r_params.transform_scan_area(
                    *self.original_scan_area, r_pos
                )
                self._r_transformed_x = x_start  # Store for settle check
                self._r_transformed_y = y_start
            else:  # "1D"
                # Transform fixed position for line scan
                if hasattr(self.inner_scan_params, 'fixed_position'):
                    if self.inner_scan_params.fixed_axis == "X":
                        fixed_x = self.inner_scan_params.fixed_position
                        scan_y = self.inner_scan_params.scan_start
                    else:
                        fixed_x = self.inner_scan_params.scan_start
                        scan_y = self.inner_scan_params.fixed_position

                    self._r_transformed_x, self._r_transformed_y = \
                        self.r_params.transform_coordinates(fixed_x, scan_y, r_pos)

        self.r_slice_started.emit(self.current_r_index, r_pos)
        self.r_series_progress.emit(self.current_r_index + 1, len(self.r_params.r_positions))

        self.status_update.emit(
            f"Moving to R slice {self.current_r_index + 1}/{len(self.r_params.r_positions)} "
            f"(R={r_pos:.0f}μdeg)")

        # Send movement command
        self.movement_command.emit(f"MOVE/R/{r_pos:.0f}")

        # If COR transformation, also move X/Y
        if self.r_params.cor_enabled and self._r_transformed_x is not None:
            self.movement_command.emit(f"MOVE/X/{self._r_transformed_x:.0f}")
            self.movement_command.emit(f"MOVE/Y/{self._r_transformed_y:.0f}")

        # Start settle wait
        self._r_target_position = r_pos
        self._start_r_settle_wait()

    def _start_r_settle_wait(self):
        """Begin waiting for R position to settle"""
        self._r_waiting_to_settle = True
        self._r_settle_consecutive = 0
        self._r_settle_start_time = time.monotonic()
        self.r_settle_timer.start(50)  # 20Hz check

    def _on_r_settle_check(self):
        """Check if R position has settled"""
        if not self._r_waiting_to_settle or not self.data_processor:
            self.r_settle_timer.stop()
            return

        x, y, z, r, tx, ty, tz, tr, now = self.data_processor.get_position_snapshot()

        if r is None or tr is None:
            return

        # Check freshness
        if abs(now - tr) > 0.5:
            return

        # Check R tolerance
        r_error = abs(r - self._r_target_position)
        r_ok = r_error <= 1000  # 1000 μdeg tolerance

        # Check X/Y if COR transformation was applied
        xy_ok = True
        if self._r_transformed_x is not None and x is not None and tx is not None:
            if abs(now - tx) < 0.5 and abs(now - ty) < 0.5:
                x_error = abs(x - self._r_transformed_x)
                y_error = abs(y - self._r_transformed_y)
                xy_ok = x_error <= 5.0 and y_error <= 5.0  # nm tolerance

        if r_ok and xy_ok:
            self._r_settle_consecutive += 1
        else:
            self._r_settle_consecutive = 0

        # Check if settled
        if self._r_settle_consecutive >= 3:
            self.r_settle_timer.stop()
            self._r_waiting_to_settle = False
            self._start_inner_scan()
            return

        # Check timeout
        if time.monotonic() - self._r_settle_start_time >= 10.0:
            self.r_settle_timer.stop()
            self._r_waiting_to_settle = False
            self.status_update.emit(f"R settle timeout. Proceeding.")
            self._start_inner_scan()

    def _start_inner_scan(self):
        """Start the inner 2D or 1D scan at current R position"""
        r_pos = self.r_params.r_positions[self.current_r_index]

        if self.scan_type == "2D":
            # Create a copy of the original parameters to avoid modifying the original
            import copy
            updated_params = copy.deepcopy(self.inner_scan_params)

            # Update R position
            updated_params.base_r_position = r_pos

            if self.r_params.cor_enabled:
                # Get current Z position for COR compensation
                if self.data_processor:
                    x_snap, y_snap, z_snap, r_snap, *_ = self.data_processor.get_position_snapshot()
                    current_z = z_snap if z_snap is not None else self.inner_scan_params.base_z_position
                else:
                    current_z = self.inner_scan_params.base_z_position

                # Choose transformation mode
                mode = getattr(self.r_params, 'cor_mode', 'center_rotate')

                if mode == "center_rotate":
                    # New approach: rotate center, keep size fixed
                    x_start, x_end, y_start, y_end = self.r_params.transform_scan_area_center_rotate(
                        *self.original_scan_area, r_pos, current_z  # Pass current Z
                    )
                else:  # mode == "aabb" (legacy)
                    # Old approach: transform all corners and find bounding box
                    x_start, x_end, y_start, y_end = self.r_params.transform_scan_area(
                        *self.original_scan_area, r_pos, current_z  # Pass current Z
                    )

                updated_params.x_start = x_start
                updated_params.x_end = x_end
                updated_params.y_start = y_start
                updated_params.y_end = y_end

            self.inner_scan_controller.set_scan_parameters(updated_params)
            # IMPORTANT: Also update the image reconstructor with new coordinates
            if hasattr(self.inner_scan_controller, 'image_reconstructor'):
                self.inner_scan_controller.image_reconstructor.initialize_scan(updated_params)

            self.status_update.emit(f"Starting inner 2D scan at R={r_pos:.0f}μdeg")

            # Connect to inner scan completion
            self.inner_scan_controller.scan_completed.connect(self._on_inner_scan_completed)

            # Start inner scan
            self.inner_scan_controller.start_scan()


        elif self.scan_type == "RZ":
            # Handle R+Z series - trigger Z-scan at transformed position
            import copy
            # Get Z-series parameters from the inner params
            z_series = self.inner_scan_params.z_series
            # Transform the fixed X,Y positions using COR
            fixed_x = z_series.fixed_x_position
            fixed_y = z_series.fixed_y_position
            if self.r_params.cor_enabled:
                # For RZ scans, COR will be compensated at each Z step by Z-scan controller
                # Use base Z for initial positioning
                base_z = self.inner_scan_params.base_z_position
                transformed_x, transformed_y = self.r_params.transform_coordinates(
                    fixed_x, fixed_y, r_pos, base_z  # Use base Z
                )
            else:
                transformed_x, transformed_y = fixed_x, fixed_y

            # Create updated Z-series parameters with transformed positions
            updated_z_series = copy.deepcopy(z_series)
            updated_z_series.fixed_x_position = transformed_x
            updated_z_series.fixed_y_position = transformed_y
            self.status_update.emit(
                f"Starting Z-scan at R={r_pos:.0f}μdeg, "
                f"X={transformed_x:.0f}nm, Y={transformed_y:.0f}nm"
            )

            # Set parameters and start Z-scan
            self.inner_scan_controller.set_parameters(
                updated_z_series,
                self.inner_scan_params.base_z_position
            )

            # Connect to Z-scan completion
            self.inner_scan_controller.scan_completed.connect(self._on_inner_scan_completed)

            # Start Z-scan
            self.inner_scan_controller.start_scan()

        else:  # Regular "1D" line scan (if needed)
            import copy
            updated_params = copy.deepcopy(self.inner_scan_params)
            updated_params.base_r_position = r_pos

            if self.r_params.cor_enabled and hasattr(updated_params, 'fixed_position'):
                if updated_params.fixed_axis == "X":
                    updated_params.fixed_position = self._r_transformed_x
                else:
                    updated_params.fixed_position = self._r_transformed_y

            self.inner_scan_controller.set_parameters(updated_params)
            self.status_update.emit(f"Starting inner 1D scan at R={r_pos:.0f}μdeg")

            # Connect and start
            self.inner_scan_controller.scan_completed.connect(self._on_inner_scan_completed)
            self.inner_scan_controller.start_scan()

    def _on_inner_scan_completed(self):
        """Handle completion of inner scan"""
        # Disconnect completion signal
        try:
            self.inner_scan_controller.scan_completed.disconnect(self._on_inner_scan_completed)
        except:
            pass

        r_pos = self.r_params.r_positions[self.current_r_index]
        self.r_slice_completed.emit(self.current_r_index)
        self.status_update.emit(f"Completed R slice {self.current_r_index + 1} at R={r_pos:.0f}μdeg")

        # Move to next R slice
        self.current_r_index += 1
        self._move_to_next_r_slice()

    def stop_r_series(self):
        """Stop R-series scan"""
        self.r_series_state = ScanState.IDLE
        self.r_settle_timer.stop()

        # Stop inner scan if running
        if self.inner_scan_controller:
            if hasattr(self.inner_scan_controller, 'stop_scan'):
                self.inner_scan_controller.stop_scan()

        self.complete_r_series()

    def complete_r_series(self):
        """Complete R-series scan"""
        self.r_series_state = ScanState.IDLE
        self.r_settle_timer.stop()

        # Disconnect any remaining signals
        try:
            if self.inner_scan_controller:
                self.inner_scan_controller.scan_completed.disconnect(self._on_inner_scan_completed)
        except:
            pass

        self.r_series_completed.emit()
        self.status_update.emit(f"R-series completed: {self.current_r_index} angles")

class ZScanController(QObject):
    """Controls 1D Z-series scans - moves Z (with X compensation) at fixed X,Y position"""

    # Signals (similar to LineScanController)
    scan_started = pyqtSignal()
    scan_completed = pyqtSignal()
    scan_progress = pyqtSignal(int, int)  # current_point, total_points
    movement_command = pyqtSignal(str)
    status_update = pyqtSignal(str)
    current_position_index = pyqtSignal(int)  # Emit current Z index for display

    def __init__(self):
        super().__init__()
        self.z_params = None
        self.scan_state = ScanState.IDLE
        self.current_z_index = 0
        self.z_positions = []
        self.base_x_position = 0.0
        self.base_y_position = 0.0

        # Use line_reconstructor for Z-scan data (Z positions vs signal)
        self.line_reconstructor = None

        # Timers (same pattern as other controllers)
        self.step_timer = QTimer()
        self.step_timer.setSingleShot(True)
        self.step_timer.timeout.connect(self.on_dwell_completed)

        self.settle_timer = QTimer()
        self.settle_timer.timeout.connect(self._on_settle_check)
        self._waiting_to_settle = False
        self._settle_start_time = 0.0
        self._settle_consecutive = 0
        self._current_target_z = None
        self._current_target_x = None
        self._data_processor = None

        # Detector lag timer
        self._detector_lag_timer = QTimer()
        self._detector_lag_timer.setSingleShot(True)
        self._detector_lag_timer.timeout.connect(self._begin_dwell_after_detector_lag)
        self._in_detector_lag = False
        self._detector_lag_s = 0.350  # Default

        # Initialization
        self._initializing = False
        self.init_timer = QTimer()
        self.init_timer.timeout.connect(self._check_init_position)

    def set_data_processor(self, dp):
        """Set reference to data processor for position monitoring"""
        self._data_processor = dp

    def set_parameters(self, z_params: ZSeriesParameters, base_z: float = 0.0):
        """Set Z-scan parameters"""
        self.z_params = z_params
        self.base_z_position = base_z
        self.generate_z_positions()

    def generate_z_positions(self):
        """Generate list of Z positions"""
        if not self.z_params:
            return
        self.z_positions = self.z_params.z_positions
        self.base_x_position = self.z_params.fixed_x_position
        self.base_y_position = self.z_params.fixed_y_position

    def start_scan(self):
        """Start Z-scan"""
        if not self.z_params:
            self.status_update.emit("ERROR: No Z-scan parameters set")
            return

        self.scan_state = ScanState.SCANNING
        self.current_z_index = 0
        self.scan_started.emit()

        # Move to start position first
        self._move_to_start_position()

    def _move_to_start_position(self):
        """Move to initial Z-scan position"""
        self._initializing = True
        self._settle_start_time = time.monotonic()

        # Move to fixed X,Y position
        self.movement_command.emit(f"MOVE/X/{self.z_params.fixed_x_position:.0f}")
        self.movement_command.emit(f"MOVE/Y/{self.z_params.fixed_y_position:.0f}")

        # Move to start Z position
        start_z = self.z_positions[0]
        self.movement_command.emit(f"MOVE/Z/{start_z:.0f}")

        self.status_update.emit(
            f"Moving to start: X={self.z_params.fixed_x_position:.0f}, Y={self.z_params.fixed_y_position:.0f}, Z={start_z:.0f}")

        # Start checking position
        self.init_timer.start(50)

    def _check_init_position(self):
        """Check if we've reached start position"""
        if not self._initializing or not self._data_processor:
            self.init_timer.stop()
            return

        x, y, z, r, tx, ty, tz, tr, now = self._data_processor.get_position_snapshot()

        if x is None or y is None or z is None:
            return

        # Check freshness
        if abs(now - tx) > 0.5 or abs(now - ty) > 0.5 or abs(now - tz) > 0.5:
            return

        # Check if at start position
        target_x = self.z_params.fixed_x_position
        target_y = self.z_params.fixed_y_position
        target_z = self.z_positions[0]

        dx = abs(x - target_x)
        dy = abs(y - target_y)
        dz = abs(z - target_z)

        tol = 5.0  # nm tolerance

        if dx <= tol and dy <= tol and dz <= tol:
            # Reached start position
            self.init_timer.stop()
            self._initializing = False
            self.status_update.emit("At start position. Beginning Z-scan...")
            self.execute_next_z_point()
            return

        # Check timeout
        if time.monotonic() - self._settle_start_time > 30:
            self.init_timer.stop()
            self._initializing = False
            self.status_update.emit("ERROR: Timeout reaching start position")
            self.complete_scan()

    def execute_next_z_point(self):
        """Move to next Z position"""
        if self.scan_state != ScanState.SCANNING or self.current_z_index >= len(self.z_positions):
            self.complete_scan()
            return

        # Get next Z position
        z_pos = self.z_positions[self.current_z_index]

        # Calculate X compensation (check ratio instead of enabled flag)
        if abs(self.z_params.x_compensation_ratio) > 1e-9:
            z_offset = z_pos - self.base_z_position
            x_offset = z_offset * self.z_params.x_compensation_ratio
            compensated_x = self.base_x_position + x_offset
        else:
            compensated_x = self.base_x_position

        # Calculate Y compensation
        if abs(self.z_params.y_compensation_ratio) > 1e-9:
            z_offset = z_pos - self.base_z_position
            y_offset = z_offset * self.z_params.y_compensation_ratio
            compensated_y = self.base_y_position + y_offset
        else:
            compensated_y = self.base_y_position

        # Set current point index in line reconstructor for Z vs signal plot
        if self.line_reconstructor:
            self.line_reconstructor.set_current_point(self.current_z_index)

        # Emit current position index for display tracking
        self.current_position_index.emit(self.current_z_index)

        # Send movement commands
        if abs(self.z_params.x_compensation_ratio) > 1e-9:
            self.movement_command.emit(f"MOVE/X/{compensated_x:.0f}")
        if abs(self.z_params.y_compensation_ratio) > 1e-9:
            self.movement_command.emit(f"MOVE/Y/{compensated_y:.0f}")
        self.movement_command.emit(f"MOVE/Z/{z_pos:.0f}")

        # Updated progress message for Z-points
        self.scan_progress.emit(self.current_z_index + 1, len(self.z_positions))
        self.status_update.emit(
            f"Moving to Z-point {self.current_z_index + 1}/{len(self.z_positions)} (Z={z_pos:.0f}nm)")

        # Start settle wait
        # Start settle wait
        self._current_target_z = z_pos
        self._current_target_x = compensated_x if abs(self.z_params.x_compensation_ratio) > 1e-9 else None
        self._current_target_y = compensated_y if abs(self.z_params.y_compensation_ratio) > 1e-9 else None
        self._start_settle_wait()

    def _start_settle_wait(self):
        """Start waiting for position to settle"""
        # Reset any detector lag state
        self._in_detector_lag = False
        self._detector_lag_timer.stop()

        self._waiting_to_settle = True
        self._settle_consecutive = 0
        self._settle_start_time = time.monotonic()
        self.settle_timer.start(20)  # 50Hz check

    def set_detector_lag(self, lag_seconds: float):
        """Set detector lag time in seconds"""
        self._detector_lag_s = lag_seconds

    def _start_detector_lag_wait(self):
        """Wait for detector lag before starting real dwell."""
        # Fast path for zero lag
        if self._detector_lag_s <= 0:
            self._start_dwell()
            return

        self._in_detector_lag = True
        self.status_update.emit(f"Waiting {self._detector_lag_s:.3f}s for detector stabilization...")
        self._detector_lag_timer.start(int(self._detector_lag_s * 1000))

    def _begin_dwell_after_detector_lag(self):
        """Begin actual dwell after detector lag period."""
        self._in_detector_lag = False
        self._start_dwell()

    def _on_settle_check(self):
        """Check if position has settled"""
        if not self._waiting_to_settle or not self._data_processor:
            self.settle_timer.stop()
            return

        x, y, z, r, tx, ty, tz, tr, now = self._data_processor.get_position_snapshot()

        if z is None or tz is None:
            return

        # Check freshness
        if abs(now - tz) > 0.3:
            return

        # Check Z tolerance
        z_error = abs(z - self._current_target_z)
        z_ok = z_error <= 5.0  # nm tolerance

        # Check X tolerance if compensation enabled
        x_ok = True
        if self._current_target_x is not None and x is not None and tx is not None:
            if abs(now - tx) < 0.3:
                x_error = abs(x - self._current_target_x)
                x_ok = x_error <= 5.0

        # Check Y tolerance if compensation enabled
        y_ok = True
        if self._current_target_y is not None and y is not None and ty is not None:
            if abs(now - ty) < 0.3:
                y_error = abs(y - self._current_target_y)
                y_ok = y_error <= 5.0

        if z_ok and x_ok and y_ok:
            self._settle_consecutive += 1
        else:
            self._settle_consecutive = 0

        # Check if settled
        if self._settle_consecutive >= 3:
            self.settle_timer.stop()
            self._waiting_to_settle = False
            self._start_detector_lag_wait()
            return

        # Check timeout
        if time.monotonic() - self._settle_start_time >= 5.0:
            self.settle_timer.stop()
            self._waiting_to_settle = False
            self.status_update.emit(f"Settle timeout at Z-point {self.current_z_index + 1}")
            self._start_detector_lag_wait()

    def _start_dwell(self):
        """Start dwell time acquisition"""
        dwell_ms = int(self.z_params.z_dwell_time * 1000)
        self.step_timer.start(dwell_ms)

    def on_dwell_completed(self):
        """Called when dwell is complete"""
        self.current_z_index += 1
        self.execute_next_z_point()

    def stop_scan(self):
        """Stop Z-scan"""
        self.scan_state = ScanState.IDLE
        self.settle_timer.stop()
        self.step_timer.stop()
        self.init_timer.stop()
        self._detector_lag_timer.stop()
        self._in_detector_lag = False

        # Send stop commands
        self.movement_command.emit("STOP/X")
        self.movement_command.emit("STOP/Y")
        self.movement_command.emit("STOP/Z")

        self.complete_scan()

    def complete_scan(self):
        """Complete the Z-scan"""
        self.scan_state = ScanState.IDLE

        # Clear current point in reconstructor
        if self.line_reconstructor:
            self.line_reconstructor.clear_current_point()
            self.line_reconstructor.complete_scan()

        self.scan_completed.emit()
        self.status_update.emit(f"Z-scan completed: {self.current_z_index} points measured")

@dataclass
class DataPoint:
    """Single data point with timestamp and position"""
    timestamp: float
    x_pos: float
    y_pos: float
    z_pos: Optional[float]
    r_pos: Optional[float]
    current: float

class CircularBuffer:
    """Thread-safe circular buffer for high-frequency data"""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.buffer = deque(maxlen=maxsize)
        self.mutex = QMutex()

    def append(self, item):
        with QMutexLocker(self.mutex):
            self.buffer.append(item)

    def get_recent(self, n: int = None) -> List:
        with QMutexLocker(self.mutex):
            if n is None:
                return list(self.buffer)
            else:
                return list(self.buffer)[-n:]

    def clear(self):
        with QMutexLocker(self.mutex):
            self.buffer.clear()

    def __len__(self):
        with QMutexLocker(self.mutex):
            return len(self.buffer)

class DataStorage:
    """Simplified HDF5-only data storage"""

    def __init__(self):
        # Separate counters for each scan type
        self.s_counter = 0  # 2D scans
        self.l_counter = 0  # Line scans
        self.z_counter = 0  # Z scans
        self.current_scan_type = None
        self.current_scan_counter = 0

    def get_next_scan_id(self, scan_type: str) -> tuple[str, int]:
        """Get next scan ID based on type
        Returns: (prefix, counter)
        """
        if scan_type == "2D":
            self.s_counter += 1
            return "S", self.s_counter
        elif scan_type == "LINE":
            self.l_counter += 1
            return "L", self.l_counter
        elif scan_type == "Z":
            self.z_counter += 1
            return "Z", self.z_counter
        else:
            raise ValueError(f"Unknown scan type: {scan_type}")

    def create_hdf5_file(self, scan_type: str, base_path: str,
                         scan_params: Optional[ScanParameters] = None,
                         line_params: Optional[LineScanParameters] = None,
                         z_index: Optional[int] = None,
                         z_position: Optional[float] = None,
                         actual_z_position: Optional[float] = None,
                         r_index: Optional[int] = None,
                         r_position: Optional[float] = None,
                         actual_r_position: Optional[float] = None) -> Optional[str]:
        """Create HDF5 file with scan parameters stored as attributes"""
        if not HDF5_AVAILABLE:
            print("Warning: HDF5 not available, skipping raw data file creation")
            return None

        # Get scan ID based on type
        if scan_type == "2D":
            self.s_counter += 1
            prefix = "S"
            counter = self.s_counter
        elif scan_type == "LINE":
            self.l_counter += 1
            prefix = "L"
            counter = self.l_counter
        elif scan_type == "Z":
            self.z_counter += 1
            prefix = "Z"
            counter = self.z_counter
        else:
            # Fallback
            self.s_counter += 1
            prefix = "S"
            counter = self.s_counter



        # Create directory if needed
        os.makedirs(base_path, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}{counter:06d}.h5"
        filepath = os.path.join(base_path, filename)

        # Ensure unique filename
        while os.path.exists(filepath):
            counter += 1
            if scan_type == "2D":
                self.s_counter = counter
            elif scan_type == "LINE":
                self.l_counter = counter
            elif scan_type == "Z":
                self.z_counter = counter

            filename = f"{prefix}{counter:06d}.h5"
            filepath = os.path.join(base_path, filename)

        self.current_scan_type = scan_type
        self.current_scan_counter = counter
        self.current_scan_name = f"{prefix}{counter:06d}"

        try:
            # Create file and store scan parameters as attributes
            with h5py.File(filepath, 'w') as f:
                # Store common attributes
                f.attrs['scan_name'] = self.current_scan_name
                f.attrs['scan_id'] = counter
                f.attrs['scan_type'] = scan_type
                f.attrs['timestamp'] = timestamp
                f.attrs['detector_lag_s'] = getattr(self, 'current_detector_lag', 0.350)

                # Store Z information in attributes
                if z_position is not None:
                    # Z-series mode: use the planned Z position + index
                    f.attrs['z_position'] = z_position
                    f.attrs['z_index'] = z_index or 0
                elif actual_z_position is not None:
                    # Regular (non-Z-series) scan: current Z
                    f.attrs['z_position'] = actual_z_position
                    f.attrs['z_index'] = 'Not in Z mode'

                # Store R information in attributes (ADD THIS SECTION)
                if r_position is not None:
                    # R-series mode: use the planned R position + index
                    f.attrs['r_position'] = r_position
                    f.attrs['r_index'] = r_index or 0
                elif actual_r_position is not None:
                    # Regular scan: current R
                    f.attrs['r_position'] = actual_r_position
                    f.attrs['r_index'] = 'Not in R mode'

                # Store appropriate scan parameters
                if line_params:
                    # 1D line (including 1D Z-scan)
                    f.attrs['scan_type'] = 'line'
                    f.attrs['fixed_axis'] = line_params.fixed_axis
                    f.attrs['fixed_position'] = line_params.fixed_position
                    f.attrs['scan_axis'] = line_params.scan_axis
                    f.attrs['scan_start'] = line_params.scan_start
                    f.attrs['scan_end'] = line_params.scan_end
                    f.attrs['num_points'] = line_params.num_points
                    f.attrs['step_size'] = line_params.step_size
                    f.attrs['dwell_time'] = line_params.dwell_time

                    # If a Z-series is enabled on the line, persist its UI values
                    if getattr(line_params, 'z_series', None) and line_params.z_series.enabled:
                        zs = line_params.z_series
                        f.attrs['z_series_enabled'] = True
                        f.attrs['z_series_start'] = zs.z_start
                        f.attrs['z_series_end'] = zs.z_end
                        f.attrs['z_series_numbers'] = zs.z_numbers
                        f.attrs['z_series_step'] = zs.z_step

                        # persist fixed X/Y and base Z used during a 1D Z-scan
                        f.attrs['z_series_fixed_x'] = zs.fixed_x_position
                        f.attrs['z_series_fixed_y'] = zs.fixed_y_position
                        f.attrs['z_series_base_z'] = getattr(line_params, 'base_z_position', 0.0)
                        f.attrs['z_series_x_compensation_ratio'] = zs.x_compensation_ratio
                        f.attrs['z_series_y_compensation_ratio'] = zs.y_compensation_ratio
                    f.attrs['z_series_dwell_time'] = zs.z_dwell_time

                    # If R-series is enabled, persist its values
                    if getattr(line_params, 'r_series', None) and line_params.r_series.enabled:
                        rs = line_params.r_series
                        f.attrs['r_series_enabled'] = True
                        f.attrs['r_series_start'] = rs.r_start
                        f.attrs['r_series_end'] = rs.r_end
                        f.attrs['r_series_numbers'] = rs.r_numbers
                        f.attrs['r_series_step'] = rs.r_step
                        f.attrs['r_series_base_position'] = rs.base_r_position
                        f.attrs['r_series_cor_enabled'] = rs.cor_enabled
                        f.attrs['r_series_cor_x'] = rs.cor_x
                        f.attrs['r_series_cor_y'] = rs.cor_y
                        f.attrs['r_series_cor_base_z'] = rs.cor_base_z
                        f.attrs['r_series_cor_x_compensation_ratio'] = rs.cor_x_compensation_ratio
                        f.attrs['r_series_cor_y_compensation_ratio'] = rs.cor_y_compensation_ratio
                        if rs.combine_with_z:
                            f.attrs['r_series_combined_with_z'] = True

                elif scan_params:
                    # 2D scan
                    f.attrs['scan_type'] = '2d'
                    f.attrs['x_start_nm'] = scan_params.x_start
                    f.attrs['x_end_nm'] = scan_params.x_end
                    f.attrs['y_start_nm'] = scan_params.y_start
                    f.attrs['y_end_nm'] = scan_params.y_end
                    f.attrs['x_pixels'] = scan_params.x_pixels
                    f.attrs['y_pixels'] = scan_params.y_pixels
                    f.attrs['x_step_nm'] = scan_params.x_step
                    f.attrs['y_step_nm'] = scan_params.y_step
                    f.attrs['scan_mode'] = scan_params.mode.value
                    f.attrs['dwell_time_s'] = scan_params.dwell_time
                    f.attrs['scan_speed_nm_s'] = scan_params.scan_speed
                    f.attrs['scan_pattern'] = scan_params.pattern

                    #Save actual Z and R positions for 2D scans
                    if actual_z_position is not None:
                        f.attrs['z_position'] = actual_z_position
                        f.attrs['z_index'] = 'Not in Z mode'
                    if actual_r_position is not None:
                        f.attrs['r_position'] = actual_r_position
                        f.attrs['r_index'] = 'Not in R mode'

                    if getattr(scan_params, 'z_series', None) and scan_params.z_series.enabled:
                        zs = scan_params.z_series
                        f.attrs['z_series_enabled'] = True
                        f.attrs['z_series_start'] = zs.z_start
                        f.attrs['z_series_end'] = zs.z_end
                        f.attrs['z_series_numbers'] = zs.z_numbers
                        f.attrs['z_series_step'] = zs.z_step

                    # If R-series is enabled, persist its values
                    if getattr(scan_params, 'r_series', None) and scan_params.r_series.enabled:
                        rs = scan_params.r_series
                        f.attrs['r_series_enabled'] = True
                        f.attrs['r_series_start'] = rs.r_start
                        f.attrs['r_series_end'] = rs.r_end
                        f.attrs['r_series_numbers'] = rs.r_numbers
                        f.attrs['r_series_step'] = rs.r_step
                        f.attrs['r_series_base_position'] = rs.base_r_position
                        f.attrs['r_series_cor_enabled'] = rs.cor_enabled
                        f.attrs['r_series_cor_x'] = rs.cor_x
                        f.attrs['r_series_cor_y'] = rs.cor_y
                        f.attrs['r_series_cor_base_z'] = rs.cor_base_z
                        f.attrs['r_series_cor_x_compensation_ratio'] = rs.cor_x_compensation_ratio
                        f.attrs['r_series_cor_y_compensation_ratio'] = rs.cor_y_compensation_ratio

                # Ensure groups exist (raw_data always; reconstructed_image only for 2D)
                f.create_group('raw_data')
                if scan_params:
                    f.create_group('reconstructed_image')

            print(f"Created HDF5 file: {filepath}")
            return filepath

        except Exception as e:
            print(f"Error creating HDF5 file: {e}")
            return None

    def save_raw_data_batch(self, filepath: str, data_points: List[DataPoint]):
        """Save batch of raw data to HDF5"""
        if not HDF5_AVAILABLE or not filepath or not data_points:
            return

        try:
            with h5py.File(filepath, 'a') as f:
                grp = f.require_group('raw_data')

                # Create or extend datasets
                if 'timestamps' not in grp:
                    n = len(data_points)
                    # Specify dtype explicitly for better compatibility
                    grp.create_dataset('timestamps', data=[dp.timestamp for dp in data_points],
                                       maxshape=(None,), chunks=True, compression='gzip', dtype='f8')
                    grp.create_dataset('x_positions', data=[dp.x_pos for dp in data_points],
                                       maxshape=(None,), chunks=True, compression='gzip', dtype='f8')
                    grp.create_dataset('y_positions', data=[dp.y_pos for dp in data_points],
                                       maxshape=(None,), chunks=True, compression='gzip', dtype='f8')
                    grp.create_dataset('z_positions',
                                       data=[dp.z_pos if dp.z_pos is not None else np.nan for dp in data_points],
                                       maxshape=(None,), chunks=True, compression='gzip', dtype='f8')
                    grp.create_dataset('r_positions',
                                       data=[dp.r_pos if dp.r_pos is not None else np.nan for dp in data_points],
                                       maxshape=(None,), chunks=True, compression='gzip', dtype='f8')
                    grp.create_dataset('currents', data=[dp.current for dp in data_points],
                                       maxshape=(None,), chunks=True, compression='gzip', dtype='f8')
                    #print(f"Created HDF5 datasets with {n} initial data points")
                else:
                    # Append to existing datasets
                    for name, data in [
                        ('timestamps', [dp.timestamp for dp in data_points]),
                        ('x_positions', [dp.x_pos for dp in data_points]),
                        ('y_positions', [dp.y_pos for dp in data_points]),
                        ('z_positions', [dp.z_pos if dp.z_pos is not None else np.nan for dp in data_points]),
                        ('r_positions', [dp.r_pos if dp.r_pos is not None else np.nan for dp in data_points]),
                        ('currents', [dp.current for dp in data_points])
                    ]:
                        dataset = grp[name]
                        old_size = dataset.shape[0]
                        new_size = old_size + len(data)
                        dataset.resize((new_size,))  # Explicitly pass as tuple
                        dataset[old_size:new_size] = data

                    print(f"Appended {len(data_points)} data points to HDF5 (total: {new_size})")

        except Exception as e:
            print(f"Error saving raw data batch: {e}")
            import traceback
            traceback.print_exc()

    def save_final_image(self, filepath: str, image_data: np.ndarray, scan_params: ScanParameters = None):
        """Save final reconstructed image with position mapping to HDF5"""
        if not HDF5_AVAILABLE or not filepath:
            return

        try:
            with h5py.File(filepath, 'a') as f:
                grp = f.require_group('reconstructed_image')

                # Clear existing data
                for key in ['image', 'x_coordinates', 'y_coordinates']:
                    if key in grp:
                        del grp[key]

                # Save image data
                grp.create_dataset('image', data=image_data, compression='gzip')
                grp['image'].attrs['timestamp'] = datetime.now().isoformat()

                # Create position mapping if scan parameters available
                if scan_params:
                    # Create coordinate arrays
                    y_pixels, x_pixels = image_data.shape

                    # Determine scan direction
                    x_forward = scan_params.x_end >= scan_params.x_start
                    y_forward = scan_params.y_end >= scan_params.y_start

                    # Calculate actual step with direction
                    x_step = scan_params.x_step_input if x_forward else -scan_params.x_step_input
                    y_step = scan_params.y_step_input if y_forward else -scan_params.y_step_input

                    # Generate coordinate arrays
                    x_coords = np.array([scan_params.x_start + i * x_step for i in range(x_pixels)])
                    y_coords = np.array([scan_params.y_start + i * y_step for i in range(y_pixels)])

                    # Round to integers to match commanded positions
                    x_coords = np.round(x_coords).astype(int)
                    y_coords = np.round(y_coords).astype(int)

                    # Save coordinate arrays
                    grp.create_dataset('x_coordinates', data=x_coords, compression='gzip')
                    grp.create_dataset('y_coordinates', data=y_coords, compression='gzip')

                    # Add metadata
                    grp['x_coordinates'].attrs['units'] = 'nm'
                    grp['y_coordinates'].attrs['units'] = 'nm'
                    grp['x_coordinates'].attrs['description'] = 'X position'
                    grp['y_coordinates'].attrs['description'] = 'Y position'

        except Exception as e:
            print(f"Error saving final image: {e}")

    def parse_device_timestamp(self, timestamp_str: str, fallback_time: float) -> float:
        """Parse device timestamp with fallback"""
        try:
            ts = float(timestamp_str)
            # Convert based on magnitude
            if ts > 1e12:  # microseconds
                ts *= 1e-6
            elif ts > 1e10:  # milliseconds
                ts *= 1e-3
            # Check if reasonable
            if abs(ts - fallback_time) < 86400:  # within 1 day
                return ts
            return fallback_time
        except (ValueError, TypeError):
            return fallback_time

    def export_reconstructed_image(self, image_data: np.ndarray, scan_params: ScanParameters,
                                   base_path: str,
                                   z_index: Optional[int] = None, r_index: Optional[int] = None,
                                   z_position: Optional[float] = None, r_position: Optional[float] = None):
        """Export reconstructed image as PNG and CSV in main data folder"""
        try:
            # Save directly to base_path, not a subdirectory
            os.makedirs(base_path, exist_ok=True)

            # Use the stored scan name (already has prefix and counter)
            base_name = self.current_scan_name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Simple naming - just base name
            png_path = os.path.join(base_path, f"{base_name}.png")
            csv_path = os.path.join(base_path, f"{base_name}.csv")

            # PNG export (rest of method unchanged)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))

            extent = [scan_params.x_start, scan_params.x_end,
                      scan_params.y_start, scan_params.y_end]

            im = ax.imshow(image_data, extent=extent, origin='lower',
                           interpolation='nearest', cmap='viridis', aspect='equal')
            plt.colorbar(im, ax=ax, label='Signal (nA)')
            ax.set_xlabel('X Position (nm)')
            ax.set_ylabel('Y Position (nm)')
            ax.set_title(f'Scan {base_name}')
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            # CSV export with metadata
            with open(csv_path, 'w') as f:
                f.write("# 2D Scan Data\n")
                f.write(f"# Scan ID: {base_name}\n")
                f.write(f"# Timestamp: {timestamp}\n")
                f.write(f"# X range: {scan_params.x_start} to {scan_params.x_end} nm\n")
                f.write(f"# Y range: {scan_params.y_start} to {scan_params.y_end} nm\n")
                f.write(f"# Pixels: {scan_params.x_pixels} x {scan_params.y_pixels}\n")

                if z_position is not None:
                    f.write(f"# Z position: {z_position:.0f} nm\n")
                if r_position is not None:
                    r_deg = r_position / 1e6 if abs(r_position) > 720 else r_position
                    f.write(f"# R position: {r_deg:.6f} deg\n")

                # Add R-series COR information if available
                if hasattr(scan_params, 'r_series') and scan_params.r_series and scan_params.r_series.enabled:
                    if scan_params.r_series.cor_enabled:
                        f.write(f"# COR X: {scan_params.r_series.cor_x:.1f} nm\n")
                        f.write(f"# COR Y: {scan_params.r_series.cor_y:.1f} nm\n")
                        f.write(f"# COR Base Z: {scan_params.r_series.cor_base_z:.1f} nm\n")
                        f.write(f"# COR X Compensation Ratio: {scan_params.r_series.cor_x_compensation_ratio}\n")
                        f.write(f"# COR Y Compensation Ratio: {scan_params.r_series.cor_y_compensation_ratio}\n")

                f.write("# Format: X(nm), Y(nm), Signal(nA)\n")
                f.write("X_nm,Y_nm,Signal_nA\n")

                y_pixels, x_pixels = image_data.shape
                x_coords = np.linspace(scan_params.x_start, scan_params.x_end, x_pixels)
                y_coords = np.linspace(scan_params.y_start, scan_params.y_end, y_pixels)

                for yi, y in enumerate(y_coords):
                    row = image_data[yi]
                    for xi, x in enumerate(x_coords):
                        v = row[xi]
                        if not np.isnan(v):
                            f.write(f"{x:.0f},{y:.0f},{v:.6f}\n")

            print(f"Exported image to: {png_path}")
            print(f"Exported data to: {csv_path}")
            return png_path, csv_path

        except Exception as e:
            print(f"Error exporting image: {e}")
            return None, None

    def export_line_scan_data(self, positions: np.ndarray, signals: np.ndarray,
                              line_params: LineScanParameters, base_path: str,
                              z_position: Optional[float] = None,
                              r_position: Optional[float] = None):
        """Export line scan data as PNG plot and CSV in main data folder"""
        try:
            # Save directly to base_path, not a subdirectory
            os.makedirs(base_path, exist_ok=True)

            # Use the stored scan name
            base_name = self.current_scan_name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            valid = ~np.isnan(signals)
            if not np.any(valid):
                print("No valid data to export")
                return None, None
            xp = positions[valid]
            yp = signals[valid]

            # PNG
            png_path = os.path.join(base_path, f"{base_name}.png")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(xp, yp, '-', linewidth=1.5, marker='.', markersize=4)
            ax.grid(True, alpha=0.3)

            # Set axis labels based on scan type
            if self.current_scan_type == "Z":
                ax.set_xlabel('Z Position (nm)')
                ax.set_title(f'Z-Scan {base_name}')
            else:
                if hasattr(line_params, 'scan_axis'):
                    ax.set_xlabel(f'{line_params.scan_axis} Position (nm)')
                else:
                    ax.set_xlabel('Position (nm)')
                ax.set_title(f'Line Scan {base_name}')

            ax.set_ylabel('Signal (nA)')

            # Set limits
            ax.set_xlim([np.min(xp), np.max(xp)])
            y_margin = (np.max(yp) - np.min(yp)) * 0.05
            ax.set_ylim([np.min(yp) - y_margin, np.max(yp) + y_margin])

            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            # CSV
            csv_path = os.path.join(base_path, f"{base_name}.csv")
            with open(csv_path, 'w') as f:
                f.write("# Line Scan Data\n")
                f.write(f"# Scan ID: {base_name}\n")
                f.write(f"# Timestamp: {timestamp}\n")

                # Write scan-specific metadata
                if self.current_scan_type == "Z":
                    f.write("# Scan type: Z-scan (1D)\n")
                    if hasattr(line_params, 'z_series') and line_params.z_series:
                        f.write(f"# Fixed X (nm): {line_params.z_series.fixed_x_position:.1f}\n")
                        f.write(f"# Fixed Y (nm): {line_params.z_series.fixed_y_position:.1f}\n")
                        f.write(
                            f"# Z range (nm): {line_params.z_series.z_start:.1f} to {line_params.z_series.z_end:.1f}\n")
                        f.write(f"# Points: {line_params.z_series.z_numbers}\n")
                    f.write("# Moving axis: Z\n")
                else:
                    f.write("# Scan type: XY line (1D)\n")
                    if hasattr(line_params, 'fixed_axis'):
                        f.write(f"# Fixed axis: {line_params.fixed_axis}\n")
                        f.write(f"# Fixed position (nm): {line_params.fixed_position:.1f}\n")
                        f.write(f"# Moving axis: {line_params.scan_axis}\n")
                        f.write(f"# Scan range (nm): {line_params.scan_start:.1f} to {line_params.scan_end:.1f}\n")
                        f.write(f"# Points: {line_params.num_points}\n")

                # Add position metadata if provided
                if z_position is not None:
                    f.write(f"# Z position (nm): {z_position:.0f}\n")
                if r_position is not None:
                    r_deg = r_position / 1e6 if abs(r_position) > 720 else r_position
                    f.write(f"# R position: {r_deg:.6f} deg\n")

                # Add R-series COR information if available
                if hasattr(line_params, 'r_series') and line_params.r_series and line_params.r_series.enabled:
                    if line_params.r_series.cor_enabled:
                        f.write(f"# COR X: {line_params.r_series.cor_x:.1f} nm\n")
                        f.write(f"# COR Y: {line_params.r_series.cor_y:.1f} nm\n")
                        f.write(f"# COR Base Z: {line_params.r_series.cor_base_z:.1f} nm\n")
                        f.write(f"# COR X Compensation Ratio: {line_params.r_series.cor_x_compensation_ratio}\n")
                        f.write(f"# COR Y Compensation Ratio: {line_params.r_series.cor_y_compensation_ratio}\n")

                # Column header
                if self.current_scan_type == "Z":
                    f.write("Z_nm,Signal_nA\n")
                else:
                    f.write("Position_nm,Signal_nA\n")

                for pos, sig in zip(xp, yp):
                    f.write(f"{pos:.0f},{sig:.6f}\n")

            print(f"Exported line plot to: {png_path}")
            print(f"Exported line data to: {csv_path}")
            return png_path, csv_path

        except Exception as e:
            print(f"Error exporting line scan: {e}")
            return None, None


class MQTTController(QObject):
    """MQTT communication controller"""

    # Signals
    connected = pyqtSignal(bool)
    position_data_received = pyqtSignal(str)  # Raw position payload
    current_data_received = pyqtSignal(str)  # Raw current payload
    command_result_received = pyqtSignal(str)  # Command result
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.client = None
        self.broker_host = "localhost"
        self.broker_port = 1883
        self.connected_status = False

        # Topics
        self.topics = {
            'picoammeter': "picoammeter/current",
            'stage_position': "microscope/stage/position",
            'stage_command': "microscope/stage/command",
            'stage_result': "microscope/stage/result"
        }

    def setup_mqtt(self, host: str, port: int):
        """Setup MQTT client"""
        self.broker_host = host
        self.broker_port = port

        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except Exception:
            self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """MQTT connection callback"""
        if rc == 0:
            self.connected_status = True
            self.connected.emit(True)

            # : Subscribe only to inbound topics (avoid noise)
            client.subscribe(self.topics['picoammeter'])
            client.subscribe(self.topics['stage_position'])
            client.subscribe(self.topics['stage_result'])
            # Don't subscribe to stage_command - we only publish to it

            self.status_update.emit(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
        else:
            self.connected_status = False
            self.connected.emit(False)
            self.error_occurred.emit(f"MQTT connection failed: code {rc}")

    def _on_disconnect(self, client, userdata, rc, properties=None, reason_code=None):
        """MQTT disconnection callback - compatible with both API versions"""
        self.connected_status = False
        self.connected.emit(False)

        # Handle both old and new callback signatures
        if rc is not None:
            if rc == 0:
                self.status_update.emit("Disconnected from MQTT broker")
            else:
                self.status_update.emit(f"Disconnected from MQTT broker (rc: {rc})")
        else:
            self.status_update.emit("Disconnected from MQTT broker")

    def _on_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            topic = msg.topic
            payload = msg.payload.decode()

            if topic == self.topics['picoammeter']:
                self.current_data_received.emit(payload)
            elif topic == self.topics['stage_position']:
                self.position_data_received.emit(payload)
            elif topic == self.topics['stage_result']:
                self.command_result_received.emit(payload)

        except Exception as e:
            self.error_occurred.emit(f"Message processing error: {e}")

    @pyqtSlot()
    def connect_mqtt(self):
        """Connect to MQTT broker"""
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
        except Exception as e:
            self.error_occurred.emit(f"Connection error: {e}")

    @pyqtSlot()
    def disconnect_mqtt(self):
        """Disconnect from MQTT broker"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()

    @pyqtSlot(str)
    def send_command(self, command: str):
        """Send command via MQTT"""
        if self.client and self.connected_status:
            try:
                self.client.publish(self.topics['stage_command'], command)
                self.status_update.emit(f"Sent command: {command}")
            except Exception as e:
                self.error_occurred.emit(f"Command send error: {e}")

class DataProcessor(QObject):
    """Process and synchronize high-frequency data"""

    # Signals
    new_data_point = pyqtSignal(object)  # DataPoint object
    statistics_update = pyqtSignal(dict)
    live_readings = pyqtSignal(dict)

    def __init__(self):
        super().__init__()

        self.data_storage = None  # Will be set by MainWindow
        self.scan_controller = None  # will be set by MainWindow
        self.line_scan_controller = None

        # Current state
        self.latest_positions = {"X": None, "Y": None, "Z": None, "R": None}
        self.latest_current = None
        # : Track position timestamps for freshness
        self.latest_pos_time = {"X": None, "Y": None, "Z": None, "R": None}
        self.mutex = QMutex()

        # Statistics tracking
        self.stats = {
            'position_messages': 0,
            'current_messages': 0,
            'data_points_created': 0,
            'start_time': time.time(),
            'last_position_time': None,
            'last_current_time': None
        }

    def set_data_storage(self, data_storage):
        """Set reference to data storage for timestamp parsing"""
        self.data_storage = data_storage

    def get_position_snapshot(self):
        """Return latest (X,Y,Z) positions + their timestamps in a thread-safe way."""
        with QMutexLocker(self.mutex):
            return (
                self.latest_positions.get('X'),
                self.latest_positions.get('Y'),
                self.latest_positions.get('Z'),
                self.latest_positions.get('R'),
                self.latest_pos_time.get('X'),
                self.latest_pos_time.get('Y'),
                self.latest_pos_time.get('Z'),
                self.latest_pos_time.get('R'),
                time.time(),  # 'now' for freshness checks
            )

    @pyqtSlot(str)
    def process_position_data(self, payload: str):
        """Process ECC100 position data - 2: Only update positions, don't emit"""
        try:
            current_time = time.time()
            lines = payload.strip().split('\n')

            for line in lines:
                if not line:
                    continue

                parts = line.split('/')
                if len(parts) == 5:
                    timestamp_str, x_str, y_str, z_str, r_str = parts

                    # : Parse device timestamp with fallback
                    device_time = current_time
                    if self.data_storage:
                        device_time = self.data_storage.parse_device_timestamp(timestamp_str, current_time)

                    # Parse positions
                    positions = {}
                    for axis, val_str in zip(['X', 'Y', 'Z', 'R'],
                                             [x_str, y_str, z_str, r_str]):
                        if val_str != "NaN":
                            positions[axis] = float(val_str)
                        else:
                            positions[axis] = None

                    with QMutexLocker(self.mutex):
                        # : Update positions and timestamps
                        for axis, val in positions.items():
                            self.latest_positions[axis] = val
                            self.latest_pos_time[axis] = device_time

                        self.stats['position_messages'] += 1
                        self.stats['last_position_time'] = device_time
                    
                        self.live_readings.emit({
                        "time": current_time,
                        "current": self.latest_current,
                        "current_time": self.stats.get("last_current_time"),
                        "X": self.latest_positions.get('X'),
                        "Y": self.latest_positions.get('Y'),
                        "Z": self.latest_positions.get('Z'),
                        "R": self.latest_positions.get('R'),
                        "tX": self.latest_pos_time.get('X'),
                        "tY": self.latest_pos_time.get('Y'),
                        "tZ": self.latest_pos_time.get('Z'),
                        "tR": self.latest_pos_time.get('R'),
                    })

        except Exception as e:
            print(f"Error processing position data: {e}")

    @pyqtSlot(str)
    def process_current_data(self, payload: str):
        """Process picoammeter current data - gate to dwell windows for active controller"""
        try:
            current_time = time.time()
            parts = payload.strip().split('/')
            if len(parts) != 2:
                return

            timestamp_str, current_str = parts
            current_value = float(current_str)
            current_value = -current_value

            # Device timestamp (with fallback)
            device_time = current_time
            if self.data_storage:
                device_time = self.data_storage.parse_device_timestamp(timestamp_str, current_time)

            with QMutexLocker(self.mutex):
                self.latest_current = current_value
                self.stats['current_messages'] += 1
                self.stats['last_current_time'] = device_time

                # Always emit latest live readings (for UI widgets)
                self.live_readings.emit({
                    "time": current_time,
                    "current": self.latest_current,
                    "current_time": self.stats.get("last_current_time"),
                    "X": self.latest_positions.get('X'),
                    "Y": self.latest_positions.get('Y'),
                    "Z": self.latest_positions.get('Z'),
                    "R": self.latest_positions.get('R'),
                    "tX": self.latest_pos_time.get('X'),
                    "tY": self.latest_pos_time.get('Y'),
                    "tZ": self.latest_pos_time.get('Z'),
                    "tR": self.latest_pos_time.get('R'),
                })

                # Determine which controller is actively DWELLING (step_timer running, not settling)
                controller = None

                # 1) Prefer Z-scan controller if scanning (new)
                zsc = getattr(self, 'z_scan_controller', None)
                if zsc and getattr(zsc, 'scan_state', None) == ScanState.SCANNING:
                    controller = zsc

                # 2) Else line-scan (1D XY) controller
                if controller is None:
                    lsc = getattr(self, 'line_scan_controller', None)
                    if lsc and getattr(lsc, 'scan_state', None) == ScanState.SCANNING:
                        controller = lsc

                # 3) Else 2D scan controller
                if controller is None:
                    sc = getattr(self, 'scan_controller', None)
                    if sc and getattr(sc, 'scan_state', None) == ScanState.SCANNING:
                        controller = sc

                if (controller and
                        getattr(controller, 'step_timer', None) and controller.step_timer.isActive() and
                        not getattr(controller, '_waiting_to_settle', False)):

                    # Skip if in detector lag period
                    if getattr(controller, "_in_detector_lag", False):
                        return

                    # Only emit when positions are fresh enough
                    x = self.latest_positions.get('X')
                    y = self.latest_positions.get('Y')
                    z = self.latest_positions.get('Z')
                    r = self.latest_positions.get('R')
                    tx = self.latest_pos_time.get('X')
                    ty = self.latest_pos_time.get('Y')
                    tz = self.latest_pos_time.get('Z')

                    FRESH = 0.300  # seconds

                    if controller == zsc:
                        # For a Z-scan dwell, require fresh X, Y, and Z
                        if (x is not None and y is not None and z is not None and
                                tx is not None and ty is not None and tz is not None and
                                abs(device_time - tx) < FRESH and
                                abs(device_time - ty) < FRESH and
                                abs(device_time - tz) < FRESH):
                            dp = DataPoint(
                                timestamp=device_time,
                                x_pos=x,
                                y_pos=y,
                                z_pos=z,
                                r_pos=r,
                                current=current_value
                            )
                            self.new_data_point.emit(dp)
                            self.stats['data_points_created'] += 1
                    else:
                        # Regular 2D or 1D XY line dwell: need X and Y fresh
                        if (x is not None and y is not None and
                                tx is not None and ty is not None and
                                abs(device_time - tx) < FRESH and
                                abs(device_time - ty) < FRESH):
                            dp = DataPoint(
                                timestamp=device_time,
                                x_pos=x,
                                y_pos=y,
                                z_pos=z,
                                r_pos=r,
                                current=current_value
                            )
                            self.new_data_point.emit(dp)
                            self.stats['data_points_created'] += 1

        except Exception as e:
            print(f"Error processing current data: {e}")

    @pyqtSlot()
    def emit_statistics(self):
        """Emit current statistics with data rates"""
        with QMutexLocker(self.mutex):
            stats_copy = self.stats.copy()
            runtime = time.time() - stats_copy['start_time']
            stats_copy['runtime'] = runtime

            # Calculate data rates
            if runtime > 0:
                stats_copy['position_rate'] = stats_copy['position_messages'] / runtime
                stats_copy['current_rate'] = stats_copy['current_messages'] / runtime
                stats_copy['datapoint_rate'] = stats_copy['data_points_created'] / runtime
            else:
                stats_copy['position_rate'] = 0
                stats_copy['current_rate'] = 0
                stats_copy['datapoint_rate'] = 0

            self.statistics_update.emit(stats_copy)

class ImageReconstructor(QObject):
    """Real-time image reconstruction"""

    # Signals
    image_updated = pyqtSignal(object)  # numpy array
    pixel_updated = pyqtSignal(int, int, float)  # x, y, value

    def __init__(self):
        super().__init__()
        self.scan_params = None
        self.image = None
        self.pixel_counts = None  # Track number of samples per pixel
        self.mutex = QMutex()

        # Reconstruction settings
        self.min_samples_per_pixel = 1
        self.interpolation_method = "nearest"  # nearest, linear, cubic

    def initialize_scan(self, scan_params: ScanParameters):
        """Initialize for new scan"""
        with QMutexLocker(self.mutex):
            self.scan_params = scan_params
            # Initialize with zeros instead of NaN to avoid rendering issues
            self.image = np.full((scan_params.y_pixels, scan_params.x_pixels), np.nan, dtype=np.float64)
            self.pixel_counts = np.zeros((scan_params.y_pixels, scan_params.x_pixels))

            # Emit initial image with proper levels to set up display properly
            self.image_updated.emit(self.image.copy())

    @pyqtSlot(int, int)
    def set_current_pixel(self, x_idx: int, y_idx: int):
        """Thread-safe setter for current pixel indices"""
        with QMutexLocker(self.mutex):
            self._current_scan_pixel_x = int(x_idx)
            self._current_scan_pixel_y = int(y_idx)

    @pyqtSlot()
    def clear_current_pixel(self):
        """Clear current pixel indices when scan stops"""
        with QMutexLocker(self.mutex):
            for attr in ('_current_scan_pixel_x', '_current_scan_pixel_y'):
                if hasattr(self, attr):
                    delattr(self, attr)

    @pyqtSlot(object)
    def add_data_point(self, data_point: DataPoint):
        """Add data point to image reconstruction using scan grid coordinates"""
        if not self.scan_params:
            return

        try:
            with QMutexLocker(self.mutex):
                # Check if we have valid pixel coordinates
                if not (hasattr(self, '_current_scan_pixel_x') and
                        hasattr(self, '_current_scan_pixel_y')):
                    return

                x_pixel = self._current_scan_pixel_x
                y_pixel = self._current_scan_pixel_y

                # Check for None values
                if x_pixel is None or y_pixel is None:
                    return

                # Bounds check
                if not (0 <= x_pixel < self.scan_params.x_pixels and
                        0 <= y_pixel < self.scan_params.y_pixels):
                    return

                # Update pixel value (running average)
                current_count = self.pixel_counts[y_pixel, x_pixel]
                if current_count == 0:
                    self.image[y_pixel, x_pixel] = data_point.current
                else:
                    old_value = self.image[y_pixel, x_pixel]
                    # Handle NaN for first measurement
                    if np.isnan(old_value):
                        self.image[y_pixel, x_pixel] = data_point.current
                    else:
                        self.image[y_pixel, x_pixel] = (old_value * current_count + data_point.current) / (
                                    current_count + 1)

                self.pixel_counts[y_pixel, x_pixel] += 1
                self.pixel_updated.emit(x_pixel, y_pixel, self.image[y_pixel, x_pixel])

        except Exception as e:
            print(f"Error adding data point to image: {e}")

    @pyqtSlot()
    def emit_full_image(self):
        """Emit complete current image"""
        with QMutexLocker(self.mutex):
            if self.image is not None:
                self.image_updated.emit(self.image.copy())

    def get_completion_percentage(self) -> float:
        """Get scan completion percentage"""
        with QMutexLocker(self.mutex):
            if self.pixel_counts is None:
                return 0.0

            filled_pixels = np.sum(self.pixel_counts > 0)
            total_pixels = self.pixel_counts.size
            return (filled_pixels / total_pixels) * 100.0

class LineReconstructor(QObject):
    """Real-time line scan reconstruction - mirrors ImageReconstructor pattern"""

    # Signals
    line_updated = pyqtSignal(object, object)  # positions array, signals array
    point_added = pyqtSignal(int, float, float)  # index, position, value
    scan_completed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.line_params = None
        self.positions = None
        self.signals = None
        self.point_counts = None
        self.mutex = QMutex()
        self._current_point_index = None

    def initialize_scan(self, line_params: LineScanParameters):
        """Initialize for new line scan"""
        with QMutexLocker(self.mutex):
            self.line_params = line_params

            # Pre-allocate arrays
            self.positions = np.full(line_params.num_points, np.nan, dtype=np.float64)
            self.signals = np.full(line_params.num_points, np.nan, dtype=np.float64)
            self.point_counts = np.zeros(line_params.num_points, dtype=int)

            # Generate expected positions
            forward = line_params.scan_end >= line_params.scan_start
            step = line_params.step_size if forward else -line_params.step_size

            for i in range(line_params.num_points):
                self.positions[i] = line_params.scan_start + i * step

            # Emit initial empty line
            # self.line_updated.emit(self.positions.copy(), self.signals.copy())

    @pyqtSlot(int)
    def set_current_point(self, point_index: int):
        """Set current point index for data acquisition"""
        with QMutexLocker(self.mutex):
            self._current_point_index = point_index

    @pyqtSlot()
    def clear_current_point(self):
        """Clear current point when scan stops"""
        with QMutexLocker(self.mutex):
            self._current_point_index = None

    @pyqtSlot(object)
    def add_data_point(self, data_point: DataPoint):
        """Add data point to line reconstruction using scan grid coordinates"""
        if not self.line_params:
            return

        try:
            with QMutexLocker(self.mutex):
                # Check if we have valid point index
                if self._current_point_index is None:
                    return

                idx = self._current_point_index

                # Bounds check
                if not (0 <= idx < self.line_params.num_points):
                    return

                # Extract position based on scan axis
                if self.line_params.scan_axis == "Z":
                    actual_position = data_point.z_pos
                elif self.line_params.scan_axis == "X":
                    actual_position = data_point.x_pos
                else:
                    actual_position = data_point.y_pos

                # Update signal value (running average like ImageReconstructor)
                current_count = self.point_counts[idx]
                if current_count == 0 or np.isnan(self.signals[idx]):
                    self.signals[idx] = data_point.current
                else:
                    old_value = self.signals[idx]
                    self.signals[idx] = (old_value * current_count + data_point.current) / (current_count + 1)

                self.point_counts[idx] += 1

                # Emit updates
                self.point_added.emit(idx, self.positions[idx], self.signals[idx])

        except Exception as e:
            print(f"Error adding data point to line: {e}")

    @pyqtSlot()
    def emit_full_line(self):
        """Emit complete current line data"""
        with QMutexLocker(self.mutex):
            if self.positions is not None and self.signals is not None:
                # Create mask for valid data points
                valid_mask = ~np.isnan(self.signals)
                valid_positions = self.positions[valid_mask]
                valid_signals = self.signals[valid_mask]

                if len(valid_positions) > 0:
                    self.line_updated.emit(valid_positions, valid_signals)

    def get_completion_percentage(self) -> float:
        """Get scan completion percentage"""
        with QMutexLocker(self.mutex):
            if self.point_counts is None:
                return 0.0
            filled_points = np.sum(self.point_counts > 0)
            total_points = len(self.point_counts)
            return (filled_points / total_points) * 100.0 if total_points > 0 else 0.0

    def complete_scan(self):
        """Mark scan as complete and emit final data"""
        self.emit_full_line()
        self.scan_completed.emit()

class ScanController(QObject):
    """Controls scan execution and movement patterns"""

    # Signals
    scan_started = pyqtSignal()
    scan_completed = pyqtSignal()
    scan_progress = pyqtSignal(int, int)  # current_pixel, total_pixels
    movement_command = pyqtSignal(str)  # MQTT command to send
    status_update = pyqtSignal(str)
    initialization_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.scan_params = None
        self.scan_state = ScanState.IDLE
        self.current_pixel = 0
        self.scan_pattern = []
        self.movement_timer = QTimer()
        self.movement_timer.timeout.connect(self.execute_next_movement)

        # Step-and-stop timing
        self.step_timer = QTimer()
        self.step_timer.setSingleShot(True)
        self.step_timer.timeout.connect(self.on_dwell_completed)

        # --- settle gating state & timer ---
        self.settle_timer = QTimer()
        self.settle_timer.timeout.connect(self._on_settle_check)
        self._waiting_to_settle = False
        self._settle_start_monotonic = 0.0
        self._settle_consecutive = 0
        self._current_target = (None, None)
        self._data_processor = None  # set via set_data_processor(...)

        # Detector lag timer (cancelable)
        self._detector_lag_timer = QTimer()
        self._detector_lag_timer.setSingleShot(True)
        self._detector_lag_timer.timeout.connect(self._begin_dwell_after_detector_lag)
        self._in_detector_lag = False
        self._detector_lag_s = 0.350  # Default, will be updated from UI

        # Add initialization movement state
        self._initializing_scan = False
        self._init_start_time = 0.0
        self._init_timeout_s = 60.0  # Fixed 60 second timeout

        # Timer for initialization checks
        self.init_timer = QTimer()
        self.init_timer.timeout.connect(self._on_init_position_check)

    def set_data_processor(self, dp):
        """Inject DataProcessor so we can read live positions."""
        self._data_processor = dp

    def set_scan_parameters(self, scan_params: ScanParameters):
        """Set scan parameters"""
        self.scan_params = scan_params
        self.generate_scan_pattern()

    def generate_scan_pattern(self):
        """Generate scan pattern handling any scan direction"""
        if not self.scan_params:
            return

        self.scan_pattern = []

        # Determine scan direction
        x_forward = self.scan_params.x_end >= self.scan_params.x_start
        y_forward = self.scan_params.y_end >= self.scan_params.y_start

        # Calculate actual step with direction
        x_step = self.scan_params.x_step_input if x_forward else -self.scan_params.x_step_input
        y_step = self.scan_params.y_step_input if y_forward else -self.scan_params.y_step_input

        for y_idx in range(self.scan_params.y_pixels):
            y_pos = self.scan_params.y_start + y_idx * y_step

            # Determine X scan order based on pattern
            if self.scan_params.pattern == "snake" and y_idx % 2 == 1:
                # Snake: reverse X order for odd rows
                x_indices = range(self.scan_params.x_pixels - 1, -1, -1)
            else:
                x_indices = range(self.scan_params.x_pixels)

            for x_idx_in_row in x_indices:
                x_pos = self.scan_params.x_start + x_idx_in_row * x_step
                # Store grid position for image reconstruction
                self.scan_pattern.append((x_pos, y_pos, x_idx_in_row, y_idx))

    def get_scan_preview_path(self) -> List[Tuple[float, float]]:
        """Get scan path for preview visualization"""
        if not self.scan_params:
            return []

        path = []

        # Determine scan direction
        x_forward = self.scan_params.x_end >= self.scan_params.x_start
        y_forward = self.scan_params.y_end >= self.scan_params.y_start

        # Calculate actual step with direction
        x_step = self.scan_params.x_step_input if x_forward else -self.scan_params.x_step_input
        y_step = self.scan_params.y_step_input if y_forward else -self.scan_params.y_step_input

        for y_idx in range(self.scan_params.y_pixels):
            y_pos = self.scan_params.y_start + y_idx * y_step

            # Determine X direction based on pattern
            if self.scan_params.pattern == "snake" and y_idx % 2 == 1:
                # Snake pattern: reverse direction for odd rows
                x_range = range(self.scan_params.x_pixels - 1, -1, -1)
            else:
                # Normal direction
                x_range = range(self.scan_params.x_pixels)

            for x_idx in x_range:
                x_pos = self.scan_params.x_start + x_idx * x_step
                path.append((x_pos, y_pos))

        return path

    def _start_settle_wait(self, x_tgt: float, y_tgt: float):
        """Begin non-blocking 'in-position' wait before dwell."""

        # Reset any detector lag state
        self._in_detector_lag = False
        self._detector_lag_timer.stop()

        self._current_target = (x_tgt, y_tgt)
        self._settle_consecutive = 0
        self._waiting_to_settle = True
        self._settle_start_monotonic = time.monotonic()
        self.status_update.emit(f"Settling to ({x_tgt:.0f}, {y_tgt:.0f}) nm ...")
        self.settle_timer.start(20)  # 50 Hz check rate

    def set_detector_lag(self, lag_seconds: float):
        """Set detector lag time in seconds"""
        self._detector_lag_s = lag_seconds

    def _start_detector_lag_wait(self):
        if self._in_detector_lag:
            return
        if self._detector_lag_s <= 0:
            self._start_dwell()
            return
        self._in_detector_lag = True
        self.status_update.emit(f"Waiting {self._detector_lag_s:.3f}s for detector stabilization...")
        self._detector_lag_timer.start(int(self._detector_lag_s * 1000))

    def _begin_dwell_after_detector_lag(self):
        """Begin actual dwell after detector lag period."""
        self._in_detector_lag = False
        self._start_dwell()  # Existing method

    def _start_dwell(self):
        """Start the pixel dwell once we're settled (or timed out)."""
        self.settle_timer.stop()
        self._waiting_to_settle = False
        dwell_ms = int(self.scan_params.dwell_time * 1000)
        self.step_timer.start(dwell_ms)

    @pyqtSlot()
    def _on_settle_check(self):
        """Periodic 'are we on target?' check."""
        if not self._waiting_to_settle or self._data_processor is None:
            self.settle_timer.stop()
            return

        x, y, z, r, tx, ty, tz, tr, now = self._data_processor.get_position_snapshot()
        # Require fresh XY (your DataProcessor already uses 0.2 s freshness elsewhere)
        if x is None or y is None or tx is None or ty is None:
            return
        if abs(now - tx) > 0.300 or abs(now - ty) > 0.300:
            return

        dx = abs(x - self._current_target[0])
        dy = abs(y - self._current_target[1])

        if (dx <= self.scan_params.pos_tol_x_nm and
                dy <= self.scan_params.pos_tol_y_nm):
            self._settle_consecutive += 1
        else:
            self._settle_consecutive = 0

        # Success path: enough consecutive in-tolerance samples
        if self._settle_consecutive >= self.scan_params.settle_required_samples:
            self.settle_timer.stop()
            self._waiting_to_settle = False
            self.status_update.emit("In position. Waiting for detector...")
            self._start_detector_lag_wait()
            return

        # Timeout path
        if (time.monotonic() - self._settle_start_monotonic) >= self.scan_params.settle_timeout_s:
            self.settle_timer.stop()
            self._waiting_to_settle = False
            self.status_update.emit(f"Settle timeout (Δ≈{dx:.0f},{dy:.0f} nm). Proceeding.")
            self._start_detector_lag_wait()
            return

    def start_scan(self):
        """Start scan - main entry point that moves to start position first"""
        if not self.scan_params:
            self.status_update.emit("ERROR: No scan parameters set")
            return

        self.scan_state = ScanState.SCANNING
        self.scan_started.emit()

        # Move to start position first, then begin scan pattern
        self._move_to_start_position()

    def _move_to_start_position(self):
        """Move all axes to scan start position before beginning scan pattern"""
        self._initializing_scan = True
        self._init_start_time = time.monotonic()

        start_x = self.scan_params.x_start
        start_y = self.scan_params.y_start

        self.status_update.emit(f"Moving to scan start position ({start_x:.0f}, {start_y:.0f}) nm...")

        # Send movement commands to start position
        self.movement_command.emit(f"MOVE/X/{start_x:.0f}")
        self.movement_command.emit(f"MOVE/Y/{start_y:.0f}")

        # Start checking for arrival at start position
        self.init_timer.start(50)  # Check every 50ms

    def start_step_stop_scan(self):
        """Start step-and-stop scan"""
        self.status_update.emit("Starting step-and-stop scan")
        self.current_pixel = 0
        self.execute_next_movement()

    def start_continuous_scan(self):
        """Start continuous scan"""
        self.status_update.emit("Starting continuous scan")
        # For continuous, we send move commands with calculated timing
        interval_ms = int(1000 / 10)  # 10 Hz movement updates
        self.movement_timer.start(interval_ms)

    @pyqtSlot()
    def _on_init_position_check(self):
        """Check if we've reached the scan start position"""
        if not self._initializing_scan or self._data_processor is None:
            self.init_timer.stop()
            return

        # Get current position
        x, y, z, r, tx, ty, tz, tr, now = self._data_processor.get_position_snapshot()

        # Check if we have fresh position data
        if x is None or y is None or tx is None or ty is None:
            return
        if abs(now - tx) > 0.500 or abs(now - ty) > 0.500:  # 500ms freshness
            return

        # Check if we're at start position (use larger tolerance for initial move)
        start_x = self.scan_params.x_start
        start_y = self.scan_params.y_start

        init_tol_x = self.scan_params.pos_tol_x_nm
        init_tol_y = self.scan_params.pos_tol_y_nm

        dx = abs(x - start_x)
        dy = abs(y - start_y)

        if dx <= init_tol_x and dy <= init_tol_y:
            # Successfully reached start position
            self.init_timer.stop()
            self._initializing_scan = False
            self.status_update.emit("Reached scan start position. Beginning scan pattern...")

            # Now start the actual scan pattern
            if self.scan_params.mode == ScanMode.STEP_STOP:
                self.start_step_stop_scan()
            else:
                self.start_continuous_scan()
            return

        # Check for timeout
        elapsed = time.monotonic() - self._init_start_time
        if elapsed >= self._init_timeout_s:
            # TIMEOUT ERROR - Stop the scan
            self.init_timer.stop()
            self._initializing_scan = False

            error_msg = (f"ERROR: Cannot reach scan start position after {elapsed:.1f}s. "
                         f"Current position error: ΔX={dx:.0f}nm, ΔY={dy:.0f}nm. "
                         f"Scan aborted.")

            self.status_update.emit(error_msg)

            # Emit error signal to show message box in main window
            if hasattr(self, 'initialization_error'):
                self.initialization_error.emit(error_msg)

            # Stop the scan completely
            self.scan_state = ScanState.IDLE
            self.scan_completed.emit()  # This will reset UI state
            return

        # Still moving - update status every 5 seconds
        if int(elapsed * 2) % 10 == 0:  # Every 5 seconds
            remaining = self._init_timeout_s - elapsed
            self.status_update.emit(
                f"Moving to start... {elapsed:.1f}s elapsed, {remaining:.1f}s remaining "
                f"(ΔX={dx:.0f}nm, ΔY={dy:.0f}nm)"
            )

    @pyqtSlot()
    def execute_next_movement(self):
        """Execute next movement in scan pattern"""
        if (self.scan_state != ScanState.SCANNING or
                self.current_pixel >= len(self.scan_pattern)):
            self.complete_scan()
            return

        x_pos, y_pos, x_idx, y_idx = self.scan_pattern[self.current_pixel]

        # Send pixel coordinates to image reconstructor using thread-safe method
        if self.image_reconstructor:
            self.image_reconstructor.set_current_pixel(x_idx, y_idx)

        # Send movement commands
        self.movement_command.emit(f"MOVE/X/{x_pos:.0f}")
        self.movement_command.emit(f"MOVE/Y/{y_pos:.0f}")

        self.scan_progress.emit(self.current_pixel + 1, len(self.scan_pattern))

        if self.scan_params.mode == ScanMode.STEP_STOP:
            self._start_settle_wait(x_pos, y_pos)
        else:
            self.current_pixel += 1

    @pyqtSlot()
    def on_dwell_completed(self):
        """Called when dwell time is completed in step-stop mode"""
        self.current_pixel += 1
        self.execute_next_movement()

    @pyqtSlot()
    def pause_scan(self):
        """Pause the scan"""
        if self.scan_state == ScanState.SCANNING:
            self.scan_state = ScanState.PAUSED
            self.movement_timer.stop()
            self.step_timer.stop()
            self.settle_timer.stop()
            self.status_update.emit("Scan paused")

    @pyqtSlot()
    def resume_scan(self):
        """Resume the scan"""
        if self.scan_state == ScanState.PAUSED:
            self.scan_state = ScanState.SCANNING
            if self.scan_params.mode == ScanMode.STEP_STOP:
                self.execute_next_movement()
            else:
                self.movement_timer.start()
            self.status_update.emit("Scan resumed")

    @pyqtSlot()
    def stop_scan(self):
        """Stop the scan"""
        self.scan_state = ScanState.STOPPING
        self.movement_timer.stop()
        self.step_timer.stop()
        self.init_timer.stop()  # Also stop initialization timer
        self.settle_timer.stop()
        self._detector_lag_timer.stop()
        self._in_detector_lag = False
        self._initializing_scan = False  # Reset initialization state

        # Send stop commands
        self.movement_command.emit("STOP/X")
        self.movement_command.emit("STOP/Y")

        self.complete_scan()

    def complete_scan(self):
        """Complete the scan"""
        self.scan_state = ScanState.IDLE
        self.movement_timer.stop()
        self.step_timer.stop()

        # Clear pixel indices in reconstructor
        if self.image_reconstructor:
            self.image_reconstructor.clear_current_pixel()

        self.scan_completed.emit()
        self.status_update.emit("Scan completed")

class ScanControlWidget(QWidget):
    """Widget for scan parameter control"""

    # Signals
    scan_parameters_changed = pyqtSignal(object)  # ScanParameters
    start_scan_requested = pyqtSignal()
    pause_scan_requested = pyqtSignal()
    stop_scan_requested = pyqtSignal()
    scan_path_preview = pyqtSignal(list)  # List of (x, y) tuples

    def __init__(self):
        super().__init__()
        self._updating = False
        self.setup_ui()
        self.connect_signals()
        QTimer.singleShot(100, self.update_calculations)

    def setup_ui(self):
        """Setup the compact 2D scan control UI"""
        layout = QVBoxLayout()
        layout.setSpacing(5)

        # Single parameters group
        params_group = QGroupBox("2D Scan Parameters")
        params_layout = QGridLayout()
        params_layout.setSpacing(5)

        # Row 0: X axis parameters
        params_layout.addWidget(QLabel("X:"), 0, 0)

        params_layout.addWidget(QLabel("Start"), 0, 1)
        self.x_start_spin = QDoubleSpinBox()
        self.x_start_spin.setRange(-20000000, 20000000)
        self.x_start_spin.setValue(0)
        self.x_start_spin.setDecimals(0)
        self.x_start_spin.setMaximumWidth(100)
        self.x_start_spin.setGroupSeparatorShown(True)
        params_layout.addWidget(self.x_start_spin, 0, 2)

        params_layout.addWidget(QLabel("End"), 0, 3)
        self.x_end_spin = QDoubleSpinBox()
        self.x_end_spin.setRange(-20000000, 20000000)
        self.x_end_spin.setValue(10000)
        self.x_end_spin.setDecimals(0)
        self.x_end_spin.setMaximumWidth(100)
        self.x_end_spin.setGroupSeparatorShown(True)
        params_layout.addWidget(self.x_end_spin, 0, 4)

        params_layout.addWidget(QLabel("Pixels"), 0, 5)
        self.x_pixels_spin = QSpinBox()
        self.x_pixels_spin.setRange(2, 2000)
        self.x_pixels_spin.setValue(100)
        self.x_pixels_spin.setMaximumWidth(60)
        self.x_pixels_spin.setGroupSeparatorShown(True)
        params_layout.addWidget(self.x_pixels_spin, 0, 6)

        params_layout.addWidget(QLabel("Step"), 0, 7)
        self.x_step_spin = QDoubleSpinBox()
        self.x_step_spin.setRange(1, 1000000)
        self.x_step_spin.setDecimals(0)
        self.x_step_spin.setValue(101)
        self.x_step_spin.setMaximumWidth(100)
        self.x_step_spin.setGroupSeparatorShown(True)
        params_layout.addWidget(self.x_step_spin, 0, 8)

        # Row 1: Y axis parameters
        params_layout.addWidget(QLabel("Y:"), 1, 0)

        params_layout.addWidget(QLabel("Start"), 1, 1)
        self.y_start_spin = QDoubleSpinBox()
        self.y_start_spin.setRange(-20000000, 20000000)
        self.y_start_spin.setValue(0)
        self.y_start_spin.setDecimals(0)
        self.y_start_spin.setMaximumWidth(100)
        self.y_start_spin.setGroupSeparatorShown(True)
        params_layout.addWidget(self.y_start_spin, 1, 2)

        params_layout.addWidget(QLabel("End"), 1, 3)
        self.y_end_spin = QDoubleSpinBox()
        self.y_end_spin.setRange(-20000000, 20000000)
        self.y_end_spin.setValue(10000)
        self.y_end_spin.setDecimals(0)
        self.y_end_spin.setMaximumWidth(100)
        self.y_end_spin.setGroupSeparatorShown(True)
        params_layout.addWidget(self.y_end_spin, 1, 4)

        params_layout.addWidget(QLabel("Pixels"), 1, 5)
        self.y_pixels_spin = QSpinBox()
        self.y_pixels_spin.setRange(2, 2000)
        self.y_pixels_spin.setValue(100)
        self.y_pixels_spin.setMaximumWidth(60)
        self.y_pixels_spin.setGroupSeparatorShown(True)
        params_layout.addWidget(self.y_pixels_spin, 1, 6)

        params_layout.addWidget(QLabel("Step"), 1, 7)
        self.y_step_spin = QDoubleSpinBox()
        self.y_step_spin.setRange(1, 1000000)
        self.y_step_spin.setDecimals(0)
        self.y_step_spin.setValue(101)
        self.y_step_spin.setMaximumWidth(100)
        self.y_step_spin.setGroupSeparatorShown(True)
        params_layout.addWidget(self.y_step_spin, 1, 8)

        # Row 2: Effective FOV display
        params_layout.addWidget(QLabel("Effective FOV:"), 2, 0, 1, 2)
        self.effective_fov_label = QLabel("9999×9999 nm")
        self.effective_fov_label.setStyleSheet("QLabel { font-weight: bold; background-color: #e8f4e8; padding: 2px; }")
        params_layout.addWidget(self.effective_fov_label, 2, 2, 1, 7)

        # Row 3: Mode and pattern
        params_layout.addWidget(QLabel("Mode:"), 3, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Step-Stop", ScanMode.STEP_STOP)
        self.mode_combo.addItem("Continuous", ScanMode.CONTINUOUS)
        params_layout.addWidget(self.mode_combo, 3, 1, 1, 3)

        params_layout.addWidget(QLabel("Pattern:"), 3, 4, 1, 2)
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItem("Snake", "snake")
        self.pattern_combo.addItem("Raster", "raster")
        params_layout.addWidget(self.pattern_combo, 3, 6, 1, 3)

        # Row 4: Timing parameters (dwell and speed)
        params_layout.addWidget(QLabel("Dwell Time:"), 4, 0, 1, 2)
        self.dwell_time_spin = QDoubleSpinBox()
        self.dwell_time_spin.setRange(0.001, 10.0)
        self.dwell_time_spin.setSingleStep(0.1)
        self.dwell_time_spin.setValue(0.75)
        self.dwell_time_spin.setSuffix(" s")
        params_layout.addWidget(self.dwell_time_spin, 4, 2, 1, 2)

        params_layout.addWidget(QLabel("Scan Speed:"), 4, 4, 1, 2)
        self.scan_speed_spin = QDoubleSpinBox()
        self.scan_speed_spin.setRange(1.0, 10000.0)
        self.scan_speed_spin.setValue(1000.0)
        self.scan_speed_spin.setSuffix(" nm/s")
        params_layout.addWidget(self.scan_speed_spin, 4, 6, 1, 3)

        # Row 5: Total pixels and estimated time
        params_layout.addWidget(QLabel("Total Pixels:"), 5, 0, 1, 2)
        self.total_pixels_label = QLabel("10,000")
        self.total_pixels_label.setStyleSheet("QLabel { font-weight: bold; }")
        params_layout.addWidget(self.total_pixels_label, 5, 2, 1, 2)

        params_layout.addWidget(QLabel("Est. Time:"), 5, 4, 1, 2)
        self.estimated_time_label = QLabel("2h 5m")
        self.estimated_time_label.setStyleSheet("QLabel { font-weight: bold; color: #0066cc; }")
        params_layout.addWidget(self.estimated_time_label, 5, 6, 1, 3)

        # Set column stretches
        for col in range(9):
            if col in [2, 4, 6, 8]:  # Input columns
                params_layout.setColumnStretch(col, 1)

        params_group.setLayout(params_layout)
        params_group.setMaximumHeight(200)
        layout.addWidget(params_group)

        # Add this after the existing params_group in setup_ui()

        # Z-Series Parameters Group
        z_series_group = QGroupBox("Z-Series Parameters")
        z_series_layout = QGridLayout()
        z_series_layout.setSpacing(5)

        # Row 0: Enable checkbox and base Z
        self.z_series_enable_cb = QCheckBox("Enable Z-Series")
        self.z_series_enable_cb.stateChanged.connect(self.on_z_series_toggled)
        z_series_layout.addWidget(self.z_series_enable_cb, 0, 0, 1, 2)

        z_series_layout.addWidget(QLabel("Base Z:"), 0, 2)
        self.base_z_spin = QDoubleSpinBox()
        self.base_z_spin.setRange(-20000000, 20000000)
        self.base_z_spin.setValue(0)
        self.base_z_spin.setDecimals(0)
        self.base_z_spin.setSuffix(" nm")
        self.base_z_spin.setMaximumWidth(100)
        self.base_z_spin.setGroupSeparatorShown(True)
        z_series_layout.addWidget(self.base_z_spin, 0, 3)

        z_series_layout.addWidget(QLabel("X Comp Ratio:"), 0, 4)
        self.x_compensation_spin = QDoubleSpinBox()
        self.x_compensation_spin.setRange(-10.0, 10.0)
        self.x_compensation_spin.setValue(1.0)
        self.x_compensation_spin.setDecimals(8)
        self.x_compensation_spin.setSingleStep(0.00000001)
        self.x_compensation_spin.setToolTip("X movement per Z movement (1.0 = 45° beam, 0.0 = disabled)")
        self.x_compensation_spin.setMaximumWidth(120)
        z_series_layout.addWidget(self.x_compensation_spin, 0, 5)

        z_series_layout.addWidget(QLabel("Y Comp Ratio:"), 0, 6)
        self.y_compensation_spin = QDoubleSpinBox()
        self.y_compensation_spin.setRange(-10.0, 10.0)
        self.y_compensation_spin.setValue(0.0)
        self.y_compensation_spin.setDecimals(8)
        self.y_compensation_spin.setSingleStep(0.00000001)
        self.y_compensation_spin.setToolTip("Y movement per Z movement (0.0 = disabled)")
        self.y_compensation_spin.setMaximumWidth(120)
        z_series_layout.addWidget(self.y_compensation_spin, 0, 7)

        # Row 1: Z range and resolution
        z_series_layout.addWidget(QLabel("Z Start:"), 1, 0)
        self.z_start_spin = QDoubleSpinBox()
        self.z_start_spin.setRange(-20000000, 20000000)
        self.z_start_spin.setValue(0)
        self.z_start_spin.setDecimals(0)
        self.z_start_spin.setMaximumWidth(100)
        self.z_start_spin.setGroupSeparatorShown(True)
        z_series_layout.addWidget(self.z_start_spin, 1, 1)

        z_series_layout.addWidget(QLabel("Z End:"), 1, 2)
        self.z_end_spin = QDoubleSpinBox()
        self.z_end_spin.setRange(-20000000, 20000000)
        self.z_end_spin.setValue(1000)
        self.z_end_spin.setDecimals(0)
        self.z_end_spin.setMaximumWidth(100)
        self.z_end_spin.setGroupSeparatorShown(True)
        z_series_layout.addWidget(self.z_end_spin, 1, 3)

        z_series_layout.addWidget(QLabel("Z Numbers:"), 1, 4)
        self.z_numbers_spin = QSpinBox()
        self.z_numbers_spin.setRange(2, 1000)
        self.z_numbers_spin.setValue(10)
        self.z_numbers_spin.setMaximumWidth(60)
        z_series_layout.addWidget(self.z_numbers_spin, 1, 5)

        z_series_layout.addWidget(QLabel("Z Step:"), 1, 6)
        self.z_step_spin = QDoubleSpinBox()
        self.z_step_spin.setRange(1, 100000)
        self.z_step_spin.setValue(111)
        self.z_step_spin.setDecimals(0)
        self.z_step_spin.setSuffix(" nm")
        self.z_step_spin.setMaximumWidth(100)
        self.z_step_spin.setGroupSeparatorShown(True)
        z_series_layout.addWidget(self.z_step_spin, 1, 7)

        # Row 2: Calculated values and compensation
        z_series_layout.addWidget(QLabel("Z Eff. Range:"), 2, 0, 1, 2)
        self.z_effective_range_label = QLabel("999 nm")
        self.z_effective_range_label.setStyleSheet(
            "QLabel { font-weight: bold; background-color: #e8f4e8; padding: 2px; }")
        z_series_layout.addWidget(self.z_effective_range_label, 2, 2)

        z_series_layout.addWidget(QLabel("Total Images:"), 2, 5)
        self.total_images_label = QLabel("10")
        self.total_images_label.setStyleSheet("QLabel { font-weight: bold; }")
        z_series_layout.addWidget(self.total_images_label, 2, 6)

        # Row 3: Total time estimate with warning
        z_series_layout.addWidget(QLabel("Total Time Est:"), 3, 0, 1, 2)
        self.z_total_time_label = QLabel("20h 50m (10 × 2h 5m)")
        self.z_total_time_label.setStyleSheet("QLabel { font-weight: bold; color: #ff8800; }")
        z_series_layout.addWidget(self.z_total_time_label, 3, 2, 1, 6)

        z_series_group.setLayout(z_series_layout)
        z_series_group.setMaximumHeight(150)

        # Initially disable Z-series controls
        self.set_z_series_controls_enabled(False)

        layout.addWidget(z_series_group)

        # R-Series Parameters Group
        r_series_group = QGroupBox("R-Series Parameters")
        r_series_layout = QGridLayout()
        r_series_layout.setSpacing(5)

        # Row 0: Enable and mode
        self.r_series_enable_cb = QCheckBox("Enable R-Series")
        self.r_series_enable_cb.stateChanged.connect(self.on_r_series_toggled)
        r_series_layout.addWidget(self.r_series_enable_cb, 0, 0, 1, 2)

        r_series_layout.addWidget(QLabel("Mode:"), 0, 2)
        self.r_mode_combo = QComboBox()
        self.r_mode_combo.addItems(["Simple Rotation", "COR Transform"])
        self.r_mode_combo.currentTextChanged.connect(self.on_r_mode_changed)
        r_series_layout.addWidget(self.r_mode_combo, 0, 3, 1, 2)

        r_series_layout.addWidget(QLabel("Base R:"), 0, 5)
        self.base_r_spin = QDoubleSpinBox()
        self.base_r_spin.setRange(-360000000, 360000000)
        self.base_r_spin.setValue(0)
        self.base_r_spin.setDecimals(0)
        self.base_r_spin.setSuffix(" μdeg")
        self.base_r_spin.setMaximumWidth(100)
        self.base_r_spin.setGroupSeparatorShown(True)
        r_series_layout.addWidget(self.base_r_spin, 0, 6)

        # Row 1: R range and resolution
        r_series_layout.addWidget(QLabel("R Start:"), 1, 0)
        self.r_start_spin = QDoubleSpinBox()
        self.r_start_spin.setRange(-360000000, 360000000)
        self.r_start_spin.setValue(0)
        self.r_start_spin.setDecimals(0)
        self.r_start_spin.setMaximumWidth(100)
        self.r_start_spin.setGroupSeparatorShown(True)
        r_series_layout.addWidget(self.r_start_spin, 1, 1)

        r_series_layout.addWidget(QLabel("R End:"), 1, 2)
        self.r_end_spin = QDoubleSpinBox()
        self.r_end_spin.setRange(-360000000, 360000000)
        self.r_end_spin.setValue(360000)
        self.r_end_spin.setDecimals(0)
        self.r_end_spin.setMaximumWidth(100)
        self.r_end_spin.setGroupSeparatorShown(True)
        r_series_layout.addWidget(self.r_end_spin, 1, 3)

        r_series_layout.addWidget(QLabel("R Numbers:"), 1, 4)
        self.r_numbers_spin = QSpinBox()
        self.r_numbers_spin.setRange(2, 1000)
        self.r_numbers_spin.setValue(10)
        self.r_numbers_spin.setMaximumWidth(60)
        r_series_layout.addWidget(self.r_numbers_spin, 1, 5)

        r_series_layout.addWidget(QLabel("R Step:"), 1, 6)
        self.r_step_spin = QDoubleSpinBox()
        self.r_step_spin.setRange(1, 360000000)
        self.r_step_spin.setValue(40000)
        self.r_step_spin.setDecimals(0)
        self.r_step_spin.setSuffix(" μdeg")
        self.r_step_spin.setMaximumWidth(100)
        self.r_step_spin.setGroupSeparatorShown(True)
        r_series_layout.addWidget(self.r_step_spin, 1, 7)

        # Add this in the R-series UI setup (around row 2 or 3)
        r_series_layout.addWidget(QLabel("Transform:"), 2, 0)
        self.r_transform_combo = QComboBox()
        self.r_transform_combo.addItems(["Center Rotate", "AABB"])
        self.r_transform_combo.setCurrentText("Center Rotate")
        r_series_layout.addWidget(self.r_transform_combo, 2, 1)

        # Row 3: COR positions
        r_series_layout.addWidget(QLabel("COR X:"), 3, 0)
        self.cor_x_spin = QDoubleSpinBox()
        self.cor_x_spin.setRange(-20000000, 20000000)
        self.cor_x_spin.setValue(0)
        self.cor_x_spin.setDecimals(0)
        self.cor_x_spin.setSuffix(" nm")
        self.cor_x_spin.setMaximumWidth(100)
        self.cor_x_spin.setGroupSeparatorShown(True)
        r_series_layout.addWidget(self.cor_x_spin, 3, 1)

        r_series_layout.addWidget(QLabel("COR Y:"), 3, 2)
        self.cor_y_spin = QDoubleSpinBox()
        self.cor_y_spin.setRange(-20000000, 20000000)
        self.cor_y_spin.setValue(0)
        self.cor_y_spin.setDecimals(0)
        self.cor_y_spin.setSuffix(" nm")
        self.cor_y_spin.setMaximumWidth(100)
        self.cor_y_spin.setGroupSeparatorShown(True)
        r_series_layout.addWidget(self.cor_y_spin, 3, 3)

        r_series_layout.addWidget(QLabel("COR Base Z:"), 3, 4)
        self.cor_base_z_spin = QDoubleSpinBox()
        self.cor_base_z_spin.setRange(-20000000, 20000000)
        self.cor_base_z_spin.setValue(0)
        self.cor_base_z_spin.setDecimals(0)
        self.cor_base_z_spin.setSuffix(" nm")
        self.cor_base_z_spin.setMaximumWidth(100)
        self.cor_base_z_spin.setGroupSeparatorShown(True)
        self.cor_base_z_spin.setToolTip("Z height where COR X,Y are measured")
        r_series_layout.addWidget(self.cor_base_z_spin, 3, 5)

        # Row 4: COR compensation ratios
        r_series_layout.addWidget(QLabel("COR X Comp:"), 4, 0)
        self.cor_x_comp_spin = QDoubleSpinBox()
        self.cor_x_comp_spin.setRange(-10.0, 10.0)
        self.cor_x_comp_spin.setValue(1.0)
        self.cor_x_comp_spin.setDecimals(8)
        self.cor_x_comp_spin.setSingleStep(0.00000001)
        self.cor_x_comp_spin.setMaximumWidth(120)
        self.cor_x_comp_spin.setToolTip("COR X movement per Z movement (1.0 = 45° beam, 0.0 = disabled)")
        r_series_layout.addWidget(self.cor_x_comp_spin, 4, 1)

        r_series_layout.addWidget(QLabel("COR Y Comp:"), 4, 2)
        self.cor_y_comp_spin = QDoubleSpinBox()
        self.cor_y_comp_spin.setRange(-10.0, 10.0)
        self.cor_y_comp_spin.setValue(0.0)
        self.cor_y_comp_spin.setDecimals(8)
        self.cor_y_comp_spin.setSingleStep(0.00000001)
        self.cor_y_comp_spin.setMaximumWidth(120)
        self.cor_y_comp_spin.setToolTip("COR Y movement per Z movement (typically 0.0 for pure X-Z tilt)")
        r_series_layout.addWidget(self.cor_y_comp_spin, 4, 3)

        # Row 3: Calculated values and time
        r_series_layout.addWidget(QLabel("R Eff. Range:"), 5, 0, 1, 2)
        self.r_effective_range_label = QLabel("360000 μdeg")
        self.r_effective_range_label.setStyleSheet(
            "QLabel { font-weight: bold; background-color: #e8f4e8; padding: 2px; }")
        r_series_layout.addWidget(self.r_effective_range_label, 5, 2, 1, 2)

        r_series_layout.addWidget(QLabel("Total Images:"), 5, 4)
        self.r_total_images_label = QLabel("10")
        self.r_total_images_label.setStyleSheet("QLabel { font-weight: bold; }")
        r_series_layout.addWidget(self.r_total_images_label, 5, 5)

        # Row 4: Total time estimate
        r_series_layout.addWidget(QLabel("Total Time Est:"), 6, 0, 1, 2)
        self.r_total_time_label = QLabel("20h 50m (10 × 2h 5m)")
        self.r_total_time_label.setStyleSheet("QLabel { font-weight: bold; color: #ff8800; }")
        r_series_layout.addWidget(self.r_total_time_label, 6, 2, 1, 6)

        r_series_group.setLayout(r_series_layout)
        r_series_group.setMaximumHeight(180)

        # Initially disable R-series controls
        self.set_r_series_controls_enabled(False)

        layout.addWidget(r_series_group)

        # Control buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(5)

        self.start_btn = QPushButton("Start Scan")
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.start_btn.clicked.connect(self.start_scan_requested.emit)
        button_layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.pause_scan_requested.emit)
        button_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        self.stop_btn.clicked.connect(self.stop_scan_requested.emit)
        button_layout.addWidget(self.stop_btn)

        #button_layout.addStretch()

        layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        self.setLayout(layout)

    def connect_signals(self):
        """Connect internal signals"""
        # FOV change signals
        self.x_start_spin.valueChanged.connect(self.update_calculations)
        self.x_end_spin.valueChanged.connect(self.update_calculations)
        self.y_start_spin.valueChanged.connect(self.update_calculations)
        self.y_end_spin.valueChanged.connect(self.update_calculations)

        # Pixel/step bidirectional updates
        self.x_pixels_spin.valueChanged.connect(self.on_x_pixels_changed)
        self.y_pixels_spin.valueChanged.connect(self.on_y_pixels_changed)
        self.x_step_spin.valueChanged.connect(self.on_x_step_changed)
        self.y_step_spin.valueChanged.connect(self.on_y_step_changed)

        # Pattern and mode changes
        self.pattern_combo.currentTextChanged.connect(self.update_calculations)
        self.mode_combo.currentTextChanged.connect(self.update_mode_controls)
        self.dwell_time_spin.valueChanged.connect(self.update_calculations)
        self.scan_speed_spin.valueChanged.connect(self.update_calculations)

        # Add Z-series signal connections
        self.connect_z_series_signals()

        # Add R-series signal connections
        self.connect_r_series_signals()

    def on_x_pixels_changed(self):
        """Handle X pixel count change - update step size"""
        if self._updating:
            return
        self._updating = True

        x_fov = abs(self.x_end_spin.value() - self.x_start_spin.value())
        x_pixels = self.x_pixels_spin.value()

        if x_pixels > 1 and x_fov > 0:
            x_step = x_fov / (x_pixels - 1)
            x_step = max(1, round(x_step))  # Ensure minimum 1nm step
            self.x_step_spin.setValue(x_step)

        self._updating = False
        self.update_calculations()

    def on_y_pixels_changed(self):
        """Handle Y pixel count change - update step size"""
        if self._updating:
            return
        self._updating = True

        y_fov = abs(self.y_end_spin.value() - self.y_start_spin.value())
        y_pixels = self.y_pixels_spin.value()

        if y_pixels > 1 and y_fov > 0:
            y_step = y_fov / (y_pixels - 1)
            y_step = max(1, round(y_step))  # Ensure minimum 1nm step
            self.y_step_spin.setValue(y_step)

        self._updating = False
        self.update_calculations()

    def on_x_step_changed(self):
        """Handle X step size change - update pixel count"""
        if self._updating:
            return
        self._updating = True

        x_fov = abs(self.x_end_spin.value() - self.x_start_spin.value())
        x_step = self.x_step_spin.value()

        if x_step > 0:
            x_pixels = int(x_fov / x_step) + 1
            x_pixels = max(2, x_pixels)  # Ensure minimum 2 pixels
            self.x_pixels_spin.setValue(x_pixels)

        self._updating = False
        self.update_calculations()

    def on_y_step_changed(self):
        """Handle Y step size change - update pixel count"""
        if self._updating:
            return
        self._updating = True

        y_fov = abs(self.y_end_spin.value() - self.y_start_spin.value())
        y_step = self.y_step_spin.value()

        if y_step > 0:
            y_pixels = int(y_fov / y_step) + 1
            y_pixels = max(2, y_pixels)  # Ensure minimum 2 pixels
            self.y_pixels_spin.setValue(y_pixels)

        self._updating = False
        self.update_calculations()

    def update_calculations(self):
        """Update all calculated displays"""
        if self._updating:
            return

        # Calculate effective FOVs
        x_fov = abs(self.x_end_spin.value() - self.x_start_spin.value())
        y_fov = abs(self.y_end_spin.value() - self.y_start_spin.value())

        x_pixels = self.x_pixels_spin.value()
        y_pixels = self.y_pixels_spin.value()
        x_step = self.x_step_spin.value()
        y_step = self.y_step_spin.value()

        x_eff_fov = (x_pixels - 1) * x_step if x_pixels > 1 else 0
        y_eff_fov = (y_pixels - 1) * y_step if y_pixels > 1 else 0
        self.effective_fov_label.setText(f"{x_eff_fov:.0f}×{y_eff_fov:.0f} nm")

        # Color code effective FOV based on underscan
        x_underscan = max(0, x_fov - x_eff_fov)
        y_underscan = max(0, y_fov - y_eff_fov)
        max_underscan = max(x_underscan, y_underscan)

        if max_underscan < 50:
            self.effective_fov_label.setStyleSheet(
                "QLabel { font-weight: bold; background-color: #e8f4e8; padding: 2px; }")
        elif max_underscan < 200:
            self.effective_fov_label.setStyleSheet(
                "QLabel { font-weight: bold; background-color: #fff4e8; padding: 2px; }")
        else:
            self.effective_fov_label.setStyleSheet(
                "QLabel { font-weight: bold; background-color: #ffe8e8; padding: 2px; }")

        # Update total pixels
        total_pixels = x_pixels * y_pixels
        self.total_pixels_label.setText(f"{total_pixels:,} pixels")

        # Estimate scan time
        mode = self.mode_combo.currentData()
        if mode == ScanMode.STEP_STOP:
            dwell_time = self.dwell_time_spin.value()
            move_overhead = 0.2
            # Get detector lag from parent MainWindow
            detector_lag = 0.35  # Default value
            if self.parent() and hasattr(self.parent(), 'detector_lag_spin'):
                detector_lag = self.parent().detector_lag_spin.value()
            total_time = total_pixels * (dwell_time + move_overhead + detector_lag)
        else:
            scan_speed = self.scan_speed_spin.value()
            total_distance = x_eff_fov * y_pixels + y_eff_fov * (y_pixels - 1)
            total_time = total_distance / scan_speed if scan_speed > 0 else 0

        # Format time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        if hours > 0:
            time_str = f"{hours}h {minutes}m"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"

        self.estimated_time_label.setText(time_str)

        # Emit parameters
        self.emit_scan_parameters()

    def emit_scan_parameters(self):
        """Emit current scan parameters including Z-series"""
        # Create Z-series parameters
        z_series = ZSeriesParameters(
            enabled=self.z_series_enable_cb.isChecked(),
            z_start=self.z_start_spin.value(),
            z_end=self.z_end_spin.value(),
            z_numbers=self.z_numbers_spin.value(),
            z_step_input=self.z_step_spin.value(),
            x_compensation_ratio=self.x_compensation_spin.value(),
            y_compensation_ratio=self.y_compensation_spin.value(),
        )

        # Create R-series parameters
        r_series = RSeriesParameters(
            enabled=self.r_series_enable_cb.isChecked(),
            r_start=self.r_start_spin.value(),
            r_end=self.r_end_spin.value(),
            r_numbers=self.r_numbers_spin.value(),
            r_step_input=self.r_step_spin.value(),
            base_r_position=self.base_r_spin.value(),
            cor_enabled=(self.r_mode_combo.currentText() == "COR Transform"),
            cor_x=self.cor_x_spin.value(),
            cor_y=self.cor_y_spin.value(),
            cor_base_z=self.cor_base_z_spin.value(),
            cor_x_compensation_ratio=self.cor_x_comp_spin.value(),
            cor_y_compensation_ratio=self.cor_y_comp_spin.value(),
            cor_mode="center_rotate" if self.r_transform_combo.currentText() == "Center Rotate" else "aabb"
        )

        params = ScanParameters(
            x_start=self.x_start_spin.value(),
            x_end=self.x_end_spin.value(),
            y_start=self.y_start_spin.value(),
            y_end=self.y_end_spin.value(),
            x_pixels=self.x_pixels_spin.value(),
            y_pixels=self.y_pixels_spin.value(),
            x_step_input=self.x_step_spin.value(),
            y_step_input=self.y_step_spin.value(),
            mode=self.mode_combo.currentData(),
            dwell_time=self.dwell_time_spin.value(),
            scan_speed=self.scan_speed_spin.value(),
            pattern=self.pattern_combo.currentData(),
            base_z_position=self.base_z_spin.value(),
            base_r_position=self.base_r_spin.value(),
            z_series=z_series,
            r_series=r_series
        )
        self.scan_parameters_changed.emit(params)

        # Auto-update preview path when parameters change
        if hasattr(self, 'show_preview_cb'):
            self.preview_scan_path()

    def update_mode_controls(self):
        """Update controls based on scan mode"""
        is_step_stop = (self.mode_combo.currentData() == ScanMode.STEP_STOP)
        self.dwell_time_spin.setEnabled(is_step_stop)
        self.scan_speed_spin.setEnabled(not is_step_stop)
        self.update_calculations()

    def update_fov_calculations_only(self):
        """Update only FOV-related displays without triggering preview update"""
        # Calculate nominal FOVs (use absolute value)
        x_fov = abs(self.x_end_spin.value() - self.x_start_spin.value())
        y_fov = abs(self.y_end_spin.value() - self.y_start_spin.value())
        self.nominal_fov_label.setText(f"{x_fov:.0f} × {y_fov:.0f} nm")

        # Get current values
        x_pixels = self.x_pixels_spin.value()
        y_pixels = self.y_pixels_spin.value()
        x_step = int(self.x_step_spin.value())
        y_step = int(self.y_step_spin.value())

        # Calculate effective FOVs
        x_eff_fov = (x_pixels - 1) * x_step if x_pixels > 1 else 0
        y_eff_fov = (y_pixels - 1) * y_step if y_pixels > 1 else 0
        self.effective_fov_label.setText(f"{x_eff_fov:.0f} × {y_eff_fov:.0f} nm")

        # Color code based on underscan
        x_underscan = max(0, x_fov - x_eff_fov)
        y_underscan = max(0, y_fov - y_eff_fov)
        max_underscan = max(x_underscan, y_underscan)

        if max_underscan < 50:
            self.effective_fov_label.setStyleSheet("QLabel { background-color: #e8f4e8; padding: 3px; }")
        elif max_underscan < 200:
            self.effective_fov_label.setStyleSheet("QLabel { background-color: #fff4e8; padding: 3px; }")
        else:
            self.effective_fov_label.setStyleSheet("QLabel { background-color: #ffe8e8; padding: 3px; }")

        # Update time estimate
        self.update_time_estimate()

    def get_detector_lag(self) -> float:
        """Get detector lag value from parent MainWindow"""
        try:
            # Try to get from parent's detector_lag_spin
            parent = self.parent()
            while parent and not hasattr(parent, 'detector_lag_spin'):
                parent = parent.parent()

            if parent and hasattr(parent, 'detector_lag_spin'):
                return parent.detector_lag_spin.value()
        except:
            pass

        return 0.35  # Default detector lag in seconds

    def update_time_estimate(self):
        """Calculate and display time estimate only"""
        x_pixels = self.x_pixels_spin.value()
        y_pixels = self.y_pixels_spin.value()
        x_step = int(self.x_step_spin.value())
        y_step = int(self.y_step_spin.value())

        total_pixels = x_pixels * y_pixels
        self.total_pixels_label.setText(f"{total_pixels:,}")

        # Calculate effective FOVs for time estimate
        x_eff_fov = (x_pixels - 1) * x_step if x_pixels > 1 else 0
        y_eff_fov = (y_pixels - 1) * y_step if y_pixels > 1 else 0

        # Estimate scan time
        mode = self.mode_combo.currentData()
        if mode == ScanMode.STEP_STOP:
            dwell_time = self.dwell_time_spin.value()
            move_overhead = 0.2
            total_time = total_pixels * (dwell_time + move_overhead)
        else:
            scan_speed = self.scan_speed_spin.value()
            total_distance = x_eff_fov * y_pixels + y_eff_fov * (y_pixels - 1)
            total_time = total_distance / scan_speed if scan_speed > 0 else 0

        # Format time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        if hours > 0:
            time_str = f"{hours}h {minutes}m"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"

        self.estimated_time_label.setText(time_str)

    def on_z_series_toggled(self):
        """Handle Z-series enable/disable"""
        enabled = self.z_series_enable_cb.isChecked()

        # Disable R-series if Z-series is enabled (mutual exclusivity)
        if enabled and self.r_series_enable_cb.isChecked():
            self.r_series_enable_cb.setChecked(False)
            QMessageBox.information(self, "Info",
                                    "R-series has been disabled. Only one series type can be active at a time.")

        self.set_z_series_controls_enabled(enabled)
        self.update_calculations()

    def set_z_series_controls_enabled(self, enabled: bool):
        """Enable/disable Z-series controls"""
        self.base_z_spin.setEnabled(enabled)
        self.z_start_spin.setEnabled(enabled)
        self.z_end_spin.setEnabled(enabled)
        self.z_numbers_spin.setEnabled(enabled)
        self.z_step_spin.setEnabled(enabled)
        self.x_compensation_spin.setEnabled(enabled)
        self.y_compensation_spin.setEnabled(enabled)

    def connect_z_series_signals(self):
        """Connect Z-series specific signals"""
        self.z_start_spin.valueChanged.connect(self.update_z_calculations)
        self.z_end_spin.valueChanged.connect(self.update_z_calculations)
        self.z_numbers_spin.valueChanged.connect(self.on_z_numbers_changed)
        self.z_step_spin.valueChanged.connect(self.on_z_step_changed)
        self.x_compensation_spin.valueChanged.connect(self.update_calculations)
        self.y_compensation_spin.valueChanged.connect(self.update_calculations)

    def on_z_numbers_changed(self):
        """Handle Z numbers change - update step size"""
        if self._updating:
            return
        self._updating = True

        z_range = abs(self.z_end_spin.value() - self.z_start_spin.value())
        z_numbers = self.z_numbers_spin.value()

        if z_numbers > 1 and z_range > 0:
            z_step = z_range / (z_numbers - 1)
            z_step = max(1, round(z_step))
            self.z_step_spin.setValue(z_step)

        self._updating = False
        self.update_z_calculations()

    def on_z_step_changed(self):
        """Handle Z step change - update numbers"""
        if self._updating:
            return
        self._updating = True

        z_range = abs(self.z_end_spin.value() - self.z_start_spin.value())
        z_step = self.z_step_spin.value()

        if z_step > 0:
            z_numbers = int(z_range / z_step) + 1
            z_numbers = max(2, z_numbers)
            self.z_numbers_spin.setValue(z_numbers)

        self._updating = False
        self.update_z_calculations()

    def update_z_calculations(self):
        """Update Z-series calculated displays"""
        if self._updating:
            return

        # Calculate effective Z range
        z_numbers = self.z_numbers_spin.value()
        z_step = self.z_step_spin.value()
        z_eff_range = (z_numbers - 1) * z_step if z_numbers > 1 else 0
        self.z_effective_range_label.setText(f"{z_eff_range:.0f} nm")

        # Update total images
        self.total_images_label.setText(str(z_numbers))

        # Update total time estimate
        if self.z_series_enable_cb.isChecked():
            # Get base 2D scan time
            base_time_text = self.estimated_time_label.text()
            try:
                # Recalculate base time with current settings to ensure accuracy
                x_pixels = self.x_pixels_spin.value()
                y_pixels = self.y_pixels_spin.value()
                total_pixels = x_pixels * y_pixels
                mode = self.mode_combo.currentData()

                if mode == ScanMode.STEP_STOP:
                    dwell_time = self.dwell_time_spin.value()
                    move_overhead = 0.2
                    detector_lag = 0.35  # Default
                    if self.parent() and hasattr(self.parent(), 'detector_lag_spin'):
                        detector_lag = self.parent().detector_lag_spin.value()
                    base_seconds = total_pixels * (dwell_time + move_overhead + detector_lag)
                else:
                    # Continuous mode
                    x_step = self.x_step_spin.value()
                    y_step = self.y_step_spin.value()
                    x_eff_fov = (x_pixels - 1) * x_step if x_pixels > 1 else 0
                    y_eff_fov = (y_pixels - 1) * y_step if y_pixels > 1 else 0
                    scan_speed = self.scan_speed_spin.value()
                    total_distance = x_eff_fov * y_pixels + y_eff_fov * (y_pixels - 1)
                    base_seconds = total_distance / scan_speed if scan_speed > 0 else 0

                # Add inter-slice overhead (time to move Z and settle)
                z_move_overhead = 2.0  # seconds per Z movement
                total_seconds = base_seconds * z_numbers + (z_numbers - 1) * z_move_overhead

                total_hours = total_seconds // 3600
                total_minutes = (total_seconds % 3600) // 60

                if total_hours > 0:
                    time_str = f"{int(total_hours)}h {int(total_minutes)}m ({z_numbers} × {base_time_text})"
                else:
                    time_str = f"{int(total_minutes)}m ({z_numbers} × {base_time_text})"

                # Color code based on duration
                if total_hours > 10:
                    self.z_total_time_label.setStyleSheet("QLabel { font-weight: bold; color: #ff0000; }")
                elif total_hours > 5:
                    self.z_total_time_label.setStyleSheet("QLabel { font-weight: bold; color: #ff8800; }")
                else:
                    self.z_total_time_label.setStyleSheet("QLabel { font-weight: bold; color: #0066cc; }")

                self.z_total_time_label.setText(time_str)

            except:
                self.z_total_time_label.setText(f"{z_numbers} images")

        self.update_calculations()

    def preview_scan_path(self):
        """Preview the scan path"""
        # Get current parameters
        params = ScanParameters(
            x_start=self.x_start_spin.value(),
            x_end=self.x_end_spin.value(),
            y_start=self.y_start_spin.value(),
            y_end=self.y_end_spin.value(),
            x_pixels=self.x_pixels_spin.value(),
            y_pixels=self.y_pixels_spin.value(),
            x_step_input=self.x_step_spin.value(),
            y_step_input=self.y_step_spin.value(),
            mode=self.mode_combo.currentData(),
            dwell_time=self.dwell_time_spin.value(),
            scan_speed=self.scan_speed_spin.value(),
            pattern=self.pattern_combo.currentData()
        )

        # Generate preview path with proper direction handling
        preview_path = []

        # Determine scan direction
        x_forward = params.x_end >= params.x_start
        y_forward = params.y_end >= params.y_start

        # Calculate actual step with direction
        x_step = params.x_step_input if x_forward else -params.x_step_input
        y_step = params.y_step_input if y_forward else -params.y_step_input

        for y_idx in range(params.y_pixels):
            y_pos = params.y_start + y_idx * y_step

            # Determine X direction based on pattern
            if params.pattern == "snake" and y_idx % 2 == 1:
                # Snake pattern: reverse direction for odd rows
                x_range = range(params.x_pixels - 1, -1, -1)
            else:
                # Normal direction
                x_range = range(params.x_pixels)

            for x_idx in x_range:
                x_pos = params.x_start + x_idx * x_step
                preview_path.append((x_pos, y_pos))

        # Emit signal with preview path for display
        self.scan_path_preview.emit(preview_path)

    def set_scan_state(self, state: ScanState):
        """Update UI based on scan state"""
        # Disable parameter inputs during scan
        params_enabled = (state == ScanState.IDLE)

        # Disable all parameter controls
        self.x_start_spin.setEnabled(params_enabled)
        self.x_end_spin.setEnabled(params_enabled)
        self.y_start_spin.setEnabled(params_enabled)
        self.y_end_spin.setEnabled(params_enabled)
        self.x_pixels_spin.setEnabled(params_enabled)
        self.y_pixels_spin.setEnabled(params_enabled)
        self.x_step_spin.setEnabled(params_enabled)
        self.y_step_spin.setEnabled(params_enabled)
        self.mode_combo.setEnabled(params_enabled)
        self.pattern_combo.setEnabled(params_enabled)
        self.dwell_time_spin.setEnabled(params_enabled)
        self.scan_speed_spin.setEnabled(params_enabled)

        # Handle button states
        if state == ScanState.IDLE:
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
            self.start_btn.setText("Start Scan")

        elif state == ScanState.SCANNING:
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setVisible(True)

        elif state == ScanState.PAUSED:
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.start_btn.setText("Resume Scan")

    @pyqtSlot(int, int)
    def update_progress(self, current_pixel: int, total_pixels: int):
        """Update scan progress"""
        if total_pixels > 0:
            progress = int((current_pixel / total_pixels) * 100)
            self.progress_bar.setValue(progress)
            self.progress_bar.setFormat(f"{current_pixel}/{total_pixels} ({progress}%)")

    def on_r_series_toggled(self):
        """Handle R-series enable/disable"""
        enabled = self.r_series_enable_cb.isChecked()

        # Disable Z-series if R-series is enabled (mutual exclusivity)
        if enabled and self.z_series_enable_cb.isChecked():
            self.z_series_enable_cb.setChecked(False)
            QMessageBox.information(self, "Info",
                                    "Z-series has been disabled. Only one series type can be active at a time.")

        self.set_r_series_controls_enabled(enabled)
        self.update_calculations()

    def set_r_series_controls_enabled(self, enabled: bool):
        """Enable/disable R-series controls"""
        self.base_r_spin.setEnabled(enabled)
        self.r_start_spin.setEnabled(enabled)
        self.r_end_spin.setEnabled(enabled)
        self.r_numbers_spin.setEnabled(enabled)
        self.r_step_spin.setEnabled(enabled)
        self.r_mode_combo.setEnabled(enabled)

        # COR controls enabled based on mode
        cor_enabled = enabled and self.r_mode_combo.currentText() == "COR Transform"
        self.cor_x_spin.setEnabled(cor_enabled)
        self.cor_y_spin.setEnabled(cor_enabled)

    def on_r_mode_changed(self):
        """Handle R mode change"""
        if not self.r_series_enable_cb.isChecked():
            return

        cor_enabled = self.r_mode_combo.currentText() == "COR Transform"
        self.cor_x_spin.setEnabled(cor_enabled)
        self.cor_y_spin.setEnabled(cor_enabled)
        self.update_calculations()

    def connect_r_series_signals(self):
        """Connect R-series specific signals"""
        self.r_start_spin.valueChanged.connect(self.update_r_calculations)
        self.r_end_spin.valueChanged.connect(self.update_r_calculations)
        self.r_numbers_spin.valueChanged.connect(self.on_r_numbers_changed)
        self.r_step_spin.valueChanged.connect(self.on_r_step_changed)
        self.cor_x_spin.valueChanged.connect(self.update_calculations)
        self.cor_y_spin.valueChanged.connect(self.update_calculations)
        self.cor_base_z_spin.valueChanged.connect(self.update_calculations)
        self.cor_x_comp_spin.valueChanged.connect(self.update_calculations)
        self.cor_y_comp_spin.valueChanged.connect(self.update_calculations)

    def on_r_numbers_changed(self):
        """Handle R numbers change - update step size"""
        if self._updating:
            return
        self._updating = True

        r_range = abs(self.r_end_spin.value() - self.r_start_spin.value())
        r_numbers = self.r_numbers_spin.value()

        if r_numbers > 1 and r_range > 0:
            r_step = r_range / (r_numbers - 1)
            r_step = max(1, round(r_step))
            self.r_step_spin.setValue(r_step)

        self._updating = False
        self.update_r_calculations()

    def on_r_step_changed(self):
        """Handle R step change - update numbers"""
        if self._updating:
            return
        self._updating = True

        r_range = abs(self.r_end_spin.value() - self.r_start_spin.value())
        r_step = self.r_step_spin.value()

        if r_step > 0:
            r_numbers = int(r_range / r_step) + 1
            r_numbers = max(2, min(1000, r_numbers))
            self.r_numbers_spin.setValue(r_numbers)

        self._updating = False
        self.update_r_calculations()

    def update_r_calculations(self):
        """Update R-series calculated displays"""
        if self._updating:
            return

        # Calculate effective R range
        r_numbers = self.r_numbers_spin.value()
        r_step = self.r_step_spin.value()
        r_eff_range = (r_numbers - 1) * r_step if r_numbers > 1 else 0
        self.r_effective_range_label.setText(f"{r_eff_range:.0f} μdeg ({r_eff_range / 1000000:.1f}°)")

        # Update total images
        self.r_total_images_label.setText(str(r_numbers))

        # Update total time estimate
        if self.r_series_enable_cb.isChecked():
            # Get base 2D scan time
            base_time_text = self.estimated_time_label.text()
            try:
                # Recalculate base time accurately
                x_pixels = self.x_pixels_spin.value()
                y_pixels = self.y_pixels_spin.value()
                total_pixels = x_pixels * y_pixels
                mode = self.mode_combo.currentData()

                if mode == ScanMode.STEP_STOP:
                    dwell_time = self.dwell_time_spin.value()
                    move_overhead = 0.2
                    detector_lag = 0.35  # Default
                    if self.parent() and hasattr(self.parent(), 'detector_lag_spin'):
                        detector_lag = self.parent().detector_lag_spin.value()
                    base_seconds = total_pixels * (dwell_time + move_overhead + detector_lag)
                else:
                    # Continuous mode calculation
                    x_step = self.x_step_spin.value()
                    y_step = self.y_step_spin.value()
                    x_eff_fov = (x_pixels - 1) * x_step if x_pixels > 1 else 0
                    y_eff_fov = (y_pixels - 1) * y_step if y_pixels > 1 else 0
                    scan_speed = self.scan_speed_spin.value()
                    total_distance = x_eff_fov * y_pixels + y_eff_fov * (y_pixels - 1)
                    base_seconds = total_distance / scan_speed if scan_speed > 0 else 0

                # Add rotation overhead (time to rotate and settle per R position)
                rotation_overhead = 10.0  # seconds per rotation
                total_seconds = base_seconds * r_numbers + (r_numbers - 1) * rotation_overhead

                total_hours = total_seconds // 3600
                total_minutes = (total_seconds % 3600) // 60

                if total_hours > 0:
                    time_str = f"{int(total_hours)}h {int(total_minutes)}m ({r_numbers} × {base_time_text})"
                else:
                    time_str = f"{int(total_minutes)}m ({r_numbers} × {base_time_text})"

                # Color code
                if total_hours > 10:
                    self.r_total_time_label.setStyleSheet("QLabel { font-weight: bold; color: #ff0000; }")
                elif total_hours > 5:
                    self.r_total_time_label.setStyleSheet("QLabel { font-weight: bold; color: #ff8800; }")
                else:
                    self.r_total_time_label.setStyleSheet("QLabel { font-weight: bold; color: #0066cc; }")

                self.r_total_time_label.setText(time_str)
            except:
                self.r_total_time_label.setText(f"{r_numbers} images")

        self.update_calculations()

class ManualControlWidget(QWidget):
    """Widget for manual stage control"""

    # Signals
    move_requested = pyqtSignal(str, float)  # axis, position
    stop_all_requested = pyqtSignal()
    stop_axis_requested = pyqtSignal(str)  # specific axis

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """Setup the manual control UI"""
        layout = QVBoxLayout()

        # Axis control group
        axis_group = QGroupBox("Direct Axis Control")
        axis_layout = QGridLayout()

        # Create controls for each axis
        self.axis_controls = {}
        axes_config = [
            ('X', -20000000, 20000000, 'nm'),
            ('Y', -20000000, 20000000, 'nm'),
            ('Z', -20000000, 20000000, 'nm'),
            ('R', -360000000, 360000000, 'μdeg')
        ]

        for row, (axis, min_val, max_val, unit) in enumerate(axes_config):
            # Axis label
            axis_layout.addWidget(QLabel(f"{axis} ({unit}):"), row, 0)

            # Position input
            pos_spin = QDoubleSpinBox()
            pos_spin.setRange(min_val, max_val)
            pos_spin.setDecimals(0 if axis != 'R' else 0)
            pos_spin.setValue(0)
            pos_spin.setSingleStep(100 if axis != 'R' else 1000)
            pos_spin.setGroupSeparatorShown(True)
            axis_layout.addWidget(pos_spin, row, 1)

            # Move button
            move_btn = QPushButton(f"Move {axis}")
            move_btn.clicked.connect(lambda checked, a=axis: self.move_axis(a))
            axis_layout.addWidget(move_btn, row, 2)

            # Stop button for individual axis
            stop_btn = QPushButton(f"Stop")
            stop_btn.clicked.connect(lambda checked, a=axis: self.stop_axis(a))
            axis_layout.addWidget(stop_btn, row, 3)

            # Store references
            self.axis_controls[axis] = {
                'position': pos_spin,
                'move_btn': move_btn,
                'stop_btn': stop_btn
            }

        axis_group.setLayout(axis_layout)
        layout.addWidget(axis_group)

        # Emergency stop group
        stop_group = QGroupBox("Emergency Stop")
        stop_layout = QVBoxLayout()

        self.stop_all_btn = QPushButton("STOP ALL AXES")
        self.stop_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #ff0000;
            }
        """)
        self.stop_all_btn.clicked.connect(self.stop_all)
        stop_layout.addWidget(self.stop_all_btn)

        stop_group.setLayout(stop_layout)
        layout.addWidget(stop_group)

        # Movement status
        status_group = QGroupBox("Movement Status")
        status_layout = QGridLayout()

        status_layout.addWidget(QLabel("Status:"), 0, 0)
        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 3px; }")
        status_layout.addWidget(self.status_label, 0, 1)

        status_layout.addWidget(QLabel("Last Command:"), 1, 0)
        self.last_command_label = QLabel("None")
        self.last_command_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 3px; }")
        status_layout.addWidget(self.last_command_label, 1, 1)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        layout.addStretch()
        self.setLayout(layout)

    def move_axis(self, axis: str):
        """Request axis movement"""
        position = self.axis_controls[axis]['position'].value()
        self.move_requested.emit(axis, position)
        self.status_label.setText(f"Moving {axis}...")
        self.status_label.setStyleSheet("QLabel { background-color: #fffacd; padding: 3px; }")
        self.last_command_label.setText(f"MOVE/{axis}/{position:.0f}")

    def stop_axis(self, axis: str):
        """Stop specific axis"""
        self.stop_axis_requested.emit(axis)
        self.status_label.setText(f"Stopping {axis}")
        self.status_label.setStyleSheet("QLabel { background-color: #ffcccc; padding: 3px; }")
        self.last_command_label.setText(f"STOP/{axis}")

    def stop_all(self):
        """Emergency stop all axes"""
        self.stop_all_requested.emit()
        self.status_label.setText("EMERGENCY STOP")
        self.status_label.setStyleSheet("QLabel { background-color: #ff0000; color: white; padding: 3px; }")
        self.last_command_label.setText("STOP/ALL")

    def set_movement_complete(self):
        """Update UI when movement is complete"""
        self.status_label.setText("Idle")
        self.status_label.setStyleSheet("QLabel { background-color: #ccffcc; padding: 3px; }")

    def set_enabled(self, enabled: bool):
        """Enable/disable all controls"""
        for axis in self.axis_controls:
            self.axis_controls[axis]['position'].setEnabled(enabled)
            self.axis_controls[axis]['move_btn'].setEnabled(enabled)
            self.axis_controls[axis]['stop_btn'].setEnabled(enabled)
        # Stop button always enabled for safety
        self.stop_all_btn.setEnabled(True)

class LineScanControlWidget(QWidget):
    """Widget for line scan control - compact layout"""

    # Signals
    parameters_changed = pyqtSignal(object)  # LineScanParameters
    start_scan_requested = pyqtSignal()
    stop_scan_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._updating = False
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """Setup the line scan control UI - compact single group layout"""
        layout = QVBoxLayout()

        #Standalone Scan Mode Selection Group
        mode_group = QGroupBox("Scan Mode Selection")
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(10)

        mode_layout.addWidget(QLabel("Mode:"))
        self.z_mode_combo = QComboBox()
        self.z_mode_combo.addItems(["Line Scan", "Z-Scan", "R+Z Series"])
        self.z_mode_combo.currentTextChanged.connect(self.on_z_mode_changed)
        mode_layout.addWidget(self.z_mode_combo)

        mode_layout.addStretch()
        mode_group.setLayout(mode_layout)
        mode_group.setMaximumHeight(70)
        layout.addWidget(mode_group)

        # Single combined settings group
        settings_group = QGroupBox("Line Scan Settings")
        settings_layout = QGridLayout()
        settings_layout.setSpacing(5)

        # Row 0: Fixed axis configuration
        settings_layout.addWidget(QLabel("Fixed Axis:"), 0, 0)
        self.fixed_axis_combo = QComboBox()
        self.fixed_axis_combo.addItems(["X", "Y"])
        settings_layout.addWidget(self.fixed_axis_combo, 0, 1)

        settings_layout.addWidget(QLabel("Fixed Pos:"), 0, 2)
        self.fixed_position_spin = QDoubleSpinBox()
        self.fixed_position_spin.setRange(-20000000, 20000000)
        self.fixed_position_spin.setValue(0)
        self.fixed_position_spin.setDecimals(0)
        self.fixed_position_spin.setSuffix(" nm")
        settings_layout.addWidget(self.fixed_position_spin, 0, 3)

        # Row 1: Scan axis and range
        settings_layout.addWidget(QLabel("Scan Axis:"), 1, 0)
        self.scan_axis_label = QLabel("Y")
        settings_layout.addWidget(self.scan_axis_label, 1, 1)

        settings_layout.addWidget(QLabel("Start:"), 1, 2)
        self.scan_start_spin = QDoubleSpinBox()
        self.scan_start_spin.setRange(-20000000, 20000000)
        self.scan_start_spin.setValue(0)
        self.scan_start_spin.setDecimals(0)
        self.scan_start_spin.setGroupSeparatorShown(True)
        settings_layout.addWidget(self.scan_start_spin, 1, 3)

        settings_layout.addWidget(QLabel("End:"), 1, 4)
        self.scan_end_spin = QDoubleSpinBox()
        self.scan_end_spin.setRange(-20000000, 20000000)
        self.scan_end_spin.setValue(10000)
        self.scan_end_spin.setDecimals(0)
        self.scan_end_spin.setGroupSeparatorShown(True)
        settings_layout.addWidget(self.scan_end_spin, 1, 5)

        # Row 2: Resolution settings
        settings_layout.addWidget(QLabel("Points:"), 2, 0)
        self.num_points_spin = QSpinBox()
        self.num_points_spin.setRange(2, 10000)
        self.num_points_spin.setValue(100)
        settings_layout.addWidget(self.num_points_spin, 2, 1)

        settings_layout.addWidget(QLabel("Step:"), 2, 2)
        self.step_size_spin = QDoubleSpinBox()
        self.step_size_spin.setRange(1, 100000)
        self.step_size_spin.setValue(101)
        self.step_size_spin.setDecimals(0)
        self.step_size_spin.setSuffix(" nm")
        self.step_size_spin.setGroupSeparatorShown(True)
        settings_layout.addWidget(self.step_size_spin, 2, 3)

        settings_layout.addWidget(QLabel("Range:"), 2, 4)
        self.effective_range_label = QLabel("9999 nm")
        self.effective_range_label.setMinimumWidth(80)
        self.effective_range_label.setStyleSheet("QLabel { background-color: #e8f4e8; padding: 2px; }")
        settings_layout.addWidget(self.effective_range_label, 2, 5)

        # Row 3: Acquisition settings
        settings_layout.addWidget(QLabel("Dwell:"), 3, 0)
        self.dwell_time_spin = QDoubleSpinBox()
        self.dwell_time_spin.setRange(0.001, 10.0)
        self.dwell_time_spin.setValue(0.1)
        self.dwell_time_spin.setSingleStep(0.01)
        self.dwell_time_spin.setSuffix(" s")
        settings_layout.addWidget(self.dwell_time_spin, 3, 1)

        settings_layout.addWidget(QLabel("Est. Time:"), 3, 2)
        self.time_estimate_label = QLabel("10 s")
        self.time_estimate_label.setStyleSheet("QLabel { font-weight: bold; }")
        settings_layout.addWidget(self.time_estimate_label, 3, 3)

        # Set column stretches for better spacing
        settings_layout.setColumnStretch(1, 1)
        settings_layout.setColumnStretch(3, 1)
        settings_layout.setColumnStretch(5, 1)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Combined Z-Scan and R+Z Series Group
        z_series_group = QGroupBox("R+Z Series Parameters")
        z_series_layout = QGridLayout()
        z_series_layout.setSpacing(5)

        # Row 1: Fixed positions (used by both Z-Scan and R+Z)
        z_series_layout.addWidget(QLabel("Fixed X:"), 0, 0)
        self.z_fixed_x_spin = QDoubleSpinBox()
        self.z_fixed_x_spin.setRange(-20000000, 20000000)
        self.z_fixed_x_spin.setValue(5000)
        self.z_fixed_x_spin.setDecimals(0)
        self.z_fixed_x_spin.setSuffix(" nm")
        self.z_fixed_x_spin.setMaximumWidth(100)
        self.z_fixed_x_spin.setGroupSeparatorShown(True)
        z_series_layout.addWidget(self.z_fixed_x_spin, 0, 1)

        z_series_layout.addWidget(QLabel("Fixed Y:"), 0, 2)
        self.z_fixed_y_spin = QDoubleSpinBox()
        self.z_fixed_y_spin.setRange(-20000000, 20000000)
        self.z_fixed_y_spin.setValue(5000)
        self.z_fixed_y_spin.setDecimals(0)
        self.z_fixed_y_spin.setSuffix(" nm")
        self.z_fixed_y_spin.setMaximumWidth(100)
        self.z_fixed_y_spin.setGroupSeparatorShown(True)
        z_series_layout.addWidget(self.z_fixed_y_spin, 0, 3)

        z_series_layout.addWidget(QLabel("Base Z:"), 0, 4)
        self.z_base_z_spin = QDoubleSpinBox()
        self.z_base_z_spin.setRange(-20000000, 20000000)
        self.z_base_z_spin.setValue(0)
        self.z_base_z_spin.setDecimals(0)
        self.z_base_z_spin.setSuffix(" nm")
        self.z_base_z_spin.setMaximumWidth(100)
        self.z_base_z_spin.setGroupSeparatorShown(True)
        z_series_layout.addWidget(self.z_base_z_spin, 0, 5)

        # Row 2: Z parameters (for both Z-Scan and R+Z)
        z_series_layout.addWidget(QLabel("Z Start:"), 1, 0)
        self.z_start_spin = QDoubleSpinBox()
        self.z_start_spin.setRange(-20000000, 20000000)
        self.z_start_spin.setValue(0)
        self.z_start_spin.setDecimals(0)
        self.z_start_spin.setMaximumWidth(100)
        self.z_start_spin.setGroupSeparatorShown(True)
        z_series_layout.addWidget(self.z_start_spin, 1, 1)

        z_series_layout.addWidget(QLabel("Z End:"), 1, 2)
        self.z_end_spin = QDoubleSpinBox()
        self.z_end_spin.setRange(-20000000, 20000000)
        self.z_end_spin.setValue(500)
        self.z_end_spin.setDecimals(0)
        self.z_end_spin.setMaximumWidth(100)
        self.z_end_spin.setGroupSeparatorShown(True)
        z_series_layout.addWidget(self.z_end_spin, 1, 3)

        z_series_layout.addWidget(QLabel("Z Numbers:"), 1, 4)
        self.z_numbers_spin = QSpinBox()
        self.z_numbers_spin.setRange(2, 1000)
        self.z_numbers_spin.setValue(5)
        self.z_numbers_spin.setMaximumWidth(60)
        z_series_layout.addWidget(self.z_numbers_spin, 1, 5)

        z_series_layout.addWidget(QLabel("Z Step:"), 1, 6)
        self.z_step_spin = QDoubleSpinBox()
        self.z_step_spin.setRange(1, 100000)
        self.z_step_spin.setValue(125)
        self.z_step_spin.setDecimals(0)
        self.z_step_spin.setSuffix(" nm")
        self.z_step_spin.setMaximumWidth(100)
        self.z_step_spin.setGroupSeparatorShown(True)
        z_series_layout.addWidget(self.z_step_spin, 1, 7)

        # Row 3: Z dwell and X compensation
        z_series_layout.addWidget(QLabel("Z Dwell:"), 2, 0)
        self.z_dwell_spin = QDoubleSpinBox()
        self.z_dwell_spin.setRange(0.001, 10.0)
        self.z_dwell_spin.setValue(1.0)
        self.z_dwell_spin.setSingleStep(0.1)
        self.z_dwell_spin.setSuffix(" s")
        self.z_dwell_spin.setMaximumWidth(70)
        z_series_layout.addWidget(self.z_dwell_spin, 2, 1)

        z_series_layout.addWidget(QLabel("X Comp:"), 2, 2)
        self.z_x_compensation_spin = QDoubleSpinBox()
        self.z_x_compensation_spin.setRange(-10.0, 10.0)
        self.z_x_compensation_spin.setValue(1.0)
        self.z_x_compensation_spin.setDecimals(8)  # CHANGED from 3 to 8
        self.z_x_compensation_spin.setSingleStep(0.00000001)  # CHANGED for finer control
        self.z_x_compensation_spin.setMaximumWidth(120)  # CHANGED from 70 to fit more digits
        self.z_x_compensation_spin.setToolTip("X movement per Z movement (1.0 = 45° beam, 0.0 = disabled)")
        z_series_layout.addWidget(self.z_x_compensation_spin, 2, 3)

        z_series_layout.addWidget(QLabel("Y Comp:"), 2, 4)
        self.z_y_compensation_spin = QDoubleSpinBox()
        self.z_y_compensation_spin.setRange(-10.0, 10.0)
        self.z_y_compensation_spin.setValue(0.0)
        self.z_y_compensation_spin.setDecimals(8)  # CHANGED from 3 to 8
        self.z_y_compensation_spin.setSingleStep(0.00000001)  # CHANGED for finer control
        self.z_y_compensation_spin.setMaximumWidth(120)  # CHANGED from 70 to fit more digits
        self.z_y_compensation_spin.setToolTip("Y movement per Z movement (0.0 = disabled)")
        z_series_layout.addWidget(self.z_y_compensation_spin, 2, 5)

        z_series_layout.addWidget(QLabel("Z Eff. Range:"), 2, 6)
        self.z_effective_range_label = QLabel("500 nm")
        self.z_effective_range_label.setStyleSheet("QLabel { background-color: #e8f4e8; padding: 2px; }")
        z_series_layout.addWidget(self.z_effective_range_label, 2, 7, 1, 2)

        # Row 4: Separator for R+Z mode
        self.rz_separator_label = QLabel("─── R Parameters (for R+Z Series) ───")
        z_series_layout.addWidget(self.rz_separator_label, 3, 0, 1, 8)

        # Row 5: R parameters (only for R+Z mode)
        z_series_layout.addWidget(QLabel("Base R:"), 4, 4)
        self.rz_base_r_spin = QDoubleSpinBox()
        self.rz_base_r_spin.setRange(-360000000, 360000000)
        self.rz_base_r_spin.setValue(0)
        self.rz_base_r_spin.setDecimals(0)
        self.rz_base_r_spin.setSuffix(" μdeg")
        self.rz_base_r_spin.setMaximumWidth(100)
        self.rz_base_r_spin.setGroupSeparatorShown(True)
        z_series_layout.addWidget(self.rz_base_r_spin, 4, 5)

        z_series_layout.addWidget(QLabel("R Start:"), 4, 0)
        self.rz_r_start_spin = QDoubleSpinBox()
        self.rz_r_start_spin.setRange(-360000000, 360000000)
        self.rz_r_start_spin.setValue(0)
        self.rz_r_start_spin.setDecimals(0)
        self.rz_r_start_spin.setMaximumWidth(100)
        self.rz_r_start_spin.setGroupSeparatorShown(True)
        z_series_layout.addWidget(self.rz_r_start_spin, 4, 1)

        z_series_layout.addWidget(QLabel("R End:"), 4, 2)
        self.rz_r_end_spin = QDoubleSpinBox()
        self.rz_r_end_spin.setRange(-360000000, 360000000)
        self.rz_r_end_spin.setValue(180000)
        self.rz_r_end_spin.setDecimals(0)
        self.rz_r_end_spin.setMaximumWidth(100)
        self.rz_r_end_spin.setGroupSeparatorShown(True)
        z_series_layout.addWidget(self.rz_r_end_spin, 4, 3)

        z_series_layout.addWidget(QLabel("R Numbers:"), 4, 6)
        self.rz_r_numbers_spin = QSpinBox()
        self.rz_r_numbers_spin.setRange(2, 1000)
        self.rz_r_numbers_spin.setValue(5)
        self.rz_r_numbers_spin.setMaximumWidth(60)
        z_series_layout.addWidget(self.rz_r_numbers_spin, 4, 7)

        # Row 6: R step and COR
        z_series_layout.addWidget(QLabel("R Step:"), 5, 0)
        self.rz_r_step_spin = QDoubleSpinBox()
        self.rz_r_step_spin.setRange(1, 360000000)
        self.rz_r_step_spin.setValue(45000)
        self.rz_r_step_spin.setDecimals(0)
        self.rz_r_step_spin.setSuffix(" µdeg")
        self.rz_r_step_spin.setMaximumWidth(100)
        self.rz_r_step_spin.setGroupSeparatorShown(True)
        z_series_layout.addWidget(self.rz_r_step_spin, 5, 1)

        z_series_layout.addWidget(QLabel("COR X:"), 5, 2)
        self.rz_cor_x_spin = QDoubleSpinBox()
        self.rz_cor_x_spin.setRange(-20000000, 20000000)
        self.rz_cor_x_spin.setValue(0)
        self.rz_cor_x_spin.setDecimals(0)
        self.rz_cor_x_spin.setSuffix(" nm")
        self.rz_cor_x_spin.setMaximumWidth(100)
        self.rz_cor_x_spin.setGroupSeparatorShown(True)
        z_series_layout.addWidget(self.rz_cor_x_spin, 5, 3)

        z_series_layout.addWidget(QLabel("COR Y:"), 5, 4)
        self.rz_cor_y_spin = QDoubleSpinBox()
        self.rz_cor_y_spin.setRange(-20000000, 20000000)
        self.rz_cor_y_spin.setValue(0)
        self.rz_cor_y_spin.setDecimals(0)
        self.rz_cor_y_spin.setSuffix(" nm")
        self.rz_cor_y_spin.setMaximumWidth(100)
        self.rz_cor_y_spin.setGroupSeparatorShown(True)
        z_series_layout.addWidget(self.rz_cor_y_spin, 5, 5)

        z_series_layout.addWidget(QLabel("COR Base Z:"), 5, 6)
        self.rz_cor_base_z_spin = QDoubleSpinBox()
        self.rz_cor_base_z_spin.setRange(-20000000, 20000000)
        self.rz_cor_base_z_spin.setValue(0)
        self.rz_cor_base_z_spin.setDecimals(0)
        self.rz_cor_base_z_spin.setSuffix(" nm")
        self.rz_cor_base_z_spin.setMaximumWidth(100)
        self.rz_cor_base_z_spin.setGroupSeparatorShown(True)
        self.rz_cor_base_z_spin.setToolTip("Z height where COR X,Y are measured")
        z_series_layout.addWidget(self.rz_cor_base_z_spin, 5, 7)

        # Row 6: COR compensation ratios
        z_series_layout.addWidget(QLabel("COR X Comp:"), 6, 0)
        self.rz_cor_x_comp_spin = QDoubleSpinBox()
        self.rz_cor_x_comp_spin.setRange(-10.0, 10.0)
        self.rz_cor_x_comp_spin.setValue(1.0)
        self.rz_cor_x_comp_spin.setDecimals(8)  # CHANGED from 3 to 8
        self.rz_cor_x_comp_spin.setSingleStep(0.00000001)  # CHANGED for finer control
        self.rz_cor_x_comp_spin.setMaximumWidth(120)  # CHANGED from 70 to fit more digits
        self.rz_cor_x_comp_spin.setToolTip("COR X movement per Z (1.0 = 45° beam, 0.0 = disabled)")
        z_series_layout.addWidget(self.rz_cor_x_comp_spin, 6, 1)

        z_series_layout.addWidget(QLabel("COR Y Comp:"), 6, 2)
        self.rz_cor_y_comp_spin = QDoubleSpinBox()
        self.rz_cor_y_comp_spin.setRange(-10.0, 10.0)
        self.rz_cor_y_comp_spin.setValue(0.0)
        self.rz_cor_y_comp_spin.setDecimals(8)  # CHANGED from 3 to 8
        self.rz_cor_y_comp_spin.setSingleStep(0.00000001)  # CHANGED for finer control
        self.rz_cor_y_comp_spin.setMaximumWidth(120)  # CHANGED from 70 to fit more digits
        self.rz_cor_y_comp_spin.setToolTip("COR Y movement per Z (adjust if Y drift observed)")
        z_series_layout.addWidget(self.rz_cor_y_comp_spin, 6, 3)

        # Row 7: Total calculations
        z_series_layout.addWidget(QLabel("Total:"), 7, 0)
        self.z_total_label = QLabel("5 measurements")
        self.z_total_label.setStyleSheet("QLabel { font-weight: bold; }")
        z_series_layout.addWidget(self.z_total_label, 7, 1, 1, 3)

        z_series_layout.addWidget(QLabel("Est. Time:"), 7, 4)
        self.z_total_time_label = QLabel("5s")
        self.z_total_time_label.setStyleSheet("QLabel { font-weight: bold; color: #0066cc; }")
        z_series_layout.addWidget(self.z_total_time_label, 7, 5, 1, 3)

        z_series_group.setLayout(z_series_layout)
        z_series_group.setMaximumHeight(250)

        # Initially set to line scan mode
        self.set_z_mode_controls("Line Scan")

        layout.addWidget(z_series_group)

        # Control buttons
        button_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Line Scan")
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.start_btn.clicked.connect(self.start_scan_requested.emit)
        button_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        self.stop_btn.clicked.connect(self.stop_scan_requested.emit)
        button_layout.addWidget(self.stop_btn)

        layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        self.setLayout(layout)

    def connect_signals(self):
        """Connect internal signals"""
        self.fixed_axis_combo.currentTextChanged.connect(self.on_fixed_axis_changed)
        self.fixed_position_spin.valueChanged.connect(self.update_parameters)
        self.scan_start_spin.valueChanged.connect(self.update_calculations)
        self.scan_end_spin.valueChanged.connect(self.update_calculations)
        self.num_points_spin.valueChanged.connect(self.on_points_changed)
        self.step_size_spin.valueChanged.connect(self.on_step_changed)
        self.dwell_time_spin.valueChanged.connect(self.update_time_estimate)

        # Z-series signal connections
        self.z_start_spin.valueChanged.connect(self.on_z_range_changed)
        self.z_end_spin.valueChanged.connect(self.on_z_range_changed)
        self.z_numbers_spin.valueChanged.connect(self.on_z_numbers_changed)
        self.z_step_spin.valueChanged.connect(self.on_z_step_changed)
        self.z_dwell_spin.valueChanged.connect(self.update_z_calculations)
        self.z_fixed_x_spin.valueChanged.connect(self.update_parameters)
        self.z_fixed_y_spin.valueChanged.connect(self.update_parameters)
        self.z_base_z_spin.valueChanged.connect(self.update_parameters)
        self.z_x_compensation_spin.valueChanged.connect(self.update_parameters)
        self.z_y_compensation_spin.valueChanged.connect(self.update_parameters)

        """Connect R+Z series specific signals"""
        self.rz_r_start_spin.valueChanged.connect(self.update_rz_calculations)
        self.rz_r_end_spin.valueChanged.connect(self.update_rz_calculations)
        self.rz_r_numbers_spin.valueChanged.connect(self.on_rz_r_numbers_changed)
        self.rz_r_step_spin.valueChanged.connect(self.on_rz_r_step_changed)
        self.rz_base_r_spin.valueChanged.connect(self.update_parameters)
        self.rz_cor_x_spin.valueChanged.connect(self.update_parameters)
        self.rz_cor_y_spin.valueChanged.connect(self.update_parameters)
        self.rz_cor_x_comp_spin.valueChanged.connect(self.update_parameters)
        self.rz_cor_y_comp_spin.valueChanged.connect(self.update_parameters)

    def on_fixed_axis_changed(self):
        """Handle fixed axis change"""
        fixed_axis = self.fixed_axis_combo.currentText()
        scan_axis = "Y" if fixed_axis == "X" else "X"
        self.scan_axis_label.setText(scan_axis)
        self.update_parameters()

    def on_points_changed(self):
        """Handle points change - update step size"""
        if self._updating:
            return
        self._updating = True

        range_nm = abs(self.scan_end_spin.value() - self.scan_start_spin.value())
        points = self.num_points_spin.value()

        if points > 1 and range_nm > 0:
            step = range_nm / (points - 1)
            step = max(1, round(step))
            self.step_size_spin.setValue(step)

        self._updating = False
        self.update_calculations()

    def on_step_changed(self):
        """Handle step change - update points"""
        if self._updating:
            return
        self._updating = True

        range_nm = abs(self.scan_end_spin.value() - self.scan_start_spin.value())
        step = self.step_size_spin.value()

        if step > 0:
            points = int(range_nm / step) + 1
            points = max(2, points)
            self.num_points_spin.setValue(points)

        self._updating = False
        self.update_calculations()

    def on_z_series_toggled(self):
        """Handle Z-series enable/disable"""
        enabled = self.z_series_enable_cb.isChecked()
        self.set_z_series_controls_enabled(enabled)

        # Hide/show regular line scan controls
        self.fixed_axis_combo.setVisible(not enabled)
        self.fixed_position_spin.setVisible(not enabled)
        self.scan_start_spin.setVisible(not enabled)
        self.scan_end_spin.setVisible(not enabled)
        self.num_points_spin.setVisible(not enabled)
        self.step_size_spin.setVisible(not enabled)
        self.dwell_time_spin.setVisible(not enabled)

        self.update_parameters()

    def set_z_series_controls_enabled(self, enabled: bool):
        """Enable/disable Z-series controls"""
        self.fixed_x_spin.setEnabled(enabled)
        self.fixed_y_spin.setEnabled(enabled)
        self.base_z_spin.setEnabled(enabled)
        self.z_start_spin.setEnabled(enabled)
        self.z_end_spin.setEnabled(enabled)
        self.z_numbers_spin.setEnabled(enabled)
        self.z_step_spin.setEnabled(enabled)
        self.z_dwell_spin.setEnabled(enabled)
        self.z_x_compensation_spin.setEnabled(enabled)
        self.z_y_compensation_spin.setEnabled(enabled)

    def on_z_numbers_changed(self):
        """Handle Z numbers change - update step size"""
        if self._updating:
            return
        self._updating = True

        z_range = abs(self.z_end_spin.value() - self.z_start_spin.value())
        z_numbers = self.z_numbers_spin.value()

        if z_numbers > 1 and z_range > 0:
            z_step = z_range / (z_numbers - 1)
            z_step = max(1, round(z_step))
            self.z_step_spin.setValue(z_step)

        self._updating = False
        self.update_z_calculations()

    def on_z_step_changed(self):
        """Handle Z step change - update numbers"""
        if self._updating:
            return
        self._updating = True

        z_range = abs(self.z_end_spin.value() - self.z_start_spin.value())
        z_step = self.z_step_spin.value()

        if z_step > 0:
            z_numbers = int(z_range / z_step) + 1
            z_numbers = max(2, z_numbers)
            self.z_numbers_spin.setValue(z_numbers)

        self._updating = False
        self.update_z_calculations()

    def on_z_range_changed(self):
        """Handle Z range change - update step based on current numbers"""
        if self._updating:
            return
        self.on_z_numbers_changed()

    def update_z_calculations(self):
        """Update Z-series calculated displays"""
        z_numbers = self.z_numbers_spin.value()
        z_step = self.z_step_spin.value()
        z_eff_range = (z_numbers - 1) * z_step if z_numbers > 1 else 0
        self.z_effective_range_label.setText(f"{z_eff_range:.0f} nm")

        # Update time estimate
        z_dwell = self.z_dwell_spin.value()
        move_overhead = 0.2  # seconds per Z move
        # Get detector lag
        detector_lag = 0.35  # Default
        if self.parent() and hasattr(self.parent(), 'detector_lag_spin'):
            detector_lag = self.parent().detector_lag_spin.value()

        total_time = z_numbers * (z_dwell + move_overhead + detector_lag)

        if total_time < 60:
            time_str = f"{total_time:.1f}s ({z_numbers} × {z_dwell + move_overhead:.1f}s)"
        else:
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            time_str = f"{minutes}m {seconds}s ({z_numbers} × {z_dwell + move_overhead:.1f}s)"

        self.z_total_time_label.setText(time_str)
        self.update_parameters()

    def on_z_mode_changed(self):
        """Handle Z mode change"""
        mode = self.z_mode_combo.currentText()
        self.set_z_mode_controls(mode)
        self.update_parameters()

    def set_z_mode_controls(self, mode: str):
        """Enable/disable controls based on selected mode"""
        # Controls for Line Scan mode
        line_scan_active = (mode == "Line Scan")
        self.fixed_axis_combo.setVisible(line_scan_active)
        self.fixed_position_spin.setVisible(line_scan_active)
        self.scan_start_spin.setVisible(line_scan_active)
        self.scan_end_spin.setVisible(line_scan_active)
        self.num_points_spin.setVisible(line_scan_active)
        self.step_size_spin.setVisible(line_scan_active)
        self.dwell_time_spin.setVisible(line_scan_active)

        # Controls for Z-Scan or R+Z
        z_mode_active = (mode in ["Z-Scan", "R+Z Series"])
        self.z_fixed_x_spin.setEnabled(z_mode_active)
        self.z_fixed_y_spin.setEnabled(z_mode_active)
        self.z_base_z_spin.setEnabled(z_mode_active)
        self.z_start_spin.setEnabled(z_mode_active)
        self.z_end_spin.setEnabled(z_mode_active)
        self.z_numbers_spin.setEnabled(z_mode_active)
        self.z_step_spin.setEnabled(z_mode_active)
        self.z_dwell_spin.setEnabled(z_mode_active)
        self.z_x_compensation_spin.setEnabled(z_mode_active)
        self.z_y_compensation_spin.setEnabled(z_mode_active)

        # R parameters only for R+Z Series
        rz_active = (mode == "R+Z Series")
        self.rz_separator_label.setVisible(rz_active)
        self.rz_base_r_spin.setVisible(rz_active)
        self.rz_r_start_spin.setVisible(rz_active)
        self.rz_r_end_spin.setVisible(rz_active)
        self.rz_r_numbers_spin.setVisible(rz_active)
        self.rz_r_step_spin.setVisible(rz_active)
        self.rz_cor_x_spin.setVisible(rz_active)
        self.rz_cor_y_spin.setVisible(rz_active)
        self.rz_cor_base_z_spin.setVisible(rz_active)
        self.rz_cor_x_comp_spin.setVisible(rz_active)
        self.rz_cor_y_comp_spin.setVisible(rz_active)

        # Update labels
        if mode == "Z-Scan":
            self.z_total_label.setText(f"{self.z_numbers_spin.value()} Z points")
        elif mode == "R+Z Series":
            r_nums = self.rz_r_numbers_spin.value()
            z_nums = self.z_numbers_spin.value()
            self.z_total_label.setText(f"{r_nums * z_nums} points ({r_nums}R × {z_nums}Z)")

        self.update_z_calculations()

    def update_parameters(self):
        """Update parameters based on current mode"""
        mode = self.z_mode_combo.currentText()

        if mode == "Line Scan":
            # Regular line scan parameters (existing code)
            params = LineScanParameters(
                fixed_axis=self.fixed_axis_combo.currentText(),
                fixed_position=self.fixed_position_spin.value(),
                scan_axis="Y" if self.fixed_axis_combo.currentText() == "X" else "X",
                scan_start=self.scan_start_spin.value(),
                scan_end=self.scan_end_spin.value(),
                num_points=self.num_points_spin.value(),
                step_size=self.step_size_spin.value(),
                dwell_time=self.dwell_time_spin.value(),
                base_z_position=self.z_base_z_spin.value()
            )
        elif mode == "Z-Scan":
            # Z-scan parameters using ZSeriesParameters
            z_series = ZSeriesParameters(
                enabled=True,
                z_start=self.z_start_spin.value(),
                z_end=self.z_end_spin.value(),
                z_numbers=self.z_numbers_spin.value(),
                z_step_input=self.z_step_spin.value(),
                x_compensation_ratio=self.z_x_compensation_spin.value(),
                y_compensation_ratio=self.z_y_compensation_spin.value(),
                fixed_x_position=self.z_fixed_x_spin.value(),
                fixed_y_position=self.z_fixed_y_spin.value(),
                z_dwell_time=self.z_dwell_spin.value()
            )
            params = LineScanParameters(
                scan_axis="Z",  # Virtual Z axis
                z_series=z_series,
                base_z_position=self.z_base_z_spin.value()
            )
        elif mode == "R+Z Series":
            # R+Z combined parameters
            z_series = ZSeriesParameters(
                enabled=True,
                z_start=self.z_start_spin.value(),
                z_end=self.z_end_spin.value(),
                z_numbers=self.z_numbers_spin.value(),
                z_step_input=self.z_step_spin.value(),
                x_compensation_ratio=self.z_x_compensation_spin.value(),
                y_compensation_ratio=self.z_y_compensation_spin.value(),
                fixed_x_position=self.z_fixed_x_spin.value(),
                fixed_y_position=self.z_fixed_y_spin.value(),
                z_dwell_time=self.z_dwell_spin.value()
            )
            r_series = RSeriesParameters(
                enabled=True,
                r_start=self.rz_r_start_spin.value(),
                r_end=self.rz_r_end_spin.value(),
                r_numbers=self.rz_r_numbers_spin.value(),
                r_step_input=self.rz_r_step_spin.value(),
                base_r_position=self.rz_base_r_spin.value(),
                cor_enabled=True,  # Always use COR for R+Z
                cor_x=self.rz_cor_x_spin.value(),
                cor_y=self.rz_cor_y_spin.value(),
                cor_base_z=self.rz_cor_base_z_spin.value(),
                cor_x_compensation_ratio=self.rz_cor_x_comp_spin.value(),
                cor_y_compensation_ratio=self.rz_cor_y_comp_spin.value(),
                combine_with_z=True
            )
            params = LineScanParameters(
                scan_axis="RZ",  # Virtual RZ axis
                z_series=z_series,
                r_series=r_series,
                base_z_position=self.z_base_z_spin.value(),
                base_r_position=self.rz_base_r_spin.value()
            )

        self.parameters_changed.emit(params)

    def update_calculations(self):
        """Update calculated displays"""
        # Update effective range
        points = self.num_points_spin.value()
        step = self.step_size_spin.value()
        eff_range = (points - 1) * step if points > 1 else 0
        self.effective_range_label.setText(f"{eff_range:.0f} nm")

        # Color code based on underscan
        range_nm = abs(self.scan_end_spin.value() - self.scan_start_spin.value())
        underscan = max(0, range_nm - eff_range)

        if underscan < 50:
            self.effective_range_label.setStyleSheet("QLabel { background-color: #e8f4e8; padding: 2px; }")
        elif underscan < 200:
            self.effective_range_label.setStyleSheet("QLabel { background-color: #fff4e8; padding: 2px; }")
        else:
            self.effective_range_label.setStyleSheet("QLabel { background-color: #ffe8e8; padding: 2px; }")

        self.update_time_estimate()
        self.update_parameters()

    def on_rz_r_numbers_changed(self):
        """Handle R+Z R numbers change - update step size"""
        if self._updating:
            return
        self._updating = True

        r_range = abs(self.rz_r_end_spin.value() - self.rz_r_start_spin.value())
        r_numbers = self.rz_r_numbers_spin.value()

        if r_numbers > 1 and r_range > 0:
            r_step = r_range / (r_numbers - 1)
            r_step = max(1, round(r_step))
            self.rz_r_step_spin.setValue(r_step)

        self._updating = False
        self.update_rz_calculations()

    def on_rz_r_range_changed(self):
        """Handle R+Z R range change - update step based on current numbers"""
        if self._updating:
            return
        self.on_rz_r_numbers_changed()

    def on_rz_r_step_changed(self):
        """Handle R+Z R step change - update numbers"""
        if self._updating:
            return
        self._updating = True

        r_range = abs(self.rz_r_end_spin.value() - self.rz_r_start_spin.value())
        r_step = self.rz_r_step_spin.value()

        if r_step > 0:
            r_numbers = int(r_range / r_step) + 1
            r_numbers = max(2, min(1000, r_numbers))
            self.rz_r_numbers_spin.setValue(r_numbers)

        self._updating = False
        self.update_rz_calculations()

    def update_rz_calculations(self):
        """Update R+Z calculated displays"""
        if self._updating:
            return

        # Update total calculations for R+Z mode
        if self.z_mode_combo.currentText() == "R+Z Series":
            r_numbers = self.rz_r_numbers_spin.value()
            z_numbers = self.z_numbers_spin.value()
            total_measurements = r_numbers * z_numbers

            self.z_total_label.setText(f"{total_measurements} points ({r_numbers}R × {z_numbers}Z)")

            # Update time estimate
            z_dwell = self.z_dwell_spin.value()
            move_overhead = 0.2  # seconds per move (including R rotation)
            # Get detector lag
            detector_lag = 0.35  # Default
            if self.parent() and hasattr(self.parent(), 'detector_lag_spin'):
                detector_lag = self.parent().detector_lag_spin.value()

            # Each measurement needs dwell + move + detector_lag
            time_per_measurement = z_dwell + move_overhead + detector_lag
            total_time = total_measurements * time_per_measurement

            # Add time for R rotations between Z-series
            r_rotation_time = 0.2  # seconds per R rotation
            total_time += (r_numbers - 1) * r_rotation_time

            if total_time < 60:
                time_str = f"{total_time:.1f}s"
            else:
                minutes = int(total_time // 60)
                seconds = int(total_time % 60)
                if minutes >= 60:
                    hours = minutes // 60
                    minutes = minutes % 60
                    time_str = f"{hours}h {minutes}m"
                else:
                    time_str = f"{minutes}m {seconds}s"

            self.z_total_time_label.setText(time_str)

        self.update_parameters()

    def get_detector_lag(self) -> float:
        """Get detector lag value from parent MainWindow"""
        try:
            # Try to get from parent's detector_lag_spin
            parent = self.parent()
            while parent and not hasattr(parent, 'detector_lag_spin'):
                parent = parent.parent()

            if parent and hasattr(parent, 'detector_lag_spin'):
                return parent.detector_lag_spin.value()
        except:
            pass

        return 0.35  # Default detector lag in seconds

    def update_time_estimate(self):
        """Update time estimate"""
        points = self.num_points_spin.value()
        dwell = self.dwell_time_spin.value()
        move_overhead = 0.2  # seconds per point

        # Get detector lag from parent MainWindow
        detector_lag = 0.35  # Default
        if self.parent() and hasattr(self.parent(), 'detector_lag_spin'):
            detector_lag = self.parent().detector_lag_spin.value()

        total_time = points * (dwell + move_overhead + detector_lag)

        if total_time < 60:
            self.time_estimate_label.setText(f"{total_time:.1f} s")
        else:
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            self.time_estimate_label.setText(f"{minutes}m {seconds}s")

    def set_scan_state(self, state: ScanState):
        """Update UI based on scan state"""
        params_enabled = (state == ScanState.IDLE)

        # Enable/disable controls
        self.fixed_axis_combo.setEnabled(params_enabled)
        self.fixed_position_spin.setEnabled(params_enabled)
        self.scan_start_spin.setEnabled(params_enabled)
        self.scan_end_spin.setEnabled(params_enabled)
        self.num_points_spin.setEnabled(params_enabled)
        self.step_size_spin.setEnabled(params_enabled)
        self.dwell_time_spin.setEnabled(params_enabled)

        if state == ScanState.IDLE:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
        elif state == ScanState.SCANNING:
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setVisible(True)

    @pyqtSlot(int, int)
    def update_progress(self, current: int, total: int):
        """Update progress bar"""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
            self.progress_bar.setFormat(f"{current}/{total} ({progress}%)")

    def get_z_series_params(self) -> ZSeriesParameters:
        """Get Z-series parameters from UI"""
        z_start_value = float(self.z_start_spin.value())
        z_end_value = float(self.z_end_spin.value())
        z_numbers_value = int(self.z_numbers_spin.value())
        z_step_value = float(self.z_step_spin.value())

        params = ZSeriesParameters(
            enabled=True,  # Enabled when in Z-scan mode
            z_start=z_start_value,
            z_end=z_end_value,
            z_numbers=z_numbers_value,
            z_step_input=z_step_value,
            x_compensation_ratio=self.z_x_compensation_spin.value(),
            y_compensation_ratio=self.z_y_compensation_spin.value(),
            fixed_x_position=self.z_fixed_x_spin.value(),
            fixed_y_position=self.z_fixed_y_spin.value(),
            z_dwell_time=self.z_dwell_spin.value()
        )

        return params

class ImageDisplayWidget(QWidget):
    """Widget for displaying the reconstructed image"""

    def __init__(self):
        super().__init__()

        # Initialize attributes first to avoid AttributeError
        self.current_image = None
        self.scan_params = None
        self.current_colormap = 'viridis'
        self._last_preview_path = None  # Store preview path

        self.setup_ui()

    def setup_ui(self):
        """Setup the image display UI"""
        layout = QVBoxLayout()

        # Create main widget with manual layout for image and colorbar
        image_layout = QHBoxLayout()

        # Image plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setLabel('left', 'Y Position', units='nm')
        self.plot_widget.setLabel('bottom', 'X Position', units='nm')

        # Create image item
        self.image_item = pg.ImageItem(axisOrder='row-major')
        self.plot_widget.addItem(self.image_item)

        # Create scan path preview
        self.path_line = pg.PlotDataItem(
            pen=pg.mkPen(color='red', width=2, style=pg.QtCore.Qt.DashLine),
            name='Scan Path'
        )
        self.plot_widget.addItem(self.path_line)

        # Create start/end markers
        self.start_marker = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen('green', width=2),
            brush=pg.mkBrush('green'), name='Start'
        )
        self.end_marker = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen('red', width=2),
            brush=pg.mkBrush('red'), name='End'
        )
        self.plot_widget.addItem(self.start_marker)
        self.plot_widget.addItem(self.end_marker)

        # Initially hide preview elements
        self.path_line.hide()
        self.start_marker.hide()
        self.end_marker.hide()

        # Add image plot to layout
        image_layout.addWidget(self.plot_widget)

        # Create manual colorbar using a separate plot widget
        self.colorbar_widget = pg.PlotWidget()
        self.colorbar_widget.setMaximumWidth(50)
        self.colorbar_widget.setMinimumWidth(50)
        self.colorbar_widget.hideAxis('bottom')
        self.colorbar_widget.hideAxis('left')
        self.colorbar_widget.setMouseEnabled(x=False, y=False)

        # Create colorbar image
        self.colorbar_image = pg.ImageItem()
        self.colorbar_widget.addItem(self.colorbar_image)

        # Initialize colorbar with gradient
        self.update_colorbar_gradient()

        image_layout.addWidget(self.colorbar_widget)

        # Add image layout to main layout
        image_widget = QWidget()
        image_widget.setLayout(image_layout)
        layout.addWidget(image_widget)

        # Controls
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'CET-L1','CET-L2'])
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        controls_layout.addWidget(self.colormap_combo)

        self.auto_levels_btn = QPushButton("Auto Levels")
        self.auto_levels_btn.clicked.connect(self.auto_levels)
        controls_layout.addWidget(self.auto_levels_btn)

        self.save_btn = QPushButton("Save Image")
        self.save_btn.clicked.connect(self.save_image)
        controls_layout.addWidget(self.save_btn)

        self.export_btn = QPushButton("Export Data")
        self.export_btn.clicked.connect(self.export_data)
        controls_layout.addWidget(self.export_btn)

        # Preview controls
        self.show_preview_cb = QCheckBox("Show Path Preview")
        self.show_preview_cb.setChecked(True)  # Default to ON
        self.show_preview_cb.stateChanged.connect(self.toggle_preview)
        controls_layout.addWidget(self.show_preview_cb)

        controls_layout.addStretch()

        layout.addLayout(controls_layout)
        self.setLayout(layout)

        # Initialize colorbar after UI is set up
        try:
            self.update_colorbar_gradient()
        except Exception as e:
            print(f"Warning: Could not initialize colorbar: {e}")
            # Continue without colorbar if it fails

    def update_colorbar_gradient(self):
        """Update the colorbar gradient"""
        try:
            # Create a vertical gradient for colorbar
            gradient = np.linspace(0, 1, 256).reshape(256, 1)
            self.colorbar_image.setImage(gradient)

            # Set colormap
            colormap = pg.colormap.get(self.current_colormap)
            self.colorbar_image.setColorMap(colormap)

        except Exception as e:
            print(f"Error updating colorbar: {e}")

    def show_scan_preview(self, path_points):
        """
    Display a preview polyline of the planned scan path and mark start/end.
    path_points: list[(x_nm, y_nm)]
        """
        try:
        # Cache the path so we can re-show it when the checkbox is toggled
            self._last_preview_path = path_points

            if not path_points:
                self.clear_preview()
                return

            # Split into X/Y arrays
            xs, ys = zip(*path_points)
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)

            # Update the polyline
            self.path_line.setData(xs, ys)

            # Update start/end markers
            start_x, start_y = xs[0], ys[0]
            end_x, end_y   = xs[-1], ys[-1]
            self.start_marker.setData([start_x], [start_y])
            self.end_marker.setData([end_x], [end_y])

            # Show or hide depending on checkbox
            if hasattr(self, "show_preview_cb") and self.show_preview_cb.isChecked():
                self.path_line.show()
                self.start_marker.show()
                self.end_marker.show()
            else:
                self.path_line.hide()
                self.start_marker.hide()
                self.end_marker.hide()

        except Exception as e:
            print(f"Preview error: {e}")


    def toggle_preview(self, state):
        """
        Toggle preview visibility.
        """
        checked = (state == Qt.Checked)
        if checked and getattr(self, "_last_preview_path", None):
            # Re-show the last computed path
            self.path_line.show()
            self.start_marker.show()
            self.end_marker.show()
        else:
            self.clear_preview()


    def clear_preview(self):
        """Clear scan path preview"""
        self.path_line.hide()
        self.start_marker.hide()
        self.end_marker.hide()

    def export_data(self):
        """Export current image data"""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image data to export")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Data",
            f"scan_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv);;NumPy Files (*.npy);;All Files (*)"
        )

        if filename:
            try:
                if filename.endswith('.csv'):
                    # Export as CSV
                    np.savetxt(filename, self.current_image, delimiter=',', fmt='%.6f')
                elif filename.endswith('.npy'):
                    # Export as NumPy binary
                    np.save(filename, self.current_image)
                else:
                    # Default to CSV
                    np.savetxt(filename, self.current_image, delimiter=',', fmt='%.6f')

                QMessageBox.information(self, "Success", f"Data exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data: {e}")

    def save_image(self):
        """Save the current image"""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image to save")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Image",
            f"scan_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG Files (*.png);;TIFF Files (*.tif);;All Files (*)"
        )

        if filename:
            try:
                # Export the plot widget
                exporter = pg.exporters.ImageExporter(self.plot_widget.plotItem)
                exporter.export(filename)
                QMessageBox.information(self, "Success", f"Image saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {e}")

    # : Remove duplicate save_image method (keeping only one)

    @pyqtSlot(object)
    def update_image(self, image_data):
        """Update the displayed image"""
        self.current_image = image_data

        # Set the image with proper levels for float data
        if image_data is not None:
            # Calculate levels for float data (required by pyqtgraph)
            valid_data = image_data[~np.isnan(image_data)]
            if len(valid_data) > 0:
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                # Ensure min != max to avoid division by zero
                if min_val == max_val:
                    levels = [min_val - 0.1, max_val + 0.1]
                else:
                    levels = [min_val, max_val]
            else:
                # No valid data, use default levels
                levels = [0.0, 1.0]

            # Set image with explicit levels
            self.image_item.setImage(image_data, levels=levels)
        else:
            # Handle None image data
            self.image_item.setImage(np.zeros((10, 10)), levels=[0, 1])

        # Set coordinate transformation after image is set
        # Set coordinate transformation after image is set
        if self.scan_params and image_data is not None:
            try:
                # Create rectangle based on actual scan start position and effective FOV
                # The image always starts from pixel (0,0) which corresponds to the scan start position
                x_width = abs(self.scan_params.x_end - self.scan_params.x_start)
                y_height = abs(self.scan_params.y_end - self.scan_params.y_start)

                rect = pg.QtCore.QRectF(
                    self.scan_params.x_start,
                    self.scan_params.y_start,
                    x_width if self.scan_params.x_end >= self.scan_params.x_start else -x_width,
                    y_height if self.scan_params.y_end >= self.scan_params.y_start else -y_height
                )
                self.image_item.setRect(rect)
            except Exception as e:
                print(f"Warning: Could not set image rect: {e}")

        # Set colormap
        try:
            colormap = pg.colormap.get(self.current_colormap)
            self.image_item.setColorMap(colormap)
        except Exception as e:
            print(f"Warning: Could not set colormap: {e}")

        # Update colorbar levels to match image
        if image_data is not None:
            self.update_colorbar_levels(image_data)

    def update_colorbar_levels(self, image_data):
        """Update colorbar to match image levels"""
        try:
            # Get current image levels
            levels = self.image_item.getLevels()
            if levels is not None:
                # Scale colorbar gradient to match image levels
                min_val, max_val = levels
                gradient = np.linspace(min_val, max_val, 256).reshape(256, 1)
                self.colorbar_image.setImage(gradient)

        except Exception as e:
            print(f"Error updating colorbar levels: {e}")

    def set_scan_parameters(self, scan_params: ScanParameters):
        """Set scan parameters for proper scaling"""
        self.scan_params = scan_params

        # Update plot labels
        self.plot_widget.setLabel('left', f'Y Position (nm)')
        self.plot_widget.setLabel('bottom', f'X Position (nm)')
        self.plot_widget.setTitle(f'Scan Image ({scan_params.x_pixels}x{scan_params.y_pixels})')

    def update_colormap(self, colormap_name: str):
        """Update the colormap"""
        try:
            self.current_colormap = colormap_name
            colormap = pg.colormap.get(colormap_name)

            # Update image colormap
            if self.current_image is not None:
                self.image_item.setColorMap(colormap)

            # Update colorbar colormap
            self.colorbar_image.setColorMap(colormap)
            self.update_colorbar_gradient()

        except Exception as e:
            print(f"Error updating colormap: {e}")

    def auto_levels(self):
        """Auto-adjust image levels"""
        if self.current_image is not None:
            try:
                # Calculate levels excluding NaN values
                valid_data = self.current_image[~np.isnan(self.current_image)]
                if len(valid_data) > 0:
                    min_val = np.percentile(valid_data, 1)
                    max_val = np.percentile(valid_data, 99)

                    # Ensure min != max
                    if min_val == max_val:
                        min_val -= 0.1
                        max_val += 0.1

                    self.image_item.setLevels([min_val, max_val])
                    self.update_colorbar_levels(self.current_image)
                else:
                    # No valid data, use default levels
                    self.image_item.setLevels([0.0, 1.0])
            except Exception as e:
                print(f"Warning: Could not auto-adjust levels: {e}")
                # Fallback to default levels
                self.image_item.setLevels([0.0, 1.0])

class LineScanDisplayWidget(QWidget):
    """Widget for displaying line scan data from reconstructor - simplified version"""

    def __init__(self):
        super().__init__()
        self.current_scan_params = None
        self.setup_ui()

    def setup_ui(self):
        """Setup the line scan display UI"""
        layout = QVBoxLayout()

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Signal', units='nA')
        self.plot_widget.setLabel('bottom', 'Position', units='nm')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Single plot item for current scan
        self.live_plot = self.plot_widget.plot([], [],
                                               pen=pg.mkPen(color='r', width=2),
                                               symbol='o', symbolSize=4,
                                               name='Current Scan')

        layout.addWidget(self.plot_widget)

        # Controls
        controls_layout = QHBoxLayout()

        self.save_btn = QPushButton("Save Data")
        self.save_btn.clicked.connect(self.save_current_data)
        controls_layout.addWidget(self.save_btn)

        self.auto_scale_cb = QCheckBox("Auto Scale")
        self.auto_scale_cb.setChecked(True)
        controls_layout.addWidget(self.auto_scale_cb)

        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # Statistics display
        stats_layout = QHBoxLayout()

        stats_layout.addWidget(QLabel("Points:"))
        self.points_label = QLabel("0")
        stats_layout.addWidget(self.points_label)

        stats_layout.addWidget(QLabel("Min:"))
        self.min_label = QLabel("N/A")
        stats_layout.addWidget(self.min_label)

        stats_layout.addWidget(QLabel("Max:"))
        self.max_label = QLabel("N/A")
        stats_layout.addWidget(self.max_label)

        stats_layout.addWidget(QLabel("Mean:"))
        self.mean_label = QLabel("N/A")
        stats_layout.addWidget(self.mean_label)

        stats_layout.addWidget(QLabel("Completion:"))
        self.completion_label = QLabel("0%")
        stats_layout.addWidget(self.completion_label)

        stats_layout.addStretch()

        layout.addLayout(stats_layout)

        self.setLayout(layout)

    def set_scan_parameters(self, params: LineScanParameters):
        """Set scan parameters for proper display"""
        self.current_scan_params = params

        # Update axis labels based on scan type
        if hasattr(params, 'z_series') and params.z_series.enabled:
            # Z-scan mode: Z position vs signal
            self.plot_widget.setLabel('bottom', 'Z Position', units='nm')
            self.plot_widget.setTitle(f'Z-Scan: {params.z_series.z_numbers} points')
        else:
            # Regular line scan: X or Y vs signal
            if params.scan_axis == "X":
                self.plot_widget.setLabel('bottom', 'X Position', units='nm')
            else:
                self.plot_widget.setLabel('bottom', 'Y Position', units='nm')
            self.plot_widget.setTitle(f'Line Scan: {params.num_points} points')

    @pyqtSlot(object, object)
    def update_line(self, positions, signals):
        """Update the displayed line data from reconstructor"""
        if positions is None or signals is None or len(positions) == 0:
            return

        # Additional check for all-NaN data
        if len(positions) > 0 and np.all(np.isnan(signals)):
            return

        # Update live plot
        self.live_plot.setData(positions, signals)

        # Update statistics
        self.update_statistics(signals)

        # Update completion percentage
        if self.current_scan_params:
            valid_count = np.sum(~np.isnan(signals))
            total = self.current_scan_params.num_points
            percentage = (valid_count / total * 100) if total > 0 else 0
            self.completion_label.setText(f"{percentage:.1f}%")

        # Auto scale if enabled
        if self.auto_scale_cb.isChecked():
            self.plot_widget.autoRange()

    @pyqtSlot(int, float, float)
    def update_single_point(self, index, position, value):
        """Update display when a single point is updated"""
        # For now, we rely on the periodic full updates
        pass

    def update_statistics(self, signals):
        """Update statistics display"""
        if len(signals) == 0:
            return

        valid_signals = signals[~np.isnan(signals)]
        if len(valid_signals) == 0:
            return

        self.points_label.setText(str(len(valid_signals)))
        self.min_label.setText(f"{np.min(valid_signals):.3f}")
        self.max_label.setText(f"{np.max(valid_signals):.3f}")
        self.mean_label.setText(f"{np.mean(valid_signals):.3f}")

    @pyqtSlot()
    def on_scan_completed(self):
        """Handle scan completion from reconstructor"""
        # Just update the completion to 100%
        self.completion_label.setText("100%")

    def save_current_data(self):
        """Save current line scan data"""
        x_data, y_data = self.live_plot.getData()

        if x_data is None or len(x_data) == 0:
            QMessageBox.warning(self, "Warning", "No data to save")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Line Scan",
            f"line_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv);;All Files (*)"
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write("# Line Scan Data\n")
                    f.write(f"# Exported: {datetime.now()}\n")
                    if self.current_scan_params:
                        f.write(f"# Scan Axis: {self.current_scan_params.scan_axis}\n")
                        f.write(
                            f"# Fixed Axis: {self.current_scan_params.fixed_axis} at {self.current_scan_params.fixed_position} nm\n")
                        f.write(
                            f"# Range: {self.current_scan_params.scan_start} to {self.current_scan_params.scan_end} nm\n")
                    f.write("Position (nm), Signal (nA)\n")
                    for xi, yi in zip(x_data, y_data):
                        if not np.isnan(yi):  # Only save valid data points
                            f.write(f"{xi:.6f}, {yi:.6f}\n")

                QMessageBox.information(self, "Success", f"Data saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

class RZSeriesDisplayWidget(QWidget):
    """Widget for displaying R+Z series data as a polar heatmap (Z=radius, R=angle)"""

    def __init__(self):
        super().__init__()

        #throttled updates
        self._dirty = False
        self._pending_marker = None
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._on_update_timer)
        self._update_timer.start(150)  # ~6-7 FPS updates

        self.r_positions = []  # Angular positions in degrees
        self.z_positions = []  # Radial positions in nm
        self.data_matrix = None
        self.current_r_index = 0
        self.current_z_index = 0
        self.r_step_deg = 1.0  # Will be calculated from actual data
        self.current_colormap = 'viridis'  # Default colormap
        self.setup_ui()

    def setup_ui(self):
        """Setup the polar R+Z display UI"""
        layout = QVBoxLayout()

        # Create main display layout with colorbar
        display_layout = QHBoxLayout()

        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setLabel('left', 'Y', units='nm')
        self.plot_widget.setLabel('bottom', 'X', units='nm')
        self.plot_widget.setTitle('R+Z Series (Polar: Z=radius, R=angle)')

        # Remove default axes and add polar grid
        self.plot_widget.showGrid(x=False, y=False)

        # Create image item for polar heatmap
        self.image_item = pg.ImageItem(axisOrder='row-major')
        self.plot_widget.addItem(self.image_item)

        # Add polar grid overlay
        self.setup_polar_grid()

        # Add current position marker
        self.position_marker = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen('yellow', width=2),
            brush=pg.mkBrush('yellow'), name='Current'
        )
        self.plot_widget.addItem(self.position_marker)

        display_layout.addWidget(self.plot_widget)

        # Create colorbar widget (same as 2D image display)
        self.colorbar_widget = pg.PlotWidget()
        self.colorbar_widget.setMaximumWidth(50)
        self.colorbar_widget.setMinimumWidth(50)
        self.colorbar_widget.hideAxis('bottom')
        self.colorbar_widget.hideAxis('left')
        self.colorbar_widget.setMouseEnabled(x=False, y=False)

        # Create colorbar image
        self.colorbar_image = pg.ImageItem()
        self.colorbar_widget.addItem(self.colorbar_image)

        # Initialize colorbar with gradient
        self.update_colorbar_gradient()

        display_layout.addWidget(self.colorbar_widget)

        # Add display layout to main layout
        display_widget = QWidget()
        display_widget.setLayout(display_layout)
        layout.addWidget(display_widget)

        # Controls
        controls_layout = QHBoxLayout()

        # Colormap selection (same as 2D image)
        controls_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'CET-L1', 'CET-L2'])
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        controls_layout.addWidget(self.colormap_combo)

        # Auto levels button (same as 2D image)
        self.auto_levels_btn = QPushButton("Auto Levels")
        self.auto_levels_btn.clicked.connect(self.auto_levels)
        controls_layout.addWidget(self.auto_levels_btn)

        controls_layout.addWidget(QLabel("Polar View: Z=radius, R=angle"))

        self.show_grid_cb = QCheckBox("Show Polar Grid")
        self.show_grid_cb.setChecked(True)
        self.show_grid_cb.stateChanged.connect(self.toggle_polar_grid)
        controls_layout.addWidget(self.show_grid_cb)

        self.auto_scale_cb = QCheckBox("Auto Scale")
        self.auto_scale_cb.setChecked(True)
        controls_layout.addWidget(self.auto_scale_cb)

        self.save_btn = QPushButton("Save Polar Data")
        self.save_btn.clicked.connect(self.save_data)
        controls_layout.addWidget(self.save_btn)

        controls_layout.addWidget(QLabel("Current:"))
        self.current_label = QLabel("R: 0°, Z: 0nm")
        controls_layout.addWidget(self.current_label)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        self.setLayout(layout)

    def setup_polar_grid(self):
        """Add polar coordinate grid overlay"""
        self.grid_items = []
        # Will be populated when data is available

    def toggle_polar_grid(self):
        """Show/hide polar grid"""
        show = self.show_grid_cb.isChecked()
        for item in self.grid_items:
            item.setVisible(show)

    def add_polar_grid_overlay(self):
        """Add polar grid lines after data is loaded (origin = min(Z))"""
        # Clear existing grid
        for item in self.grid_items:
            self.plot_widget.removeItem(item)
        self.grid_items.clear()

        if len(self.z_positions) == 0:
            return

        min_z = float(np.min(self.z_positions))
        max_z = float(np.max(self.z_positions))
        r_max = max_z - min_z

        # Radial grid: draw at positive radii, but label with original Z
        n_radial_lines = min(6, len(self.z_positions))
        radii = np.linspace(0.0, r_max, n_radial_lines)

        for k, radius in enumerate(radii):
            if radius <= 0:
                continue
            circle = pg.QtWidgets.QGraphicsEllipseItem(-radius, -radius, 2 * radius, 2 * radius)
            circle.setPen(pg.mkPen('gray', width=1, style=pg.QtCore.Qt.DashLine))
            self.plot_widget.addItem(circle)
            self.grid_items.append(circle)

            # Label shows actual Z value at that circle
            z_at_circle = min_z + radius
            label = pg.TextItem(f'{z_at_circle:.0f}nm', color='gray')
            label.setPos(radius * 0.707, radius * 0.707)
            self.plot_widget.addItem(label)
            self.grid_items.append(label)

        # Angular spokes
        n_angular_lines = min(12, len(self.r_positions))
        angles = np.linspace(0, 360, n_angular_lines, endpoint=False)

        for angle_deg in angles:
            angle_rad = np.deg2rad(angle_deg)
            x_end = r_max * np.cos(angle_rad)
            y_end = r_max * np.sin(angle_rad)

            line = pg.PlotDataItem([0.0, x_end], [0.0, y_end],
                                   pen=pg.mkPen('gray', width=1, style=pg.QtCore.Qt.DashLine))
            self.plot_widget.addItem(line)
            self.grid_items.append(line)

            if angle_deg % 30 == 0:
                label_radius = r_max * 1.05
                label_x = label_radius * np.cos(angle_rad)
                label_y = label_radius * np.sin(angle_rad)
                label = pg.TextItem(f'{angle_deg:.0f}°', color='gray')
                label.setPos(label_x, label_y)
                self.plot_widget.addItem(label)
                self.grid_items.append(label)

    def initialize_rz_scan(self, r_params, z_params):
        """Initialize display for new R+Z scan"""
        self.r_positions = np.array(r_params.r_positions) / 1e6  # Convert to degrees
        self.z_positions = np.array(z_params.z_positions)

        # Calculate actual R step size for proper slice width
        if len(self.r_positions) > 1:
            self.r_step_deg = abs(self.r_positions[1] - self.r_positions[0])
        else:
            self.r_step_deg = 1.0

        # Initialize data matrix with NaN (rows=Z, cols=R)
        self.data_matrix = np.full((len(self.z_positions), len(self.r_positions)), np.nan)

        # Add polar grid overlay
        self.add_polar_grid_overlay()

        # Create initial polar display
        self.update_polar_display()

    def create_polar_image(self):
        """Create polar coordinate image with proper slice sizing (origin = min(Z))."""
        if self.data_matrix is None or len(self.z_positions) == 0 or len(self.r_positions) == 0:
            return np.zeros((100, 100))

        # Use min Z as origin => all display radii are non-negative
        min_z = float(np.min(self.z_positions))
        max_z = float(np.max(self.z_positions))
        r_max = max_z - min_z if max_z > min_z else 1.0  # avoid zero extent

        size = 400
        extent = r_max * 1.1

        # Cartesian grid for output
        x_out = np.linspace(-extent, extent, size)
        y_out = np.linspace(-extent, extent, size)
        X_out, Y_out = np.meshgrid(x_out, y_out)

        # Polar coords of output grid
        R_out = np.sqrt(X_out ** 2 + Y_out ** 2)
        Theta_out_deg = np.rad2deg(np.arctan2(Y_out, X_out))
        Theta_out_deg[Theta_out_deg < 0] += 360.0

        # Output image
        polar_image = np.full((size, size), np.nan)

        # If no values yet, avoid All-NaN warnings
        if np.all(np.isnan(self.data_matrix)):
            return np.zeros((size, size))

        # Tolerances
        if len(self.z_positions) > 1:
            # robust step estimate (handles nonuniform Z lists)
            z_steps = np.diff(np.sort(self.z_positions))
            z_step = float(np.median(np.abs(z_steps))) if np.any(z_steps) else max(r_max, 1.0)
        else:
            z_step = max(r_max, 1.0)
        r_tol = max(0.5 * z_step, 1e-9)
        theta_tol = max(self.r_step_deg / 2.0, 1e-6)

        # Map each (Z,R) value onto the output grid
        for i, z_pos in enumerate(self.z_positions):
            r_center = float(z_pos - min_z)  # SHIFTED radius
            if r_center < 0 or r_center > r_max:
                continue
            r_mask = np.abs(R_out - r_center) < r_tol

            for j, r_angle in enumerate(self.r_positions):
                val = self.data_matrix[i, j]
                if np.isnan(val):
                    continue
                theta_diff = np.minimum(np.abs(Theta_out_deg - r_angle),
                                        360.0 - np.abs(Theta_out_deg - r_angle))
                theta_mask = theta_diff < theta_tol
                polar_image[r_mask & theta_mask] = val

        return polar_image

    def update_polar_display(self):
        """Update the polar coordinate display"""
        if self.data_matrix is None or len(self.z_positions) == 0:
            return

        polar_image = self.create_polar_image()

        # If still all-NaN (shouldn't happen now, but be safe), use zeros and default levels
        if np.all(np.isnan(polar_image)):
            polar_image = np.zeros_like(polar_image)
            self.image_item.setLevels([0.0, 1.0])

        self.image_item.setImage(polar_image)

        # Coordinate system: extent is based on (maxZ - minZ), not maxZ
        min_z = float(np.min(self.z_positions))
        max_z = float(np.max(self.z_positions))
        extent = (max_z - min_z) * 1.1 if max_z > min_z else 1.0
        self.image_item.setRect(pg.QtCore.QRectF(-extent, -extent, 2 * extent, 2 * extent))

        # Colormap
        try:
            colormap = pg.colormap.get(self.current_colormap)
            self.image_item.setColorMap(colormap)
        except Exception as e:
            print(f"Warning: Could not set colormap: {e}")

    def update_point(self, r_index: int, z_index: int, value: float):
        """Update a single point WITHOUT rebuilding the whole image"""
        if self.data_matrix is None:
            return

        self.current_r_index = r_index
        self.current_z_index = z_index

        # Update data matrix
        if 0 <= z_index < len(self.z_positions) and 0 <= r_index < len(self.r_positions):
            self.data_matrix[z_index, r_index] = value

            # Calculate marker position but DON'T update display yet
            z_pos_original = self.z_positions[z_index]
            min_z = np.min(self.z_positions)
            z_pos_shifted = z_pos_original - min_z
            r_angle_deg = self.r_positions[r_index]
            r_angle_rad = np.deg2rad(r_angle_deg)

            x_pos = z_pos_shifted * np.cos(r_angle_rad)
            y_pos = z_pos_shifted * np.sin(r_angle_rad)

            # Store pending update
            self._pending_marker = (x_pos, y_pos, r_angle_deg, z_pos_original)
            self._dirty = True

            # Auto-scale levels if needed (lightweight operation)
            if self.auto_scale_cb.isChecked():
                valid_data = self.data_matrix[~np.isnan(self.data_matrix)]
                if len(valid_data) > 0:
                    # Just update levels, don't rebuild image
                    self.image_item.setLevels([np.min(valid_data), np.max(valid_data)])
                    self.update_colorbar_levels()

    def update_colorbar_gradient(self):
        """Update the colorbar gradient"""
        try:
            # Create a vertical gradient for colorbar
            gradient = np.linspace(0, 1, 256).reshape(256, 1)
            self.colorbar_image.setImage(gradient)

            # Set colormap
            colormap = pg.colormap.get(self.current_colormap)
            self.colorbar_image.setColorMap(colormap)

        except Exception as e:
            print(f"Error updating colorbar: {e}")

    def _on_update_timer(self):
        """Periodic update - only rebuild image if data changed"""
        if not self._dirty:
            return

        self._dirty = False

        # NOW do the expensive polar display update
        self.update_polar_display()

        # Update marker and label if pending
        if self._pending_marker is not None:
            x_pos, y_pos, r_deg, z_nm = self._pending_marker
            self.position_marker.setData([x_pos], [y_pos])
            self.current_label.setText(f"R: {r_deg:.1f}°, Z: {z_nm:.0f}nm")
            self._pending_marker = None

    def update_colormap(self, colormap_name: str):
        """Update the colormap"""
        try:
            self.current_colormap = colormap_name
            colormap = pg.colormap.get(colormap_name)

            # Update image colormap
            self.image_item.setColorMap(colormap)

            # Update colorbar colormap
            self.colorbar_image.setColorMap(colormap)
            self.update_colorbar_gradient()

        except Exception as e:
            print(f"Error updating colormap: {e}")

    def auto_levels(self):
        """Auto-adjust image levels"""
        if self.data_matrix is not None:
            try:
                # Calculate levels excluding NaN values
                valid_data = self.data_matrix[~np.isnan(self.data_matrix)]
                if len(valid_data) > 0:
                    min_val = np.percentile(valid_data, 1)
                    max_val = np.percentile(valid_data, 99)

                    # Ensure min != max
                    if min_val == max_val:
                        min_val -= 0.1
                        max_val += 0.1

                    self.image_item.setLevels([min_val, max_val])
                    self.update_colorbar_levels()
                else:
                    # No valid data, use default levels
                    self.image_item.setLevels([0.0, 1.0])
            except Exception as e:
                print(f"Warning: Could not auto-adjust levels: {e}")
                # Fallback to default levels
                self.image_item.setLevels([0.0, 1.0])

    def update_colorbar_levels(self):
        """Update colorbar to match image levels"""
        try:
            # Get current image levels
            levels = self.image_item.getLevels()
            if levels is not None:
                # Scale colorbar gradient to match image levels
                min_val, max_val = levels
                gradient = np.linspace(min_val, max_val, 256).reshape(256, 1)
                self.colorbar_image.setImage(gradient)

        except Exception as e:
            print(f"Error updating colorbar levels: {e}")

    def save_data(self):
        """Save polar R+Z matrix data"""
        if self.data_matrix is None:
            QMessageBox.warning(self, "Warning", "No polar data to save")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Polar R+Z Data",
            f"rz_polar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz",
            "NumPy Files (*.npz);;CSV Files (*.csv);;All Files (*)"
        )

        if filename:
            try:
                if filename.endswith('.npz'):
                    np.savez(filename,
                             r_angles_deg=self.r_positions,
                             z_radii_nm=self.z_positions,
                             polar_data_matrix=self.data_matrix,
                             r_step_deg=self.r_step_deg,
                             min_z_origin=np.min(self.z_positions),
                             description="Polar R+Z scan: Z=radius(nm), R=angle(deg), origin=min_z")
                else:
                    with open(filename, 'w') as f:
                        f.write("# Polar R+Z Series Data (min Z as origin)\n")
                        f.write(f"# Z positions (radius, nm): {self.z_positions.tolist()}\n")
                        f.write(f"# R positions (angle, deg): {self.r_positions.tolist()}\n")
                        f.write(f"# R step size: {self.r_step_deg:.2f} degrees\n")
                        f.write(f"# Origin at Z = {np.min(self.z_positions):.1f} nm\n")
                        f.write("# Data format: rows=Z(radius), columns=R(angle)\n")
                        np.savetxt(f, self.data_matrix, delimiter=',', fmt='%.6f')

                QMessageBox.information(self, "Success", f"Polar data saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

class HDF5WriterThread(QThread):
    """Background thread for non-blocking HDF5 writes"""

    error_occurred = pyqtSignal(str)

    def __init__(self, data_storage, max_queue_size=50):
        super().__init__()
        self.data_storage = data_storage
        self.write_queue = queue.Queue(maxsize=max_queue_size)
        self._stop_requested = False

    def run(self):
        """Worker thread main loop"""
        while not self._stop_requested:
            try:
                # Wait for write request (with timeout to check stop flag)
                filepath, data_batch = self.write_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # Do the actual HDF5 write (off the main thread)
                self.data_storage.save_raw_data_batch(filepath, data_batch)
            except Exception as e:
                self.error_occurred.emit(f"HDF5 write error: {e}")
            finally:
                self.write_queue.task_done()

    @pyqtSlot(str, list)
    def enqueue_write(self, filepath: str, data_batch: list):
        """Queue a batch for writing (non-blocking)"""
        try:
            # Try to add to queue without blocking
            self.write_queue.put_nowait((filepath, list(data_batch)))
        except queue.Full:
            # Handle backpressure: drop oldest, add new
            try:
                self.write_queue.get_nowait()
                self.write_queue.task_done()
                self.write_queue.put_nowait((filepath, list(data_batch)))
            except:
                pass  # If still can't add, just drop it

    def stop(self):
        """Request thread to stop"""
        self._stop_requested = True
        self.wait(2000)  # Wait up to 2 seconds for thread to finish

def _index_from_pos(x, x0, x1, n):
    """Convert position to pixel index, handling any scan direction"""
    if n <= 1 or x0 == x1:
        return 0
    # map x in [min(x0,x1), max(x0,x1)] → [0, n-1], regardless of direction
    t = (x - x0) / (x1 - x0)  # will be negative if x1 < x0 — that's fine
    xi = int(round(t * (n - 1)))
    return int(np.clip(xi, 0, n - 1))


class DataFileReader:
    """Read and parse scan data files (HDF5, CSV, NPZ)"""

    @staticmethod
    def read_file(filepath: str) -> dict:
        """
        Read scan data file and return standardized dict

        Returns:
            {
                'scan_type': '2D', 'LINE', 'Z', or 'RZ',
                'scan_id': str,
                'data': np.ndarray (2D image or 1D arrays),
                'positions': dict with x/y/z/r arrays if available,
                'metadata': dict with scan parameters,
                'success': bool,
                'error': str if failed
            }
        """
        ext = os.path.splitext(filepath)[1].lower()

        try:
            if ext == '.h5':
                return DataFileReader._read_h5(filepath)
            elif ext == '.csv':
                return DataFileReader._read_csv(filepath)
            elif ext == '.npz':
                return DataFileReader._read_npz(filepath)
            else:
                return {'success': False, 'error': f'Unsupported file type: {ext}'}
        except Exception as e:
            return {'success': False, 'error': f'Error reading file: {str(e)}'}

    @staticmethod
    def _read_h5(filepath: str) -> dict:
        """Read HDF5 scan file"""
        if not HDF5_AVAILABLE:
            return {'success': False, 'error': 'h5py not available'}

        result = {
            'success': True,
            'filepath': filepath,
            'scan_id': os.path.basename(filepath).replace('.h5', ''),
            'metadata': {},
            'positions': {},
            'data': None  # INITIALIZE THIS!
        }

        with h5py.File(filepath, 'r') as f:
            # Read metadata from attributes - DECODE BYTE STRINGS
            for key in f.attrs.keys():
                value = f.attrs[key]
                # Decode byte strings to regular strings
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                result['metadata'][key] = value

            # Determine scan type from metadata (normalize to uppercase)
            scan_type_raw = result['metadata'].get('scan_type', 'unknown')
            if isinstance(scan_type_raw, bytes):
                scan_type_raw = scan_type_raw.decode('utf-8')
            scan_type = str(scan_type_raw).upper()  # Normalize to uppercase

            # Try to infer scan type if not explicitly set or unclear
            if scan_type == 'UNKNOWN' or scan_type not in ['2D', 'LINE', 'Z', 'RZ']:
                # Infer from data structure
                if 'reconstructed_image' in f and 'image' in f['reconstructed_image']:
                    scan_type = '2D'
                elif 'raw_data' in f:
                    # Check if it's a line/Z scan
                    raw = f['raw_data']
                    if 'z_positions' in raw:
                        # Could be Z-scan or regular line scan
                        # Check metadata for more clues
                        if 'Moving axis' in result['metadata']:
                            moving_axis = result['metadata']['Moving axis']
                            if isinstance(moving_axis, bytes):
                                moving_axis = moving_axis.decode('utf-8')
                            if moving_axis == 'Z':
                                scan_type = 'Z'
                            else:
                                scan_type = 'LINE'
                        else:
                            # Default to LINE if ambiguous
                            scan_type = 'LINE'

            result['scan_type'] = scan_type

            # Read reconstructed image if available (2D scans)
            if 'reconstructed_image' in f and 'image' in f['reconstructed_image']:
                try:
                    result['data'] = f['reconstructed_image/image'][:]

                    # Read coordinate arrays if available
                    if 'x_coordinates' in f['reconstructed_image']:
                        result['positions']['x'] = f['reconstructed_image/x_coordinates'][:]
                    if 'y_coordinates' in f['reconstructed_image']:
                        result['positions']['y'] = f['reconstructed_image/y_coordinates'][:]

                    # If we have the image but scan_type wasn't set, it's definitely 2D
                    if scan_type == 'UNKNOWN':
                        result['scan_type'] = '2D'

                    print(f"Loaded 2D image with shape: {result['data'].shape}")

                except Exception as e:
                    print(f"Warning: Failed to read reconstructed image: {e}")
                    # data is already None, will try raw data next

            # Read raw data for line scans OR if reconstructed image failed/unavailable
            if result.get('data') is None and 'raw_data' in f:
                try:
                    raw = f['raw_data']

                    if 'x_positions' in raw and 'currents' in raw:
                        # Extract positions and currents
                        x_pos = raw['x_positions'][:]
                        y_pos = raw['y_positions'][:]
                        z_pos = raw['z_positions'][:]
                        currents = raw['currents'][:]

                        # Determine scan axis from metadata
                        scan_axis = result['metadata'].get('scan_axis', 'Y')
                        if isinstance(scan_axis, bytes):
                            scan_axis = scan_axis.decode('utf-8')

                        # Also check 'Moving axis' metadata for Z-scans
                        moving_axis = result['metadata'].get('Moving axis', scan_axis)
                        if isinstance(moving_axis, bytes):
                            moving_axis = moving_axis.decode('utf-8')

                        # For Z-scan, use Z positions; otherwise use scan_axis
                        if scan_type == 'Z' or moving_axis == 'Z' or scan_axis == 'Z':
                            positions = z_pos
                            result['scan_type'] = 'Z'  # Ensure it's marked as Z
                            print(f"Detected Z-scan from axis: {moving_axis}")
                        elif scan_axis == 'X' or moving_axis == 'X':
                            positions = x_pos
                            print(f"Detected X line scan")
                        else:  # Y or default
                            positions = y_pos
                            print(f"Detected Y line scan")

                        # Remove NaN positions and corresponding currents
                        valid_mask = ~np.isnan(positions)
                        positions = positions[valid_mask]
                        currents = currents[valid_mask]

                        # Also remove NaN currents
                        valid_mask = ~np.isnan(currents)
                        positions = positions[valid_mask]
                        currents = currents[valid_mask]

                        if len(positions) == 0:
                            result['success'] = False
                            result['error'] = 'No valid data points found (all NaN)'
                            return result

                        # Average duplicate positions (important for step-and-stop scans)
                        unique_pos = np.unique(positions)

                        signals = np.full(len(unique_pos), np.nan)
                        for i, pos in enumerate(unique_pos):
                            mask = np.abs(positions - pos) < 1.0  # 1nm tolerance
                            if np.any(mask):
                                signals[i] = np.nanmean(currents[mask])

                        result['data'] = {
                            'positions': unique_pos,
                            'signals': signals
                        }
                        result['positions']['scan_axis'] = positions
                        result['positions']['currents'] = currents

                        print(f"Loaded 1D data with {len(unique_pos)} unique points from {len(positions)} raw points")

                    else:
                        result['success'] = False
                        result['error'] = 'raw_data group missing required datasets (x_positions, currents)'
                        return result

                except Exception as e:
                    print(f"Error reading raw data: {e}")
                    import traceback
                    traceback.print_exc()
                    result['success'] = False
                    result['error'] = f'Failed to read raw data from H5 file: {e}'
                    return result

            # Final check: if still no data, fail
            if result.get('data') is None:
                result['success'] = False
                result['error'] = 'No valid data found in H5 file (no reconstructed_image or raw_data)'
                print(f"Available groups in file: {list(f.keys())}")
                if 'raw_data' in f:
                    print(f"raw_data datasets: {list(f['raw_data'].keys())}")
                return result

        return result

    @staticmethod
    def _read_csv(filepath: str) -> dict:
        """Read CSV scan file"""
        result = {
            'success': True,
            'filepath': filepath,
            'scan_id': os.path.basename(filepath).replace('.csv', ''),
            'metadata': {},
            'positions': {}
        }

        # Read file and parse header
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Parse metadata from comments
        data_start = 0
        z_position = None
        r_position = None

        for i, line in enumerate(lines):
            if line.startswith('#'):
                # Parse metadata
                if ':' in line:
                    parts = line[1:].split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        result['metadata'][key] = value

                        # PARSE Z POSITION
                        if 'z position' in key.lower() or key.lower() == 'z position':
                            try:
                                # Extract numeric value, handling units
                                z_str = value.replace('nm', '').replace('NM', '').strip()
                                z_position = float(z_str)
                            except (ValueError, AttributeError):
                                pass

                        # PARSE R POSITION
                        if 'r position' in key.lower() or key.lower() == 'r position':
                            try:
                                # Extract numeric value, handling units (deg or μdeg)
                                r_str = value.replace('deg', '').replace('°', '').replace('Â°', '').strip()
                                r_value = float(r_str)
                                # If value is small, assume it's in degrees, otherwise microdegrees
                                if abs(r_value) < 720:
                                    r_position = r_value * 1e6  # Convert to microdegrees for consistency
                                else:
                                    r_position = r_value
                            except (ValueError, AttributeError):
                                pass
            else:
                data_start = i
                break

        # Store parsed Z and R positions in metadata
        if z_position is not None:
            result['metadata']['z_position'] = z_position
        if r_position is not None:
            result['metadata']['r_position'] = r_position

        # Determine scan type from metadata or data format
        if 'Scan type' in result['metadata'] and 'Z-scan' in result['metadata']['Scan type']:
            result['scan_type'] = 'Z'
        elif 'Scan type' in result['metadata'] and 'line' in result['metadata']['Scan type'].lower():
            result['scan_type'] = 'LINE'
        else:
            # Try to infer from data columns
            header_line = lines[data_start] if data_start < len(lines) else ""
            if 'Z_nm' in header_line:
                result['scan_type'] = 'Z'
            elif 'Position_nm' in header_line:
                result['scan_type'] = 'LINE'
            elif 'X_nm' in header_line and 'Y_nm' in header_line:
                result['scan_type'] = '2D'
            else:
                result['scan_type'] = 'unknown'

        # Load data
        try:
            data = np.genfromtxt(filepath, delimiter=',', skip_header=data_start + 1)

            if result['scan_type'] in ['LINE', 'Z']:
                # 1D data: positions, signals
                result['data'] = {
                    'positions': data[:, 0],
                    'signals': data[:, 1]
                }
            elif result['scan_type'] == '2D':
                # 2D data: need to reshape
                # Format: X_nm, Y_nm, Signal_nA
                x_unique = np.unique(data[:, 0])
                y_unique = np.unique(data[:, 1])

                image = np.full((len(y_unique), len(x_unique)), np.nan)
                for row in data:
                    xi = np.argmin(np.abs(x_unique - row[0]))
                    yi = np.argmin(np.abs(y_unique - row[1]))
                    image[yi, xi] = row[2]

                result['data'] = image
                result['positions']['x'] = x_unique
                result['positions']['y'] = y_unique

        except Exception as e:
            result['success'] = False
            result['error'] = f'Error parsing CSV data: {str(e)}'

        return result

    @staticmethod
    def _read_npz(filepath: str) -> dict:
        """Read NPZ file (for R+Z polar data)"""
        result = {
            'success': True,
            'filepath': filepath,
            'scan_id': os.path.basename(filepath).replace('.npz', ''),
            'scan_type': 'RZ',
            'metadata': {},
            'positions': {}
        }

        try:
            data = np.load(filepath)

            result['data'] = {
                'r_angles': data['r_angles_deg'],
                'z_radii': data['z_radii_nm'],
                'matrix': data['polar_data_matrix']
            }

            # Store metadata if available
            if 'r_step_deg' in data:
                result['metadata']['r_step_deg'] = float(data['r_step_deg'])
            if 'min_z_origin' in data:
                result['metadata']['min_z_origin'] = float(data['min_z_origin'])

        except Exception as e:
            result['success'] = False
            result['error'] = f'Error reading NPZ: {str(e)}'

        return result


class UnifiedImageViewer(QMainWindow):
    """Unified viewer for all scan types with file browsing"""

    # NEW: Signal to send ROI back to main window
    roi_selected = pyqtSignal(float, float, float, float)  # x_start, x_end, y_start, y_end

    def __init__(self, filepath=None, parent=None):
        super().__init__(parent)

        # Set window flags for proper standalone behavior
        # This makes it appear in taskbar and allows minimize/maximize
        self.setWindowFlags(
            Qt.Window |  # Make it a top-level window
            Qt.WindowMinimizeButtonHint |  # Enable minimize button
            Qt.WindowMaximizeButtonHint |  # Enable maximize button
            Qt.WindowCloseButtonHint  # Enable close button
        )

        self.setWindowTitle("Image Viewer")
        self.resize(1000, 700)
        self.setAttribute(Qt.WA_DeleteOnClose)

        # Store loaded scans for overlay
        self.loaded_scans = []
        self.current_scan_index = 0

        self.setup_ui()

        # Load file if provided
        if filepath:
            self.load_file(filepath)

    def setup_ui(self):
        """Setup the viewer UI"""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # File control bar
        file_bar = QHBoxLayout()

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_file)
        file_bar.addWidget(self.browse_btn)

        self.recent_combo = QComboBox()
        self.recent_combo.setMinimumWidth(300)
        self.recent_combo.addItem("Recent files...")
        self.recent_combo.currentTextChanged.connect(self.load_recent_file)
        file_bar.addWidget(self.recent_combo)

        file_bar.addWidget(QLabel("Loaded:"))
        self.scan_label = QLabel("None")
        self.scan_label.setStyleSheet("QLabel { font-weight: bold; }")
        file_bar.addWidget(self.scan_label)

        file_bar.addStretch()

        self.overlay_cb = QCheckBox("Overlay Mode")
        self.overlay_cb.setToolTip("Keep previous scans when loading new ones")
        self.overlay_cb.stateChanged.connect(self.on_overlay_mode_changed)
        file_bar.addWidget(self.overlay_cb)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all_scans)
        file_bar.addWidget(self.clear_btn)

        layout.addLayout(file_bar)

        # Tab widget
        self.tabs = QTabWidget()

        # Tab 1: 2D Image View
        self.image_tab = self.create_image_tab()
        self.tabs.addTab(self.image_tab, "Image View")

        # Tab 2: Line Plot View
        self.line_tab = self.create_line_tab()
        self.tabs.addTab(self.line_tab, "Line Plot")

        # Tab 3: Polar View
        self.polar_tab = self.create_polar_tab()
        self.tabs.addTab(self.polar_tab, "Polar View")

        # Tab 4: Metadata
        self.metadata_tab = self.create_metadata_tab()
        self.tabs.addTab(self.metadata_tab, "Metadata")

        # Initially disable all tabs
        for i in range(self.tabs.count()):
            self.tabs.setTabEnabled(i, False)

        layout.addWidget(self.tabs)

        # Status bar
        self.status_label = QLabel("No file loaded")
        self.status_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; }")
        layout.addWidget(self.status_label)

    def create_image_tab(self):
        """Create 2D image view tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Matplotlib figure
        self.image_fig = Figure(figsize=(8, 6))
        self.image_canvas = FigureCanvas(self.image_fig)
        self.image_toolbar = NavigationToolbar(self.image_canvas, widget)

        layout.addWidget(self.image_toolbar)
        layout.addWidget(self.image_canvas)

        # Controls
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Colormap:"))
        self.image_cmap_combo = QComboBox()
        self.image_cmap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'gray', 'hot', 'jet'])
        self.image_cmap_combo.currentTextChanged.connect(self.update_image_display)
        controls.addWidget(self.image_cmap_combo)

        self.image_autoscale_btn = QPushButton("Auto Levels")
        self.image_autoscale_btn.clicked.connect(self.autoscale_image)
        controls.addWidget(self.image_autoscale_btn)

        self.image_snap_cb = QCheckBox("Snap to Pixels")
        self.image_snap_cb.setChecked(False)
        self.image_snap_cb.setToolTip("Show only actual measured pixel values")
        controls.addWidget(self.image_snap_cb)

        # NEW: ROI Selection button
        self.roi_select_btn = QPushButton("Select ROI")
        self.roi_select_btn.setCheckable(True)
        self.roi_select_btn.setToolTip("Draw rectangle to select scan region")
        self.roi_select_btn.clicked.connect(self.toggle_roi_selector)
        controls.addWidget(self.roi_select_btn)

        # NEW: Send to Scan button (initially hidden)
        self.send_roi_btn = QPushButton("→ Send to Scan")
        self.send_roi_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.send_roi_btn.setToolTip("Send selected region to 2D Scan Control")
        self.send_roi_btn.clicked.connect(self.send_roi_to_scan)
        self.send_roi_btn.setVisible(False)
        controls.addWidget(self.send_roi_btn)

        self.image_export_btn = QPushButton("Export Image")
        self.image_export_btn.clicked.connect(self.export_image)
        controls.addWidget(self.image_export_btn)

        controls.addStretch()
        layout.addLayout(controls)

        # Hover status
        self.image_hover_label = QLabel("Hover over image for details")
        self.image_hover_label.setStyleSheet("QLabel { background-color: #e8f4e8; padding: 3px; }")
        layout.addWidget(self.image_hover_label)

        # NEW: ROI info label (initially hidden)
        self.roi_info_label = QLabel("")
        self.roi_info_label.setStyleSheet("QLabel { background-color: #fff4e8; padding: 5px; font-weight: bold; }")
        self.roi_info_label.setVisible(False)
        layout.addWidget(self.roi_info_label)

        # Initialize ROI selector variables
        self.rect_selector = None
        self.roi_coordinates = None

        return widget

    def create_line_tab(self):
        """Create 1D line plot tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Matplotlib figure
        self.line_fig = Figure(figsize=(10, 6))
        self.line_canvas = FigureCanvas(self.line_fig)
        self.line_toolbar = NavigationToolbar(self.line_canvas, widget)

        layout.addWidget(self.line_toolbar)
        layout.addWidget(self.line_canvas)

        # Controls
        controls = QHBoxLayout()

        self.line_export_btn = QPushButton("Export Data")
        self.line_export_btn.clicked.connect(self.export_line_data)
        controls.addWidget(self.line_export_btn)

        self.line_clear_btn = QPushButton("Clear Overlays")
        self.line_clear_btn.clicked.connect(self.clear_line_overlays)
        controls.addWidget(self.line_clear_btn)

        controls.addWidget(QLabel("Legend:"))
        self.line_legend_cb = QCheckBox("Show")
        self.line_legend_cb.setChecked(True)
        self.line_legend_cb.stateChanged.connect(self.update_line_display)
        controls.addWidget(self.line_legend_cb)

        #Snap to data points checkbox
        self.line_snap_cb = QCheckBox("Snap to Data")
        self.line_snap_cb.setChecked(False)
        self.line_snap_cb.setToolTip("Show only actual measured data points")
        controls.addWidget(self.line_snap_cb)

        controls.addStretch()
        layout.addLayout(controls)

        # Hover status
        self.line_hover_label = QLabel("Hover over plot for details")
        self.line_hover_label.setStyleSheet("QLabel { background-color: #e8f4e8; padding: 3px; }")
        layout.addWidget(self.line_hover_label)

        return widget

    def create_polar_tab(self):
        """Create R+Z polar view tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Matplotlib figure
        self.polar_fig = Figure(figsize=(8, 8))
        self.polar_canvas = FigureCanvas(self.polar_fig)
        self.polar_toolbar = NavigationToolbar(self.polar_canvas, widget)

        layout.addWidget(self.polar_toolbar)
        layout.addWidget(self.polar_canvas)

        # Controls
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Colormap:"))
        self.polar_cmap_combo = QComboBox()
        self.polar_cmap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'gray', 'hot'])
        self.polar_cmap_combo.currentTextChanged.connect(self.update_polar_display)
        controls.addWidget(self.polar_cmap_combo)

        self.polar_export_btn = QPushButton("Export Polar Data")
        self.polar_export_btn.clicked.connect(self.export_polar_data)
        controls.addWidget(self.polar_export_btn)

        controls.addStretch()
        layout.addLayout(controls)

        return widget

    def create_metadata_tab(self):
        """Create metadata display tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        self.metadata_text.setFont(QFont("Monospace", 9))
        layout.addWidget(self.metadata_text)

        return widget

    def browse_file(self):
        """Browse for scan file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Scan File", "",
            "Scan Files (*.h5 *.csv *.npz);;HDF5 Files (*.h5);;CSV Files (*.csv);;NPZ Files (*.npz);;All Files (*)"
        )

        if filepath:
            self.load_file(filepath)

    def load_file(self, filepath: str):
        """Load and display scan file"""
        if not os.path.exists(filepath):
            QMessageBox.warning(self, "Error", f"File not found: {filepath}")
            return

        # Read file
        self.status_label.setText(f"Loading {os.path.basename(filepath)}...")
        QApplication.processEvents()

        data = DataFileReader.read_file(filepath)

        if not data['success']:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{data.get('error', 'Unknown error')}")
            self.status_label.setText("Load failed")
            return

        # Handle overlay mode
        if not self.overlay_cb.isChecked():
            self.loaded_scans.clear()

        self.loaded_scans.append(data)
        self.current_scan_index = len(self.loaded_scans) - 1

        # Update UI
        scan_ids = [s['scan_id'] for s in self.loaded_scans]
        self.scan_label.setText(", ".join(scan_ids))

        # BUILD ENHANCED TITLE WITH Z AND R INFORMATION
        title = f"Image Viewer - {data['scan_id']}"

        # Extract Z and R from metadata
        metadata = data.get('metadata', {})
        scan_type = data['scan_type']

        # For Z-scans, only show R (Z is what's being scanned)
        if scan_type == 'Z':
            r_pos = metadata.get('r_position')
            if r_pos is not None:
                # Convert to degrees if in microdegrees
                r_deg = r_pos / 1e6 if abs(r_pos) > 720 else r_pos
                title += f" | R={r_deg:.3f}°"

        # For 2D and LINE scans, show both Z and R
        else:
            z_pos = metadata.get('z_position')
            r_pos = metadata.get('r_position')

            position_parts = []
            if z_pos is not None:
                position_parts.append(f"Z={z_pos:.0f}nm")
            if r_pos is not None:
                # Convert to degrees if in microdegrees
                r_deg = r_pos / 1e6 if abs(r_pos) > 720 else r_pos
                position_parts.append(f"R={r_deg:.3f}°")

            if position_parts:
                title += " | " + " | ".join(position_parts)

        self.setWindowTitle(title)

        # Add to recent files
        if filepath not in [self.recent_combo.itemText(i) for i in range(self.recent_combo.count())]:
            self.recent_combo.addItem(filepath)

        # Display based on scan type (check both data and scan_type)
        scan_type = data['scan_type']

        # Debug print
        print(f"Loading scan type: {scan_type}, data type: {type(data.get('data'))}")

        # Check if it's 2D based on data structure
        is_2d = (scan_type == '2D' or
                 (isinstance(data.get('data'), np.ndarray) and data['data'].ndim == 2))

        # Check if it's 1D based on data structure
        is_1d = (scan_type in ['LINE', 'Z'] or
                 (isinstance(data.get('data'), dict) and 'positions' in data['data']))

        if is_2d and isinstance(data.get('data'), np.ndarray):
            self.tabs.setTabEnabled(0, True)
            self.tabs.setTabEnabled(1, False)
            self.tabs.setTabEnabled(2, False)
            self.tabs.setCurrentIndex(0)
            self.display_2d_data(data)

        elif is_1d and isinstance(data.get('data'), dict):
            self.tabs.setTabEnabled(0, False)
            self.tabs.setTabEnabled(1, True)
            self.tabs.setTabEnabled(2, False)
            self.tabs.setCurrentIndex(1)
            self.display_line_data(data)

        elif scan_type == 'RZ':
            self.tabs.setTabEnabled(0, False)
            self.tabs.setTabEnabled(1, False)
            self.tabs.setTabEnabled(2, True)
            self.tabs.setCurrentIndex(2)
            self.display_polar_data(data)

        else:
            QMessageBox.warning(self, "Warning",
                                f"Unknown scan type or data format: {scan_type}\n"
                                f"Data structure: {type(data.get('data'))}")
            self.status_label.setText(f"Unknown format: {scan_type}")
            # Still enable metadata tab
            self.tabs.setTabEnabled(3, True)
            self.display_metadata(data)
            return

        # Always enable metadata tab
        self.tabs.setTabEnabled(3, True)
        self.display_metadata(data)

        self.status_label.setText(f"Loaded: {data['scan_id']} ({scan_type} scan)")

    def display_2d_data(self, data: dict):
        """Display 2D image data"""
        self.image_fig.clear()
        ax = self.image_fig.add_subplot(111)

        image = data['data']

        # Get extent if available
        if 'x' in data['positions'] and 'y' in data['positions']:
            x_coords = data['positions']['x']
            y_coords = data['positions']['y']
            extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]
        else:
            extent = None

        # Plot image
        im = ax.imshow(image, cmap=self.image_cmap_combo.currentText(),
                       origin='lower', aspect='equal', extent=extent,
                       interpolation='nearest')

        self.image_fig.colorbar(im, ax=ax, label='Signal (nA)')

        ax.set_xlabel('X Position (nm)')
        ax.set_ylabel('Y Position (nm)')

        # BUILD ENHANCED PLOT TITLE WITH Z AND R INFORMATION
        metadata = data.get('metadata', {})
        plot_title = data['scan_id']

        z_pos = metadata.get('z_position')
        r_pos = metadata.get('r_position')

        position_parts = []
        if z_pos is not None:
            position_parts.append(f"Z={z_pos:.1f}nm")
        if r_pos is not None:
            r_deg = r_pos / 1e6 if abs(r_pos) > 720 else r_pos
            position_parts.append(f"R={r_deg:.3f}°")

        if position_parts:
            plot_title += " | " + " | ".join(position_parts)

        ax.set_title(plot_title)

        # Create cursor marker (red dot with white edge, initially hidden)
        self.image_cursor_marker = ax.plot([], [], 'o', markersize=8,
                                           markerfacecolor='red',
                                           markeredgecolor='white',
                                           markeredgewidth=2, zorder=10)[0]

        # Store for hover
        self.current_image_data = {
            'image': image,
            'extent': extent,
            'ax': ax
        }

        # Connect hover
        try:
            self.image_canvas.mpl_disconnect(self.image_hover_cid)
        except (AttributeError, KeyError):
            pass
        self.image_hover_cid = self.image_canvas.mpl_connect('motion_notify_event', self.on_image_hover)

        self.image_fig.tight_layout()
        self.image_canvas.draw()

    def toggle_roi_selector(self, checked):
        """Enable/disable ROI rectangle selector"""
        if checked:
            # Clear any existing rectangle patches first
            if hasattr(self, 'current_image_data') and self.current_image_data['ax']:
                ax = self.current_image_data['ax']

                # Remove old selector rectangle if it exists
                if self.rect_selector:
                    self.rect_selector.set_active(False)
                    # Remove the rectangle artist from the axes
                    if hasattr(self.rect_selector, 'artists'):
                        for artist in self.rect_selector.artists:
                            artist.remove()
                    self.rect_selector = None

                # Also remove any lingering rectangle patches
                for patch in ax.patches[:]:  # Use slice to avoid modifying list during iteration
                    patch.remove()

                self.image_canvas.draw_idle()

                # Create new rectangle selector
                self.rect_selector = RectangleSelector(
                    ax,
                    self.on_roi_selected,
                    useblit=True,
                    button=[1],  # Left mouse button
                    minspanx=5, minspany=5,
                    spancoords='pixels',
                    interactive=True,
                    props=dict(facecolor='red', edgecolor='red',
                               alpha=0.3, fill=True, linewidth=2)
                )

                self.roi_select_btn.setText("Drawing ROI...")
                self.roi_select_btn.setStyleSheet("QPushButton { background-color: #ff9800; }")
                self.image_hover_label.setText("Click and drag to draw selection rectangle")

                # Disconnect hover to avoid conflicts
                try:
                    self.image_canvas.mpl_disconnect(self.image_hover_cid)
                except (AttributeError, KeyError):
                    pass
            else:
                QMessageBox.warning(self, "No Image", "Load a 2D image first")
                self.roi_select_btn.setChecked(False)
        else:
            # Disable selector and clean up
            if self.rect_selector:
                self.rect_selector.set_active(False)
                # Remove the rectangle artist
                if hasattr(self.rect_selector, 'artists'):
                    for artist in self.rect_selector.artists:
                        try:
                            artist.remove()
                        except:
                            pass
                self.rect_selector = None

            # Also remove any lingering rectangle patches
            if hasattr(self, 'current_image_data') and self.current_image_data['ax']:
                ax = self.current_image_data['ax']
                for patch in ax.patches[:]:
                    try:
                        patch.remove()
                    except:
                        pass

            self.roi_select_btn.setText("Select ROI")
            self.roi_select_btn.setStyleSheet("")
            self.send_roi_btn.setVisible(False)
            self.roi_info_label.setVisible(False)
            self.roi_coordinates = None

            # Reconnect hover
            if hasattr(self, 'current_image_data'):
                try:
                    self.image_hover_cid = self.image_canvas.mpl_connect('motion_notify_event', self.on_image_hover)
                except:
                    pass

            self.image_canvas.draw_idle()

    def on_roi_selected(self, eclick, erelease):
        """Callback when ROI rectangle is drawn"""
        # Get rectangle coordinates
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # Ensure x1 < x2 and y1 < y2
        x_start = min(x1, x2)
        x_end = max(x1, x2)
        y_start = min(y1, y2)
        y_end = max(y1, y2)

        # Store coordinates
        self.roi_coordinates = (x_start, x_end, y_start, y_end)

        # Calculate dimensions
        width = x_end - x_start
        height = y_end - y_start

        # Update info label
        self.roi_info_label.setText(
            f"Selected ROI: X=[{x_start:.1f}, {x_end:.1f}] nm | "
            f"Y=[{y_start:.1f}, {y_end:.1f}] nm "
        )
        self.roi_info_label.setVisible(True)

        # Show send button
        self.send_roi_btn.setVisible(True)

        # Update status
        self.image_hover_label.setText(
            f"ROI selected: {width:.1f} × {height:.1f} nm | Click 'Send to Scan' to use these coordinates"
        )

        self.image_canvas.draw_idle()

    def send_roi_to_scan(self):
        """Send ROI coordinates to main window scan control"""
        if self.roi_coordinates is None:
            QMessageBox.warning(self, "No ROI", "Draw a rectangle first")
            return

        x_start, x_end, y_start, y_end = self.roi_coordinates

        # Emit signal with coordinates
        self.roi_selected.emit(x_start, x_end, y_start, y_end)

        # Show confirmation
        self.status_label.setText(
            f"Sent ROI to Scan Control: X=[{x_start:.1f}, {x_end:.1f}], Y=[{y_start:.1f}, {y_end:.1f}]"
        )

        # Flash the button to show action completed
        self.send_roi_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        QTimer.singleShot(500, lambda: self.send_roi_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
        ))

    def display_line_data(self, data: dict):
        """Display 1D line data with overlay support"""
        # Clear only if not in overlay mode or first scan
        if not self.overlay_cb.isChecked() or len(self.loaded_scans) == 1:
            self.line_fig.clear()
            self.line_ax = self.line_fig.add_subplot(111)
            # Store all scan data for snapping
            self.line_scan_data = []

        # Get data
        positions = data['data']['positions']
        signals = data['data']['signals']

        # Store for snapping
        if not hasattr(self, 'line_scan_data'):
            self.line_scan_data = []
        self.line_scan_data.append({
            'positions': positions,
            'signals': signals,
            'scan_id': data['scan_id']
        })

        # Plot with unique color
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_idx = (len(self.loaded_scans) - 1) % len(color_cycle)
        color = color_cycle[color_idx]

        label = data['scan_id']
        line, = self.line_ax.plot(positions, signals, '-o', markersize=3,
                                  linewidth=1.5, label=label, color=color)

        # Store line object for highlighting
        if not hasattr(self, 'line_plot_objects'):
            self.line_plot_objects = []
        self.line_plot_objects.append(line)

        # Create cursor line and marker (initially hidden)
        if not hasattr(self, 'line_cursor_vline'):
            self.line_cursor_vline = self.line_ax.axvline(x=0, color='red',
                                                          linestyle='--', linewidth=2, alpha=0.7)
            self.line_cursor_vline.set_visible(False)

            self.line_cursor_marker = self.line_ax.plot([], [], 'ro', markersize=8,
                                                        markerfacecolor='red',
                                                        markeredgecolor='white',
                                                        markeredgewidth=2, zorder=10)[0]

        # Labels
        if data['scan_type'] == 'Z':
            self.line_ax.set_xlabel('Z Position (nm)')
        else:
            self.line_ax.set_xlabel('Position (nm)')

        self.line_ax.set_ylabel('Signal (nA)')

        # BUILD ENHANCED PLOT TITLE WITH Z AND R INFORMATION
        metadata = data.get('metadata', {})
        scan_type = data['scan_type']

        if len(self.loaded_scans) > 1:
            # Multiple scans overlaid
            plot_title = f"Line Scan Viewer ({len(self.loaded_scans)} scans)"
        else:
            # Single scan - show detailed info
            plot_title = data['scan_id']

            # For Z-scans, only show R
            if scan_type == 'Z':
                r_pos = metadata.get('r_position')
                if r_pos is not None:
                    r_deg = r_pos / 1e6 if abs(r_pos) > 720 else r_pos
                    plot_title += f" | R={r_deg:.3f}°"

            # For LINE scans, show both Z and R
            else:
                z_pos = metadata.get('z_position')
                r_pos = metadata.get('r_position')

                position_parts = []
                if z_pos is not None:
                    position_parts.append(f"Z={z_pos:.1f}nm")
                if r_pos is not None:
                    r_deg = r_pos / 1e6 if abs(r_pos) > 720 else r_pos
                    position_parts.append(f"R={r_deg:.3f}°")

                if position_parts:
                    plot_title += " | " + " | ".join(position_parts)

        self.line_ax.set_title(plot_title)
        self.line_ax.grid(True, alpha=0.3)

        # Legend
        if self.line_legend_cb.isChecked() and len(self.loaded_scans) > 1:
            self.line_ax.legend()

        # Connect hover (disconnect old first to avoid duplicates)
        try:
            self.line_canvas.mpl_disconnect(self.line_hover_cid)
        except (AttributeError, KeyError):
            pass
        self.line_hover_cid = self.line_canvas.mpl_connect('motion_notify_event', self.on_line_hover)

        #Set tight axis limits to avoid including 0 when data doesn't include it
        if len(self.loaded_scans) == 1:  # Only for single scans to avoid breaking overlays
            if len(positions) > 0 and len(signals[~np.isnan(signals)]) > 0:
                valid_signals = signals[~np.isnan(signals)]

                # X-axis: use actual position range with small margin
                x_margin = (np.max(positions) - np.min(positions)) * 0.05
                self.line_ax.set_xlim(np.min(positions) - x_margin,
                                      np.max(positions) + x_margin)

                # Y-axis: use actual signal range with small margin
                y_margin = (np.max(valid_signals) - np.min(valid_signals)) * 0.05
                self.line_ax.set_ylim(np.min(valid_signals) - y_margin,
                                      np.max(valid_signals) + y_margin)

        self.line_fig.tight_layout()
        self.line_canvas.draw()

    def display_polar_data(self, data: dict):
        """Display R+Z polar data"""
        self.polar_fig.clear()
        ax = self.polar_fig.add_subplot(111)

        # Get data
        r_angles = data['data']['r_angles']
        z_radii = data['data']['z_radii']
        matrix = data['data']['matrix']

        # Create polar plot (simplified - convert to cartesian)
        min_z = np.min(z_radii)
        max_z = np.max(z_radii)

        # Create meshgrid
        theta = np.deg2rad(r_angles)
        r = z_radii - min_z

        # Plot as pcolormesh
        for i, z_val in enumerate(r):
            for j, angle in enumerate(theta):
                x = z_val * np.cos(angle)
                y = z_val * np.sin(angle)
                # This is simplified - proper polar plotting would be more complex

        # For now, show as regular image
        im = ax.imshow(matrix, cmap=self.polar_cmap_combo.currentText(),
                       aspect='auto', origin='lower')

        self.polar_fig.colorbar(im, ax=ax, label='Signal (nA)')
        ax.set_xlabel('R Index')
        ax.set_ylabel('Z Index')
        ax.set_title(f"Polar R+Z: {data['scan_id']}")

        self.polar_fig.tight_layout()
        self.polar_canvas.draw()

    def display_metadata(self, data: dict):
        """Display scan metadata"""
        text = f"=== Scan Information ===\n\n"
        text += f"Scan ID: {data['scan_id']}\n"
        text += f"Scan Type: {data['scan_type']}\n"
        text += f"File: {data['filepath']}\n\n"

        text += f"=== Metadata ===\n\n"
        for key, value in data['metadata'].items():
            text += f"{key}: {value}\n"

        if 'data' in data:
            text += f"\n=== Data Summary ===\n\n"
            if isinstance(data['data'], dict):
                if 'positions' in data['data']:
                    text += f"Points: {len(data['data']['positions'])}\n"
                    text += f"Position range: {np.min(data['data']['positions']):.1f} to {np.max(data['data']['positions']):.1f} nm\n"
                if 'signals' in data['data']:
                    signals = data['data']['signals']
                    valid = signals[~np.isnan(signals)]
                    if len(valid) > 0:
                        text += f"Signal range: {np.min(valid):.4f} to {np.max(valid):.4f} nA\n"
                        text += f"Signal mean: {np.mean(valid):.4f} nA\n"
            else:
                text += f"Image shape: {data['data'].shape}\n"
                valid = data['data'][~np.isnan(data['data'])]
                if len(valid) > 0:
                    text += f"Signal range: {np.min(valid):.4f} to {np.max(valid):.4f} nA\n"
                    text += f"Signal mean: {np.mean(valid):.4f} nA\n"

        self.metadata_text.setPlainText(text)

    def on_image_hover(self, event):
        """Handle hover over 2D image with optional snap-to-pixel marker"""
        if not event.inaxes or not hasattr(self, 'current_image_data'):
            # Hide marker when outside
            if hasattr(self, 'image_cursor_marker'):
                self.image_cursor_marker.set_visible(False)
                self.image_canvas.draw_idle()
            return

        x, y = event.xdata, event.ydata
        image = self.current_image_data['image']
        extent = self.current_image_data['extent']

        # Convert to pixel coordinates
        if extent:
            h, w = image.shape
            xi_float = (x - extent[0]) / (extent[1] - extent[0]) * w
            yi_float = (y - extent[2]) / (extent[3] - extent[2]) * h

            if self.image_snap_cb.isChecked():
                # Snap to nearest pixel
                xi = int(round(xi_float))
                yi = int(round(yi_float))

                # Calculate pixel center position for marker
                x_center = extent[0] + (xi + 0.5) * (extent[1] - extent[0]) / w
                y_center = extent[2] + (yi + 0.5) * (extent[3] - extent[2]) / h
            else:
                xi = int(xi_float)
                yi = int(yi_float)
                x_center = x
                y_center = y
        else:
            xi = int(round(event.xdata)) if self.image_snap_cb.isChecked() else int(event.xdata)
            yi = int(round(event.ydata)) if self.image_snap_cb.isChecked() else int(event.ydata)
            x_center = xi
            y_center = yi

        # Check bounds
        if 0 <= xi < image.shape[1] and 0 <= yi < image.shape[0]:
            value = image[yi, xi]

            # Update marker position if snap is enabled
            if self.image_snap_cb.isChecked() and hasattr(self, 'image_cursor_marker'):
                self.image_cursor_marker.set_data([x_center], [y_center])
                self.image_cursor_marker.set_visible(True)
                self.image_canvas.draw_idle()
            else:
                if hasattr(self, 'image_cursor_marker'):
                    self.image_cursor_marker.set_visible(False)

            # Update text (keep original order)
            if not np.isnan(value):
                self.image_hover_label.setText(
                    f"Position: ({x:.1f}, {y:.1f}) nm | Pixel: [{xi}, {yi}] | Signal: {value:.4f} nA"
                )
            else:
                self.image_hover_label.setText(
                    f"Position: ({x:.1f}, {y:.1f}) nm | Pixel: [{xi}, {yi}] | No data"
                )
        else:
            if hasattr(self, 'image_cursor_marker'):
                self.image_cursor_marker.set_visible(False)
            self.image_hover_label.setText("Outside image bounds")

    def on_line_hover(self, event):
        """Handle hover over line plot with snap-to-data option"""
        if not event.inaxes or not hasattr(self, 'line_scan_data'):
            # Hide cursor elements
            if hasattr(self, 'line_cursor_vline'):
                self.line_cursor_vline.set_visible(False)
                self.line_cursor_marker.set_visible(False)
                self.line_canvas.draw_idle()
            self.line_hover_label.setText("Hover over plot for details")
            return

        x_cursor = event.xdata

        if self.line_snap_cb.isChecked() and len(self.line_scan_data) > 0:
            # Find nearest point across all loaded scans
            min_distance = float('inf')
            nearest_pos = None
            nearest_sig = None
            nearest_scan = None

            for scan in self.line_scan_data:
                positions = scan['positions']
                signals = scan['signals']

                # Find nearest point in this scan
                distances = np.abs(positions - x_cursor)
                idx = np.argmin(distances)

                if distances[idx] < min_distance:
                    min_distance = distances[idx]
                    nearest_pos = positions[idx]
                    nearest_sig = signals[idx]
                    nearest_scan = scan['scan_id']

            if nearest_pos is not None:
                # Update cursor line and marker
                self.line_cursor_vline.set_xdata([nearest_pos])
                self.line_cursor_vline.set_visible(True)

                self.line_cursor_marker.set_data([nearest_pos], [nearest_sig])
                self.line_cursor_marker.set_visible(True)

                # Update status label
                scan_info = f" ({nearest_scan})" if len(self.line_scan_data) > 1 else ""
                self.line_hover_label.setText(
                    f"Position: {nearest_pos:.2f} nm | Signal: {nearest_sig:.4f} nA{scan_info}"
                )

                self.line_canvas.draw_idle()
        else:
            # Free cursor mode - just show position
            if hasattr(self, 'line_cursor_vline'):
                self.line_cursor_vline.set_visible(False)
                self.line_cursor_marker.set_visible(False)
            self.line_hover_label.setText(f"Position: {x_cursor:.2f} nm")
            self.line_canvas.draw_idle()

    def update_image_display(self):
        """Redraw image with new colormap"""
        if len(self.loaded_scans) > 0 and self.loaded_scans[-1]['scan_type'] == '2D':
            self.display_2d_data(self.loaded_scans[-1])

    def update_line_display(self):
        """Redraw line plot"""
        if len(self.loaded_scans) > 0:
            # Clear and reset
            self.line_fig.clear()
            self.line_ax = self.line_fig.add_subplot(111)
            self.line_scan_data = []
            self.line_plot_objects = []

            # Redraw all scans
            for data in self.loaded_scans:
                if data['scan_type'] in ['LINE', 'Z']:
                    positions = data['data']['positions']
                    signals = data['data']['signals']

                    # Store for snapping
                    self.line_scan_data.append({
                        'positions': positions,
                        'signals': signals,
                        'scan_id': data['scan_id']
                    })

                    # Plot
                    line, = self.line_ax.plot(positions, signals, '-o', markersize=3,
                                              linewidth=1.5, label=data['scan_id'])
                    self.line_plot_objects.append(line)

            # Recreate cursor elements
            self.line_cursor_vline = self.line_ax.axvline(x=0, color='red',
                                                          linestyle='--', linewidth=2, alpha=0.7)
            self.line_cursor_vline.set_visible(False)

            self.line_cursor_marker = self.line_ax.plot([], [], 'ro', markersize=8,
                                                        markerfacecolor='red',
                                                        markeredgecolor='white',
                                                        markeredgewidth=2, zorder=10)[0]

            # Set axis labels (determine based on scan type)
            if len(self.loaded_scans) > 0:
                first_scan = self.loaded_scans[0]
                if first_scan['scan_type'] == 'Z':
                    self.line_ax.set_xlabel('Z Position (nm)')
                else:
                    self.line_ax.set_xlabel('Position (nm)')
            else:
                self.line_ax.set_xlabel('Position (nm)')

            self.line_ax.set_ylabel('Signal (nA)')
            self.line_ax.grid(True, alpha=0.3)

            # ADD TITLE (same logic as display_line_data)
            if len(self.loaded_scans) > 1:
                # Multiple scans overlaid
                plot_title = f"Line Scan Viewer ({len(self.loaded_scans)} scans)"
            else:
                # Single scan - show detailed info
                data = self.loaded_scans[0]
                metadata = data.get('metadata', {})
                scan_type = data['scan_type']

                plot_title = data['scan_id']

                # For Z-scans, only show R
                if scan_type == 'Z':
                    r_pos = metadata.get('r_position')
                    if r_pos is not None:
                        r_deg = r_pos / 1e6 if abs(r_pos) > 720 else r_pos
                        plot_title += f" | R={r_deg:.3f}°"

                # For LINE scans, show both Z and R
                else:
                    z_pos = metadata.get('z_position')
                    r_pos = metadata.get('r_position')

                    position_parts = []
                    if z_pos is not None:
                        position_parts.append(f"Z={z_pos:.1f}nm")
                    if r_pos is not None:
                        r_deg = r_pos / 1e6 if abs(r_pos) > 720 else r_pos
                        position_parts.append(f"R={r_deg:.3f}°")

                    if position_parts:
                        plot_title += " | " + " | ".join(position_parts)

            self.line_ax.set_title(plot_title)

            # Legend
            if self.line_legend_cb.isChecked() and len(self.loaded_scans) > 1:
                self.line_ax.legend()

            # FIX: Apply tight axis limits (same logic as display_line_data)
            if len(self.loaded_scans) == 1:
                data = self.loaded_scans[0]
                if data['scan_type'] in ['LINE', 'Z']:
                    positions = data['data']['positions']
                    signals = data['data']['signals']

                    if len(positions) > 0 and len(signals[~np.isnan(signals)]) > 0:
                        valid_signals = signals[~np.isnan(signals)]

                        # X-axis: use actual position range with small margin
                        x_margin = (np.max(positions) - np.min(positions)) * 0.05
                        self.line_ax.set_xlim(np.min(positions) - x_margin,
                                              np.max(positions) + x_margin)

                        # Y-axis: use actual signal range with small margin
                        y_margin = (np.max(valid_signals) - np.min(valid_signals)) * 0.05
                        self.line_ax.set_ylim(np.min(valid_signals) - y_margin,
                                              np.max(valid_signals) + y_margin)

            self.line_fig.tight_layout()
            self.line_canvas.draw()

    def update_polar_display(self):
        """Redraw polar plot with new colormap"""
        if len(self.loaded_scans) > 0 and self.loaded_scans[-1]['scan_type'] == 'RZ':
            self.display_polar_data(self.loaded_scans[-1])

    def autoscale_image(self):
        """Auto-scale image levels"""
        if hasattr(self, 'current_image_data'):
            image = self.current_image_data['image']
            valid = image[~np.isnan(image)]
            if len(valid) > 0:
                vmin = np.percentile(valid, 1)
                vmax = np.percentile(valid, 99)

                ax = self.current_image_data['ax']
                for im in ax.get_images():
                    im.set_clim(vmin, vmax)

                self.image_canvas.draw()

    def on_overlay_mode_changed(self, state):
        """Handle overlay mode toggle"""
        if state != Qt.Checked:
            # Exiting overlay mode - keep only last scan
            if len(self.loaded_scans) > 1:
                self.loaded_scans = [self.loaded_scans[-1]]
                self.update_line_display()

    def clear_line_overlays(self):
        """Clear all overlaid line scans"""
        if len(self.loaded_scans) > 0:
            self.loaded_scans = [self.loaded_scans[-1]]
            self.update_line_display()

    def clear_all_scans(self):
        """Clear all loaded scans"""
        self.loaded_scans.clear()
        self.scan_label.setText("None")
        self.setWindowTitle("Image Viewer")

        # Clear all displays
        self.image_fig.clear()
        self.image_canvas.draw()
        self.line_fig.clear()
        self.line_canvas.draw()
        self.polar_fig.clear()
        self.polar_canvas.draw()
        self.metadata_text.clear()

        # Disable tabs
        for i in range(3):
            self.tabs.setTabEnabled(i, False)

        self.status_label.setText("All scans cleared")

    def load_recent_file(self, filepath):
        """Load file from recent list"""
        if filepath and filepath != "Recent files...":
            self.load_file(filepath)

    def export_image(self):
        """Export current 2D image"""
        if len(self.loaded_scans) == 0:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "", "PNG (*.png);;TIFF (*.tif);;All Files (*)"
        )

        if filepath:
            self.image_fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self.status_label.setText(f"Exported to {os.path.basename(filepath)}")

    def export_line_data(self):
        """Export line data"""
        if len(self.loaded_scans) == 0:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Line Data", "", "CSV (*.csv);;PNG (*.png);;All Files (*)"
        )

        if filepath:
            if filepath.endswith('.png'):
                self.line_fig.savefig(filepath, dpi=300, bbox_inches='tight')
            else:
                # Export as CSV
                with open(filepath, 'w') as f:
                    f.write("# Combined Line Scan Data\n")
                    for data in self.loaded_scans:
                        if data['scan_type'] in ['LINE', 'Z']:
                            f.write(f"# {data['scan_id']}\n")
                            positions = data['data']['positions']
                            signals = data['data']['signals']
                            for pos, sig in zip(positions, signals):
                                f.write(f"{pos:.4f},{sig:.6f}\n")

            self.status_label.setText(f"Exported to {os.path.basename(filepath)}")

    def export_polar_data(self):
        """Export polar data"""
        if len(self.loaded_scans) == 0:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Polar Data", "", "PNG (*.png);;NPZ (*.npz);;All Files (*)"
        )

        if filepath:
            if filepath.endswith('.png'):
                self.polar_fig.savefig(filepath, dpi=300, bbox_inches='tight')
            else:
                data = self.loaded_scans[-1]['data']
                np.savez(filepath, **data)

            self.status_label.setText(f"Exported to {os.path.basename(filepath)}")


class ImageRegistrationWindow(QMainWindow):
    """Window for analyzing spatial drift between multiple 2D scan images"""

    # Signal to send compensation ratios back to main window
    compensation_ratios_calculated = pyqtSignal(float, float)  # x_ratio, y_ratio

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set window properties
        self.setWindowTitle("Image Registration - Drift Analysis")
        self.setGeometry(150, 150, 1400, 900)
        self.setAttribute(Qt.WA_DeleteOnClose)

        # Data storage
        self.loaded_images = []  # List of dicts with image data
        self.reference_index = 0
        self.drift_results = []  # List of drift calculation results

        # Settings
        self.upsample_factor = 100  # For sub-pixel accuracy
        self.use_roi = False
        self.roi = None  # (x, y, w, h)

        self.setup_ui()
        self.update_ui_state()

    def setup_ui(self):
        """Setup the user interface"""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # File Management Panel
        file_panel = self.create_file_panel()
        layout.addWidget(file_panel)

        # Registration Settings Panel
        settings_panel = self.create_settings_panel()
        layout.addWidget(settings_panel)

        # Results Tabs
        self.results_tabs = QTabWidget()

        # Tab 1: Drift Table
        self.table_tab = self.create_table_tab()
        self.results_tabs.addTab(self.table_tab, "Drift Table")

        # Tab 2: Drift Plots
        self.plots_tab = self.create_plots_tab()
        self.results_tabs.addTab(self.plots_tab, "Drift Plots")

        # Tab 3: Visual Verification
        self.visual_tab = self.create_visual_tab()
        self.results_tabs.addTab(self.visual_tab, "Visual Check")

        # Tab 4: Statistics
        self.stats_tab = self.create_stats_tab()
        self.results_tabs.addTab(self.stats_tab, "Statistics")

        layout.addWidget(self.results_tabs)

        # Export & Analysis Panel
        export_panel = self.create_export_panel()
        layout.addWidget(export_panel)

        # Status bar
        self.status_label = QLabel("Ready - Load images to begin")
        self.status_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; }")
        layout.addWidget(self.status_label)

    def create_file_panel(self):
        """Create file management panel"""
        panel = QGroupBox("File Management")
        layout = QVBoxLayout()

        # Buttons
        button_layout = QHBoxLayout()

        self.add_files_btn = QPushButton("Add Files...")
        self.add_files_btn.clicked.connect(self.add_files)
        button_layout.addWidget(self.add_files_btn)

        self.add_folder_btn = QPushButton("Add Folder...")
        self.add_folder_btn.clicked.connect(self.add_folder)
        button_layout.addWidget(self.add_folder_btn)

        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self.remove_selected)
        button_layout.addWidget(self.remove_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all)
        button_layout.addWidget(self.clear_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # File list table
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(7)
        self.file_table.setHorizontalHeaderLabels([
            "✓", "Filename", "Z (nm)", "R (°)", "Size", "Pixels", "Status"
        ])
        self.file_table.horizontalHeader().setStretchLastSection(True)
        self.file_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.file_table.setMaximumHeight(200)

        layout.addWidget(self.file_table)

        # File count label
        self.file_count_label = QLabel("Loaded Images: 0")
        self.file_count_label.setStyleSheet("QLabel { font-weight: bold; }")
        layout.addWidget(self.file_count_label)

        panel.setLayout(layout)
        panel.setMaximumHeight(300)
        return panel

    def create_settings_panel(self):
        """Create registration settings panel"""
        panel = QGroupBox("Registration Settings")
        layout = QGridLayout()

        # Reference image selection
        layout.addWidget(QLabel("Reference Image:"), 0, 0)
        self.ref_combo = QComboBox()
        self.ref_combo.addItem("First Image")
        self.ref_combo.addItem("Middle Image")
        self.ref_combo.addItem("Last Image")
        self.ref_combo.addItem("Best Quality")
        self.ref_combo.currentIndexChanged.connect(self.on_reference_changed)
        layout.addWidget(self.ref_combo, 0, 1)

        # Method selection
        layout.addWidget(QLabel("Method:"), 0, 2)
        self.method_combo = QComboBox()
        if SKIMAGE_AVAILABLE:
            self.method_combo.addItem("Phase Correlation (FFT)")
        self.method_combo.addItem("Cross-Correlation (Fallback)")
        layout.addWidget(self.method_combo, 0, 3)

        # ROI checkbox (disabled for now)
        self.roi_cb = QCheckBox("Use ROI")
        self.roi_cb.setEnabled(False)
        self.roi_cb.setToolTip("ROI selection coming in Phase 3")
        layout.addWidget(self.roi_cb, 1, 0)

        # Sub-pixel accuracy
        self.subpixel_cb = QCheckBox("Sub-pixel accuracy")
        self.subpixel_cb.setChecked(True)
        layout.addWidget(self.subpixel_cb, 1, 1)

        layout.addWidget(QLabel("Upsample:"), 1, 2)
        self.upsample_spin = QSpinBox()
        self.upsample_spin.setRange(1, 1000)
        self.upsample_spin.setValue(100)
        self.upsample_spin.setSuffix("x")
        self.upsample_spin.setToolTip("Higher = more accurate but slower")
        layout.addWidget(self.upsample_spin, 1, 3)

        # Registration Mode Selection (NEW - add to row 2)
        layout.addWidget(QLabel("Registration Mode:"), 2, 0)
        self.registration_mode_combo = QComboBox()
        self.registration_mode_combo.addItems([
            "Reference-based (all → ref)",
            "Consecutive pairs (i → i-1)",
            "Both methods"
        ])
        self.registration_mode_combo.setCurrentIndex(2)  # Default to "Both"
        self.registration_mode_combo.setToolTip(
            "Reference-based: Register all images to one reference\n"
            "Consecutive pairs: Register each image to previous\n"
            "Both: Calculate using both methods for comparison"
        )
        layout.addWidget(self.registration_mode_combo, 2, 1, 1, 3)

        # Calculate button (moved to row 3)
        self.calculate_btn = QPushButton("Calculate Drift")
        self.calculate_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }"
        )
        self.calculate_btn.clicked.connect(self.calculate_drift)
        layout.addWidget(self.calculate_btn, 3, 0, 1, 4)

        panel.setLayout(layout)
        panel.setMaximumHeight(150)
        return panel

    def create_table_tab(self):
        """Create drift results table tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        self.drift_table = QTableWidget()
        self.drift_table.setColumnCount(9)
        self.drift_table.setHorizontalHeaderLabels([
            "Image", "Index", "Z (nm)", "R (°)", "ΔX (nm)", "ΔY (nm)",
            "|Δ| (nm)", "Correlation", "Status"
        ])
        self.drift_table.horizontalHeader().setStretchLastSection(True)
        self.drift_table.setSelectionBehavior(QTableWidget.SelectRows)

        layout.addWidget(self.drift_table)

        # Copy button
        copy_btn = QPushButton("Copy Table to Clipboard")
        copy_btn.clicked.connect(self.copy_table_to_clipboard)
        layout.addWidget(copy_btn)

        widget.setLayout(layout)
        return widget

    def create_plots_tab(self):
        """Create drift plots tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Create matplotlib figure with 3 subplots
        self.plot_fig = Figure(figsize=(12, 8))
        self.plot_canvas = FigureCanvas(self.plot_fig)

        layout.addWidget(self.plot_canvas)

        # Export plot button
        export_plot_btn = QPushButton("Export Plot as PNG")
        export_plot_btn.clicked.connect(self.export_plot)
        layout.addWidget(export_plot_btn)

        widget.setLayout(layout)
        return widget

    def create_visual_tab(self):
        """Create visual verification tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Controls
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Compare:"))
        self.visual_combo = QComboBox()
        self.visual_combo.currentIndexChanged.connect(self.update_visual_comparison)
        controls.addWidget(self.visual_combo)

        controls.addWidget(QLabel("Mode:"))
        self.visual_mode_combo = QComboBox()
        self.visual_mode_combo.addItems(["Overlay (Blend)", "Difference", "Checkerboard"])
        self.visual_mode_combo.currentTextChanged.connect(self.update_visual_comparison)
        controls.addWidget(self.visual_mode_combo)

        controls.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self.update_visual_comparison)
        controls.addWidget(self.opacity_slider)

        controls.addStretch()
        layout.addLayout(controls)

        # Matplotlib figure for comparison
        self.visual_fig = Figure(figsize=(10, 8))
        self.visual_canvas = FigureCanvas(self.visual_fig)
        layout.addWidget(self.visual_canvas)

        widget.setLayout(layout)
        return widget

    def create_stats_tab(self):
        """Create statistics tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Monospace", 10))
        layout.addWidget(self.stats_text)

        widget.setLayout(layout)
        return widget

    def create_export_panel(self):
        """Create export and analysis panel"""
        panel = QWidget()
        layout = QHBoxLayout()

        self.export_csv_btn = QPushButton("Export Drift CSV")
        self.export_csv_btn.clicked.connect(self.export_drift_csv)
        layout.addWidget(self.export_csv_btn)

        self.apply_compensation_btn = QPushButton("→ Apply Compensation to Scan Settings")
        self.apply_compensation_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; }"
        )
        self.apply_compensation_btn.clicked.connect(self.apply_compensation_to_scan)
        self.apply_compensation_btn.setEnabled(False)
        layout.addWidget(self.apply_compensation_btn)

        layout.addStretch()

        panel.setLayout(layout)
        return panel

    def add_files(self):
        """Add image files"""
        filepaths, _ = QFileDialog.getOpenFileNames(
            self, "Select Image Files", "",
            "Scan Files (*.h5 *.csv);;HDF5 Files (*.h5);;CSV Files (*.csv);;All Files (*)"
        )

        if filepaths:
            self.load_files(filepaths)

    def add_folder(self):
        """Add all image files from folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")

        if folder:
            # Find all .h5 and .csv files
            import glob
            filepaths = []
            filepaths.extend(glob.glob(os.path.join(folder, "*.h5")))
            filepaths.extend(glob.glob(os.path.join(folder, "*.csv")))

            if filepaths:
                self.load_files(filepaths)
            else:
                QMessageBox.warning(self, "No Files", "No .h5 or .csv files found in folder")

    def load_files(self, filepaths):
        """Load multiple image files"""
        self.status_label.setText("Loading files...")
        QApplication.processEvents()

        loaded_count = 0
        for filepath in filepaths:
            data = DataFileReader.read_file(filepath)

            if not data['success']:
                QMessageBox.warning(self, "Load Error",
                                    f"Failed to load {os.path.basename(filepath)}:\n{data.get('error', 'Unknown error')}")
                continue

            # Only accept 2D scans
            if data['scan_type'] != '2D' or not isinstance(data.get('data'), np.ndarray):
                QMessageBox.warning(self, "Invalid Scan Type",
                                    f"{os.path.basename(filepath)} is not a 2D scan (type: {data['scan_type']})")
                continue

            # Extract metadata
            metadata = data.get('metadata', {})
            z_pos = metadata.get('z_position', 'N/A')
            r_pos = metadata.get('r_position', 'N/A')

            # Convert R to degrees if in microdegrees
            if isinstance(r_pos, (int, float)) and abs(r_pos) > 720:
                r_pos = r_pos / 1e6

            image_info = {
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'data': data['data'],
                'scan_type': data['scan_type'],
                'metadata': metadata,
                'z_position': z_pos,
                'r_position': r_pos,
                'shape': data['data'].shape,
                'scan_params': data.get('positions', {}),
                'enabled': True
            }

            self.loaded_images.append(image_info)
            loaded_count += 1

        # Sort by Z position if available
        self.sort_images_by_z()

        # Update UI
        self.update_file_table()
        self.update_visual_combo()
        self.update_ui_state()

        self.status_label.setText(f"Loaded {loaded_count} image(s)")

    def sort_images_by_z(self):
        """Sort images by Z position"""

        def get_z_value(img):
            z = img['z_position']
            if isinstance(z, (int, float)):
                return z
            return 0  # Put N/A at beginning

        self.loaded_images.sort(key=get_z_value)

    def update_file_table(self):
        """Update file list table"""
        self.file_table.setRowCount(len(self.loaded_images))

        for i, img in enumerate(self.loaded_images):
            # Checkbox
            check = QCheckBox()
            check.setChecked(img['enabled'])
            check.stateChanged.connect(lambda state, idx=i: self.on_image_toggled(idx, state))
            self.file_table.setCellWidget(i, 0, check)

            # Filename
            self.file_table.setItem(i, 1, QTableWidgetItem(img['filename']))

            # Z position
            z_str = f"{img['z_position']:.1f}" if isinstance(img['z_position'], (int, float)) else "N/A"
            self.file_table.setItem(i, 2, QTableWidgetItem(z_str))

            # R position
            r_str = f"{img['r_position']:.3f}" if isinstance(img['r_position'], (int, float)) else "N/A"
            self.file_table.setItem(i, 3, QTableWidgetItem(r_str))

            # File size
            size_kb = os.path.getsize(img['filepath']) / 1024
            self.file_table.setItem(i, 4, QTableWidgetItem(f"{size_kb:.1f} KB"))

            # Image size
            h, w = img['shape']
            self.file_table.setItem(i, 5, QTableWidgetItem(f"{w}×{h}"))

            # Status
            status = "Ready" if img['enabled'] else "Disabled"
            self.file_table.setItem(i, 6, QTableWidgetItem(status))

        self.file_count_label.setText(f"Loaded Images: {len(self.loaded_images)}")

    def on_image_toggled(self, idx, state):
        """Handle image enable/disable"""
        self.loaded_images[idx]['enabled'] = (state == Qt.Checked)
        self.update_file_table()

    def remove_selected(self):
        """Remove selected images"""
        selected = self.file_table.selectedIndexes()
        if not selected:
            return

        rows = sorted(set(index.row() for index in selected), reverse=True)
        for row in rows:
            del self.loaded_images[row]

        self.update_file_table()
        self.update_visual_combo()
        self.update_ui_state()

    def clear_all(self):
        """Clear all loaded images"""
        self.loaded_images.clear()
        self.drift_results.clear()
        self.update_file_table()
        self.update_visual_combo()
        self.clear_results()
        self.update_ui_state()
        self.status_label.setText("Ready - Load images to begin")

    def on_reference_changed(self):
        """Handle reference image selection change"""
        if not self.loaded_images:
            return

        ref_text = self.ref_combo.currentText()
        if "First" in ref_text:
            self.reference_index = 0
        elif "Middle" in ref_text:
            self.reference_index = len(self.loaded_images) // 2
        elif "Last" in ref_text:
            self.reference_index = len(self.loaded_images) - 1
        elif "Best" in ref_text:
            # Find image with highest mean intensity (proxy for quality)
            best_idx = 0
            best_mean = 0
            for i, img in enumerate(self.loaded_images):
                if img['enabled']:
                    mean_val = np.nanmean(img['data'])
                    if mean_val > best_mean:
                        best_mean = mean_val
                        best_idx = i
            self.reference_index = best_idx

    def update_visual_combo(self):
        """Update visual comparison combo box"""
        self.visual_combo.clear()
        for i, img in enumerate(self.loaded_images):
            self.visual_combo.addItem(f"{i}: {img['filename']}")

    def update_ui_state(self):
        """Update UI element enabled/disabled state"""
        has_images = len(self.loaded_images) > 0
        has_multiple = len([img for img in self.loaded_images if img['enabled']]) > 1
        has_results = len(self.drift_results) > 0

        self.calculate_btn.setEnabled(has_multiple)
        self.export_csv_btn.setEnabled(has_results)
        self.apply_compensation_btn.setEnabled(has_results)

    def calculate_drift(self):
        """Calculate drift for all images using selected method(s)"""
        enabled_images = [img for img in self.loaded_images if img['enabled']]

        if len(enabled_images) < 2:
            QMessageBox.warning(self, "Insufficient Images",
                                "Need at least 2 enabled images for drift calculation")
            return

        registration_mode = self.registration_mode_combo.currentText()

        # Get reference image for reference-based mode
        self.on_reference_changed()
        ref_image = self.loaded_images[self.reference_index]

        if not ref_image['enabled']:
            QMessageBox.warning(self, "Reference Disabled",
                                "Reference image is disabled. Please enable it or select a different reference.")
            return

        self.status_label.setText("Calculating drift...")
        QApplication.processEvents()

        # Get settings
        upsample = self.upsample_spin.value() if self.subpixel_cb.isChecked() else 1
        method = self.method_combo.currentText()

        # Calculate pixel size from reference image
        pixel_size_x, pixel_size_y = self.get_pixel_size(ref_image)

        # Clear previous results
        self.drift_results.clear()
        self.consecutive_results = []  # Store consecutive results

        from PyQt5.QtWidgets import QProgressDialog

        progress = QProgressDialog("Processing images...", "Cancel", 0, len(self.loaded_images), self)
        progress.setWindowTitle("Calculating Drift")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # ==== REFERENCE-BASED REGISTRATION ====
        if "Reference" in registration_mode or "Both" in registration_mode:
            for i, img in enumerate(self.loaded_images):
                if progress.wasCanceled():
                    self.status_label.setText("Drift calculation cancelled")
                    return

                if not img['enabled']:
                    continue

                progress.setValue(i)
                progress.setLabelText(f"Reference mode: {i + 1}/{len(self.loaded_images)}: {img['filename']}")
                QApplication.processEvents()

                if i == self.reference_index:
                    # Reference image has zero drift
                    result = {
                        'index': i,
                        'filename': img['filename'],
                        'z_position': img['z_position'],
                        'r_position': img['r_position'],
                        'drift_x_nm': 0.0,
                        'drift_y_nm': 0.0,
                        'drift_magnitude_nm': 0.0,
                        'correlation': 1.0,
                        'status': 'Reference'
                    }
                else:
                    # Calculate drift
                    if "Phase Correlation" in method and SKIMAGE_AVAILABLE:
                        drift_x_px, drift_y_px, corr = self.calculate_phase_correlation(
                            ref_image['data'], img['data'], upsample
                        )
                    else:
                        drift_x_px, drift_y_px, corr = self.calculate_cross_correlation(
                            ref_image['data'], img['data']
                        )

                    # Convert to nm
                    drift_x_nm = drift_x_px * pixel_size_x
                    drift_y_nm = drift_y_px * pixel_size_y
                    magnitude = np.sqrt(drift_x_nm ** 2 + drift_y_nm ** 2)

                    # Determine status
                    if corr > 0.95:
                        status = "Excellent"
                    elif corr > 0.90:
                        status = "Good"
                    elif corr > 0.80:
                        status = "Fair"
                    else:
                        status = "Poor"

                    result = {
                        'index': i,
                        'filename': img['filename'],
                        'z_position': img['z_position'],
                        'r_position': img['r_position'],
                        'drift_x_nm': drift_x_nm,
                        'drift_y_nm': drift_y_nm,
                        'drift_magnitude_nm': magnitude,
                        'correlation': corr,
                        'status': status
                    }

                self.drift_results.append(result)

        # ==== CONSECUTIVE PAIR REGISTRATION ====
        if "Consecutive" in registration_mode or "Both" in registration_mode:
            progress.setLabelText("Calculating consecutive pairs...")
            QApplication.processEvents()
            self.consecutive_results = self.calculate_consecutive_drift()

        progress.setValue(len(self.loaded_images))
        progress.close()

        # Update all displays
        self.update_drift_table()
        self.update_drift_plots()
        self.update_statistics()
        self.update_visual_comparison()
        self.update_ui_state()

        # Summary message
        if "Both" in registration_mode:
            msg = f"Drift calculation complete - Reference: {len(self.drift_results)} images, Consecutive: {len(self.consecutive_results)} pairs"
        elif "Reference" in registration_mode:
            msg = f"Drift calculation complete - {len(self.drift_results)} images (reference-based)"
        else:
            msg = f"Drift calculation complete - {len(self.consecutive_results)} pairs (consecutive)"

        self.status_label.setText(msg)

    def calculate_phase_correlation(self, ref_image, target_image, upsample_factor):
        """Calculate drift using phase correlation (FFT-based)"""
        try:
            shift, error, diffphase = phase_cross_correlation(
                ref_image,
                target_image,
                upsample_factor=upsample_factor
            )

            drift_y, drift_x = shift  # skimage returns (row, col) = (y, x)
            correlation = 1.0 - error

            return drift_x, drift_y, correlation

        except Exception as e:
            print(f"Phase correlation error: {e}")
            return 0.0, 0.0, 0.0

    def calculate_cross_correlation(self, ref_image, target_image):
        """Fallback: Simple cross-correlation (integer pixel accuracy)"""
        try:
            # Normalize images
            ref_norm = (ref_image - np.mean(ref_image)) / np.std(ref_image)
            target_norm = (target_image - np.mean(target_image)) / np.std(target_image)

            # Calculate cross-correlation using FFT
            corr = np.fft.ifft2(np.fft.fft2(ref_norm) * np.conj(np.fft.fft2(target_norm)))
            corr = np.fft.fftshift(np.abs(corr))

            # Find peak
            peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)

            # Calculate shift
            center_y, center_x = np.array(corr.shape) // 2
            drift_x = peak_x - center_x
            drift_y = peak_y - center_y

            # Correlation quality
            correlation = np.max(corr) / (np.std(ref_norm) * np.std(target_norm) * ref_norm.size)

            return float(drift_x), float(drift_y), float(correlation)

        except Exception as e:
            print(f"Cross-correlation error: {e}")
            return 0.0, 0.0, 0.0

    def calculate_consecutive_drift(self):
        """Calculate drift using consecutive pair registration"""
        enabled_images = [img for img in self.loaded_images if img['enabled'] and
                          isinstance(img.get('z_position'), (int, float))]

        if len(enabled_images) < 2:
            return []

        # Sort by Z position
        enabled_images.sort(key=lambda x: x['z_position'])

        # Get settings
        upsample = self.upsample_spin.value() if self.subpixel_cb.isChecked() else 1
        method = self.method_combo.currentText()

        # Calculate pixel size from first image
        pixel_size_x, pixel_size_y = self.get_pixel_size(enabled_images[0])

        consecutive_results = []

        # Register each consecutive pair
        for i in range(len(enabled_images) - 1):
            img1 = enabled_images[i]
            img2 = enabled_images[i + 1]

            # Calculate drift from img1 to img2
            if "Phase Correlation" in method and SKIMAGE_AVAILABLE:
                drift_x_px, drift_y_px, corr = self.calculate_phase_correlation(
                    img1['data'], img2['data'], upsample
                )
            else:
                drift_x_px, drift_y_px, corr = self.calculate_cross_correlation(
                    img1['data'], img2['data']
                )

            # Convert to nm
            delta_x = drift_x_px * pixel_size_x
            delta_y = drift_y_px * pixel_size_y

            # Calculate Z change
            delta_z = img2['z_position'] - img1['z_position']

            # Calculate ratios for this pair
            x_ratio = delta_x / delta_z if abs(delta_z) > 1e-6 else 0.0
            y_ratio = delta_y / delta_z if abs(delta_z) > 1e-6 else 0.0

            magnitude = np.sqrt(delta_x ** 2 + delta_y ** 2)

            # Find the actual indices in loaded_images
            img1_index = next((idx for idx, img in enumerate(self.loaded_images)
                               if img['filename'] == img1['filename']), i)
            img2_index = next((idx for idx, img in enumerate(self.loaded_images)
                               if img['filename'] == img2['filename']), i + 1)

            result = {
                'pair_index': i,
                'img1_index': img1_index,
                'img2_index': img2_index,
                'img1_filename': img1['filename'],
                'img2_filename': img2['filename'],
                'z1': img1['z_position'],
                'z2': img2['z_position'],
                'delta_z': delta_z,
                'delta_x_nm': delta_x,
                'delta_y_nm': delta_y,
                'magnitude_nm': magnitude,
                'x_ratio': x_ratio,
                'y_ratio': y_ratio,
                'correlation': corr
            }

            consecutive_results.append(result)

        return consecutive_results

    def get_pixel_size(self, image_info):
        """Get pixel size in nm from image metadata"""
        metadata = image_info['metadata']

        # Try to extract from scan parameters
        if 'x_start_nm' in metadata and 'x_end_nm' in metadata and 'x_pixels' in metadata:
            x_range = abs(float(metadata['x_end_nm']) - float(metadata['x_start_nm']))
            x_pixels = int(metadata['x_pixels'])
            pixel_size_x = x_range / (x_pixels - 1) if x_pixels > 1 else 1.0
        else:
            pixel_size_x = 1.0  # Default

        if 'y_start_nm' in metadata and 'y_end_nm' in metadata and 'y_pixels' in metadata:
            y_range = abs(float(metadata['y_end_nm']) - float(metadata['y_start_nm']))
            y_pixels = int(metadata['y_pixels'])
            pixel_size_y = y_range / (y_pixels - 1) if y_pixels > 1 else 1.0
        else:
            pixel_size_y = 1.0  # Default

        return pixel_size_x, pixel_size_y

    def update_drift_table(self):
        """Update drift results table"""
        self.drift_table.setRowCount(len(self.drift_results))

        for i, result in enumerate(self.drift_results):
            # Filename
            self.drift_table.setItem(i, 0, QTableWidgetItem(result['filename']))

            # Index
            self.drift_table.setItem(i, 1, QTableWidgetItem(str(result['index'])))

            # Z position
            z_str = f"{result['z_position']:.1f}" if isinstance(result['z_position'], (int, float)) else "N/A"
            self.drift_table.setItem(i, 2, QTableWidgetItem(z_str))

            # R position
            r_str = f"{result['r_position']:.3f}" if isinstance(result['r_position'], (int, float)) else "N/A"
            self.drift_table.setItem(i, 3, QTableWidgetItem(r_str))

            # Drift X
            self.drift_table.setItem(i, 4, QTableWidgetItem(f"{result['drift_x_nm']:.2f}"))

            # Drift Y
            self.drift_table.setItem(i, 5, QTableWidgetItem(f"{result['drift_y_nm']:.2f}"))

            # Magnitude
            self.drift_table.setItem(i, 6, QTableWidgetItem(f"{result['drift_magnitude_nm']:.2f}"))

            # Correlation
            corr_item = QTableWidgetItem(f"{result['correlation']:.4f}")

            # Color code by correlation quality
            if result['correlation'] > 0.95:
                corr_item.setBackground(QColor(200, 255, 200))  # Green
            elif result['correlation'] > 0.90:
                corr_item.setBackground(QColor(255, 255, 200))  # Yellow
            elif result['correlation'] > 0.80:
                corr_item.setBackground(QColor(255, 220, 200))  # Orange
            else:
                corr_item.setBackground(QColor(255, 200, 200))  # Red

            self.drift_table.setItem(i, 7, corr_item)

            # Status
            self.drift_table.setItem(i, 8, QTableWidgetItem(result['status']))

    def update_drift_plots(self):
        """Update drift visualization plots"""
        if not self.drift_results:
            return

        self.plot_fig.clear()

        # Determine X-axis (use Z if available, otherwise index)
        has_z = any(isinstance(r['z_position'], (int, float)) for r in self.drift_results)
        has_r = any(isinstance(r['r_position'], (int, float)) for r in self.drift_results)

        if has_z:
            x_values = [r['z_position'] for r in self.drift_results]
            x_label = "Z Position (nm)"
            fit_type = "Z"
        elif has_r:
            x_values = [r['r_position'] for r in self.drift_results]
            x_label = "R Position (°)"
            fit_type = "R"
        else:
            x_values = [r['index'] for r in self.drift_results]
            x_label = "Image Index"
            fit_type = "Index"

        drift_x = [r['drift_x_nm'] for r in self.drift_results]
        drift_y = [r['drift_y_nm'] for r in self.drift_results]

        # Plot 1: X Drift
        ax1 = self.plot_fig.add_subplot(2, 2, 1)
        ax1.plot(x_values, drift_x, 'o-', color='blue', markersize=6, linewidth=1.5, label='X Drift')
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('X Drift (nm)')
        ax1.set_title('X Drift vs Position')
        ax1.grid(True, alpha=0.3)

        # Linear fit for X
        if len(x_values) > 2 and has_z:
            try:
                p_x = Polynomial.fit(x_values, drift_x, deg=1)
                x_fit = np.linspace(min(x_values), max(x_values), 100)
                y_fit = p_x(x_fit)
                ax1.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.7,
                         label=f'Fit: ΔX = {p_x.coef[1]:.6f}×Z + {p_x.coef[0]:.2f}')
                ax1.legend()
            except:
                pass

        # Plot 2: Y Drift
        ax2 = self.plot_fig.add_subplot(2, 2, 2)
        ax2.plot(x_values, drift_y, 'o-', color='red', markersize=6, linewidth=1.5, label='Y Drift')
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('Y Drift (nm)')
        ax2.set_title('Y Drift vs Position')
        ax2.grid(True, alpha=0.3)

        # Linear fit for Y
        if len(x_values) > 2 and has_z:
            try:
                p_y = Polynomial.fit(x_values, drift_y, deg=1)
                x_fit = np.linspace(min(x_values), max(x_values), 100)
                y_fit = p_y(x_fit)
                ax2.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.7,
                         label=f'Fit: ΔY = {p_y.coef[1]:.6f}×Z + {p_y.coef[0]:.2f}')
                ax2.legend()
            except:
                pass

        # Plot 3: 2D Drift Trajectory
        ax3 = self.plot_fig.add_subplot(2, 2, 3)
        ax3.plot(drift_x, drift_y, 'o-', markersize=6, linewidth=1.5, color='purple')
        ax3.plot(0, 0, 'g*', markersize=15, label='Reference')
        ax3.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax3.axvline(0, color='gray', linestyle='--', alpha=0.3)
        ax3.set_xlabel('X Drift (nm)')
        ax3.set_ylabel('Y Drift (nm)')
        ax3.set_title('2D Drift Trajectory')
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        ax3.legend()

        # Add arrows showing direction
        for i in range(len(drift_x) - 1):
            ax3.annotate('', xy=(drift_x[i + 1], drift_y[i + 1]),
                         xytext=(drift_x[i], drift_y[i]),
                         arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

        # Plot 4: Drift Magnitude
        ax4 = self.plot_fig.add_subplot(2, 2, 4)
        magnitudes = [r['drift_magnitude_nm'] for r in self.drift_results]
        ax4.plot(x_values, magnitudes, 'o-', color='green', markersize=6, linewidth=1.5)
        ax4.set_xlabel(x_label)
        ax4.set_ylabel('Drift Magnitude (nm)')
        ax4.set_title('Total Drift Magnitude')
        ax4.grid(True, alpha=0.3)

        self.plot_fig.tight_layout()
        self.plot_canvas.draw()

    def update_statistics(self):
        """Update statistics tab with both registration methods"""
        registration_mode = self.registration_mode_combo.currentText()

        stats = "Drift Statistics\n" + "=" * 70 + "\n\n"

        # ==== REFERENCE-BASED STATISTICS ====
        if self.drift_results and ("Reference" in registration_mode or "Both" in registration_mode):
            stats += "METHOD 1: REFERENCE-BASED REGISTRATION\n" + "-" * 70 + "\n"

            drift_x = [r['drift_x_nm'] for r in self.drift_results if r['status'] != 'Reference']
            drift_y = [r['drift_y_nm'] for r in self.drift_results if r['status'] != 'Reference']
            magnitudes = [r['drift_magnitude_nm'] for r in self.drift_results if r['status'] != 'Reference']
            correlations = [r['correlation'] for r in self.drift_results]

            ref_img = self.drift_results[self.reference_index]

            stats += f"Total Images:              {len(self.drift_results)}\n"
            stats += f"Reference:                 {ref_img['filename']}\n"
            stats += f"                          (Z={ref_img['z_position']}, R={ref_img['r_position']})\n\n"

            if drift_x:
                stats += f"X Drift (relative to reference):\n"
                stats += f"  Mean:                   {np.mean(drift_x):8.2f} nm\n"
                stats += f"  Std Dev:                ±{np.std(drift_x):7.2f} nm\n"
                stats += f"  Range:                  {np.min(drift_x):8.2f} to {np.max(drift_x):8.2f} nm\n\n"

                stats += f"Y Drift (relative to reference):\n"
                stats += f"  Mean:                   {np.mean(drift_y):8.2f} nm\n"
                stats += f"  Std Dev:                ±{np.std(drift_y):7.2f} nm\n"
                stats += f"  Range:                  {np.min(drift_y):8.2f} to {np.max(drift_y):8.2f} nm\n\n"

                stats += f"Magnitude:\n"
                stats += f"  Mean:                   {np.mean(magnitudes):8.2f} nm\n"
                stats += f"  Max:                    {np.max(magnitudes):8.2f} nm\n\n"

                stats += f"Correlation Quality:\n"
                stats += f"  Mean:                   {np.mean(correlations):8.4f}\n"
                stats += f"  Range:                  {np.min(correlations):8.4f} to {np.max(correlations):8.4f}\n\n"

            # Linear fit for reference-based
            has_z = any(isinstance(r['z_position'], (int, float)) for r in self.drift_results)
            if has_z and len(drift_x) > 2:
                z_values = [r['z_position'] for r in self.drift_results
                            if isinstance(r['z_position'], (int, float))]
                try:
                    p_x = Polynomial.fit(z_values, [r['drift_x_nm'] for r in self.drift_results
                                                    if isinstance(r['z_position'], (int, float))], deg=1)
                    p_y = Polynomial.fit(z_values, [r['drift_y_nm'] for r in self.drift_results
                                                    if isinstance(r['z_position'], (int, float))], deg=1)

                    x_ratio_ref = p_x.coef[1]
                    y_ratio_ref = p_y.coef[1]

                    stats += f"Linear Fit (drift vs Z):\n"
                    stats += f"  ΔX = {p_x.coef[1]:.6f} × Z + {p_x.coef[0]:.2f}\n"
                    stats += f"  ΔY = {p_y.coef[1]:.6f} × Z + {p_y.coef[0]:.2f}\n\n"
                    stats += f"Compensation Ratios (from linear fit):\n"
                    stats += f"  X_ratio: {x_ratio_ref:.8f}\n"
                    stats += f"  Y_ratio: {y_ratio_ref:.8f}\n\n"
                except:
                    stats += "Linear fit failed\n\n"

        # ==== CONSECUTIVE PAIR STATISTICS ====
        if self.consecutive_results and ("Consecutive" in registration_mode or "Both" in registration_mode):
            if self.drift_results:
                stats += "\n" + "=" * 70 + "\n\n"

            stats += "METHOD 2: CONSECUTIVE PAIR REGISTRATION\n" + "-" * 70 + "\n"

            delta_x_list = [r['delta_x_nm'] for r in self.consecutive_results]
            delta_y_list = [r['delta_y_nm'] for r in self.consecutive_results]
            delta_z_list = [r['delta_z'] for r in self.consecutive_results]
            magnitude_list = [r['magnitude_nm'] for r in self.consecutive_results]
            x_ratio_list = [r['x_ratio'] for r in self.consecutive_results]
            y_ratio_list = [r['y_ratio'] for r in self.consecutive_results]
            corr_list = [r['correlation'] for r in self.consecutive_results]

            stats += f"Total Pairs:               {len(self.consecutive_results)}\n"
            stats += f"Images:                    {len(self.consecutive_results) + 1}\n\n"

            stats += f"Step-by-Step Changes:\n"
            stats += f"  ΔZ per step:\n"
            stats += f"    Mean:                 {np.mean(delta_z_list):8.2f} nm\n"
            stats += f"    Range:                {np.min(delta_z_list):8.2f} to {np.max(delta_z_list):8.2f} nm\n"
            stats += f"  ΔX per step:\n"
            stats += f"    Mean:                 {np.mean(delta_x_list):8.2f} nm\n"
            stats += f"    Std Dev:              ±{np.std(delta_x_list):7.2f} nm\n"
            stats += f"    Range:                {np.min(delta_x_list):8.2f} to {np.max(delta_x_list):8.2f} nm\n"
            stats += f"  ΔY per step:\n"
            stats += f"    Mean:                 {np.mean(delta_y_list):8.2f} nm\n"
            stats += f"    Std Dev:              ±{np.std(delta_y_list):7.2f} nm\n"
            stats += f"    Range:                {np.min(delta_y_list):8.2f} to {np.max(delta_y_list):8.2f} nm\n\n"

            stats += f"Total Accumulated Drift:\n"
            stats += f"  Total ΔZ:               {sum(delta_z_list):8.2f} nm\n"
            stats += f"  Total ΔX:               {sum(delta_x_list):8.2f} nm\n"
            stats += f"  Total ΔY:               {sum(delta_y_list):8.2f} nm\n"
            stats += f"  Total distance:         {sum(magnitude_list):8.2f} nm\n\n"

            stats += f"Compensation Ratios (average of pairs):\n"
            stats += f"  X_ratio:                {np.mean(x_ratio_list):.8f} ± {np.std(x_ratio_list):.8f}\n"
            stats += f"  Y_ratio:                {np.mean(y_ratio_list):.8f} ± {np.std(y_ratio_list):.8f}\n\n"

            stats += f"Correlation Quality:\n"
            stats += f"  Mean:                   {np.mean(corr_list):8.4f}\n"
            stats += f"  Range:                  {np.min(corr_list):8.4f} to {np.max(corr_list):8.4f}\n\n"

            # Detailed pair information
            stats += "Individual Pairs:\n"
            for i, r in enumerate(self.consecutive_results):
                stats += f"  Pair {i + 1}: Z={r['z1']:.1f}→{r['z2']:.1f} (ΔZ={r['delta_z']:.1f}) | "
                stats += f"ΔX={r['delta_x_nm']:.2f}, ΔY={r['delta_y_nm']:.2f} | "
                stats += f"X_ratio={r['x_ratio']:.6f}, Y_ratio={r['y_ratio']:.6f}\n"

        if not self.drift_results and not self.consecutive_results:
            stats = "No results yet - calculate drift first"

        self.stats_text.setPlainText(stats)

    def update_visual_comparison(self):
        """Update visual verification display"""
        if not self.drift_results or not self.loaded_images:
            return

        idx = self.visual_combo.currentIndex()
        if idx < 0 or idx >= len(self.loaded_images):
            return

        ref_img = self.loaded_images[self.reference_index]
        target_img = self.loaded_images[idx]

        mode = self.visual_mode_combo.currentText()
        opacity = self.opacity_slider.value() / 100.0

        self.visual_fig.clear()
        ax = self.visual_fig.add_subplot(111)

        ref_data = ref_img['data']
        target_data = target_img['data']

        # Normalize images to 0-1
        ref_norm = (ref_data - np.nanmin(ref_data)) / (np.nanmax(ref_data) - np.nanmin(ref_data))
        target_norm = (target_data - np.nanmin(target_data)) / (np.nanmax(target_data) - np.nanmin(target_data))

        if "Overlay" in mode:
            # Blend images
            blended = ref_norm * (1 - opacity) + target_norm * opacity
            ax.imshow(blended, cmap='gray', origin='lower')
            ax.set_title(f"Overlay: Reference + {target_img['filename']} (opacity={opacity:.2f})")

        elif "Difference" in mode:
            # Difference image
            diff = target_norm - ref_norm
            im = ax.imshow(diff, cmap='RdBu', origin='lower', vmin=-0.5, vmax=0.5)
            self.visual_fig.colorbar(im, ax=ax, label='Difference')
            ax.set_title(f"Difference: {target_img['filename']} - Reference")

        elif "Checkerboard" in mode:
            # Checkerboard pattern
            h, w = ref_data.shape
            checker = np.zeros((h, w))
            block_size = max(h // 20, w // 20, 10)

            for i in range(0, h, block_size):
                for j in range(0, w, block_size):
                    if ((i // block_size) + (j // block_size)) % 2 == 0:
                        checker[i:i + block_size, j:j + block_size] = 1

            combined = np.where(checker, ref_norm, target_norm)
            ax.imshow(combined, cmap='gray', origin='lower')
            ax.set_title(f"Checkerboard: Reference / {target_img['filename']}")

        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')

        # Add drift info if available
        if self.drift_results:
            result = next((r for r in self.drift_results if r['index'] == idx), None)
            if result:
                ax.text(0.02, 0.98,
                        f"Drift: ΔX={result['drift_x_nm']:.2f}nm, ΔY={result['drift_y_nm']:.2f}nm\n"
                        f"Correlation: {result['correlation']:.4f}",
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        self.visual_fig.tight_layout()
        self.visual_canvas.draw()

    def copy_table_to_clipboard(self):
        """Copy drift table to clipboard"""
        if not self.drift_results:
            return

        # Build CSV text
        text = "Image,Index,Z_nm,R_deg,Drift_X_nm,Drift_Y_nm,Drift_Magnitude_nm,Correlation,Status\n"

        for r in self.drift_results:
            z_str = f"{r['z_position']:.1f}" if isinstance(r['z_position'], (int, float)) else "N/A"
            r_str = f"{r['r_position']:.3f}" if isinstance(r['r_position'], (int, float)) else "N/A"

            text += f"{r['filename']},{r['index']},{z_str},{r_str},"
            text += f"{r['drift_x_nm']:.3f},{r['drift_y_nm']:.3f},{r['drift_magnitude_nm']:.3f},"
            text += f"{r['correlation']:.4f},{r['status']}\n"

        QApplication.clipboard().setText(text)
        self.status_label.setText("Table copied to clipboard")

    def export_drift_csv(self):
        """Export drift results to CSV"""
        if not self.drift_results:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Drift Data",
            f"drift_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv);;All Files (*)"
        )

        if not filepath:
            return

        try:
            with open(filepath, 'w') as f:
                # Write header comments
                f.write("# Image Registration Results\n")
                f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                ref_img = self.drift_results[self.reference_index]
                f.write(f"# Reference: {ref_img['filename']} (Z={ref_img['z_position']}, R={ref_img['r_position']})\n")
                f.write(f"# Method: {self.method_combo.currentText()}\n")
                f.write(f"# Upsample Factor: {self.upsample_spin.value()}\n")
                f.write("#\n")

                # Calculate and write compensation ratios if available
                has_z = any(isinstance(r['z_position'], (int, float)) for r in self.drift_results)
                if has_z and len(self.drift_results) > 2:
                    z_values = [r['z_position'] for r in self.drift_results
                                if isinstance(r['z_position'], (int, float))]
                    drift_x = [r['drift_x_nm'] for r in self.drift_results
                               if isinstance(r['z_position'], (int, float))]
                    drift_y = [r['drift_y_nm'] for r in self.drift_results
                               if isinstance(r['z_position'], (int, float))]

                    try:
                        p_x = Polynomial.fit(z_values, drift_x, deg=1)
                        p_y = Polynomial.fit(z_values, drift_y, deg=1)

                        f.write("# Fitted Compensation Ratios:\n")
                        f.write(f"# X_compensation_ratio: {p_x.coef[1]:.8f}\n")
                        f.write(f"# Y_compensation_ratio: {p_y.coef[1]:.8f}\n")
                        f.write("#\n")
                    except:
                        pass

                # Write column headers
                f.write("Image,Index,Z_nm,R_deg,Drift_X_nm,Drift_Y_nm,Drift_Magnitude_nm,Correlation,Status\n")

                # Write data
                for r in self.drift_results:
                    z_str = f"{r['z_position']:.1f}" if isinstance(r['z_position'], (int, float)) else "N/A"
                    r_str = f"{r['r_position']:.3f}" if isinstance(r['r_position'], (int, float)) else "N/A"

                    f.write(f"{r['filename']},{r['index']},{z_str},{r_str},")
                    f.write(f"{r['drift_x_nm']:.4f},{r['drift_y_nm']:.4f},{r['drift_magnitude_nm']:.4f},")
                    f.write(f"{r['correlation']:.6f},{r['status']}\n")

            self.status_label.setText(f"Exported to {os.path.basename(filepath)}")
            QMessageBox.information(self, "Export Complete", f"Drift data exported to:\n{filepath}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{str(e)}")

    def export_plot(self):
        """Export drift plots as PNG"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Plot",
            f"drift_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
        )

        if filepath:
            self.plot_fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self.status_label.setText(f"Plot exported to {os.path.basename(filepath)}")

    def apply_compensation_to_scan(self):
        """Calculate and emit compensation ratios using selected method"""
        registration_mode = self.registration_mode_combo.currentText()

        # Check what data we have
        has_reference = len(self.drift_results) > 0
        has_consecutive = len(self.consecutive_results) if hasattr(self, 'consecutive_results') else 0

        if not has_reference and not has_consecutive:
            QMessageBox.warning(self, "No Data",
                                "Calculate drift first before applying compensation")
            return

        # Check if we have Z data
        if has_reference:
            has_z_ref = any(isinstance(r['z_position'], (int, float)) for r in self.drift_results)
        else:
            has_z_ref = False

        if not has_z_ref and not has_consecutive:
            QMessageBox.warning(self, "No Z Data",
                                "Cannot calculate compensation ratios without Z position data")
            return

        try:
            results_text = "COMPENSATION RATIO CALCULATION\n" + "=" * 70 + "\n\n"

            x_ratio_ref = None
            y_ratio_ref = None
            x_ratio_cons = None
            y_ratio_cons = None
            x_ratio_cons_std = None
            y_ratio_cons_std = None

            # ==== REFERENCE-BASED CALCULATION ====
            if has_reference and has_z_ref and len(self.drift_results) >= 3:
                z_values = [r['z_position'] for r in self.drift_results
                            if isinstance(r['z_position'], (int, float))]
                drift_x = [r['drift_x_nm'] for r in self.drift_results
                           if isinstance(r['z_position'], (int, float))]
                drift_y = [r['drift_y_nm'] for r in self.drift_results
                           if isinstance(r['z_position'], (int, float))]

                p_x = Polynomial.fit(z_values, drift_x, deg=1)
                p_y = Polynomial.fit(z_values, drift_y, deg=1)

                x_ratio_ref = p_x.coef[1]
                y_ratio_ref = p_y.coef[1]

                z_range = max(z_values) - min(z_values)
                x_drift_range = max(drift_x) - min(drift_x)
                y_drift_range = max(drift_y) - min(drift_y)

                results_text += "METHOD 1: REFERENCE-BASED (Linear Fit)\n" + "-" * 70 + "\n"
                results_text += f"Images: {len(z_values)}\n"
                results_text += f"Z Range: {min(z_values):.1f} to {max(z_values):.1f} nm (ΔZ = {z_range:.1f} nm)\n"
                results_text += f"X Drift Range: {min(drift_x):.2f} to {max(drift_x):.2f} nm (Δ = {x_drift_range:.2f} nm)\n"
                results_text += f"Y Drift Range: {min(drift_y):.2f} to {max(drift_y):.2f} nm (Δ = {y_drift_range:.2f} nm)\n\n"
                results_text += f"Linear Model:\n"
                results_text += f"  drift_X = {x_ratio_ref:.6f} × Z + {p_x.coef[0]:.2f}\n"
                results_text += f"  drift_Y = {y_ratio_ref:.6f} × Z + {p_y.coef[0]:.2f}\n\n"
                results_text += f"Compensation Ratios:\n"
                results_text += f"  X: {x_ratio_ref:.8f}\n"
                results_text += f"  Y: {y_ratio_ref:.8f}\n\n"

            # ==== CONSECUTIVE PAIR CALCULATION ====
            if has_consecutive and len(self.consecutive_results) >= 1:
                x_ratio_list = [r['x_ratio'] for r in self.consecutive_results]
                y_ratio_list = [r['y_ratio'] for r in self.consecutive_results]

                x_ratio_cons = np.mean(x_ratio_list)
                y_ratio_cons = np.mean(y_ratio_list)
                x_ratio_cons_std = np.std(x_ratio_list) if len(x_ratio_list) > 1 else 0.0
                y_ratio_cons_std = np.std(y_ratio_list) if len(y_ratio_list) > 1 else 0.0

                total_z = sum([r['delta_z'] for r in self.consecutive_results])
                total_x = sum([r['delta_x_nm'] for r in self.consecutive_results])
                total_y = sum([r['delta_y_nm'] for r in self.consecutive_results])

                if x_ratio_ref is not None:
                    results_text += "\n" + "=" * 70 + "\n\n"

                results_text += "METHOD 2: CONSECUTIVE PAIRS (Average)\n" + "-" * 70 + "\n"
                results_text += f"Pairs: {len(self.consecutive_results)}\n"
                results_text += f"Total ΔZ: {total_z:.1f} nm\n"
                results_text += f"Total ΔX: {total_x:.2f} nm\n"
                results_text += f"Total ΔY: {total_y:.2f} nm\n\n"
                results_text += f"Individual Pairs:\n"
                for i, r in enumerate(self.consecutive_results):
                    results_text += f"  Pair {i + 1}: X_ratio={r['x_ratio']:.6f}, Y_ratio={r['y_ratio']:.6f}\n"
                results_text += f"\nCompensation Ratios (average):\n"
                results_text += f"  X: {x_ratio_cons:.8f} ± {x_ratio_cons_std:.8f}\n"
                results_text += f"  Y: {y_ratio_cons:.8f} ± {y_ratio_cons_std:.8f}\n\n"

            # ==== COMPARISON ====
            if x_ratio_ref is not None and x_ratio_cons is not None:
                results_text += "\n" + "=" * 70 + "\n\n"
                results_text += "COMPARISON\n" + "-" * 70 + "\n"
                results_text += f"{'Method':<25} {'X Ratio':<20} {'Y Ratio':<20}\n"
                results_text += f"{'-' * 25} {'-' * 20} {'-' * 20}\n"
                results_text += f"{'Reference-based':<25} {x_ratio_ref:>20.8f} {y_ratio_ref:>20.8f}\n"
                results_text += f"{'Consecutive pairs':<25} {x_ratio_cons:>20.8f} {y_ratio_cons:>20.8f}\n"
                results_text += f"{'-' * 25} {'-' * 20} {'-' * 20}\n"
                x_diff = abs(x_ratio_ref - x_ratio_cons)
                y_diff = abs(y_ratio_ref - y_ratio_cons)
                x_pct = (x_diff / abs(x_ratio_cons) * 100) if x_ratio_cons != 0 else 0
                y_pct = (y_diff / abs(y_ratio_cons) * 100) if y_ratio_cons != 0 else 0
                results_text += f"{'Difference':<25} {x_diff:>20.8f} {y_diff:>20.8f}\n"
                results_text += f"{'Percent Difference':<25} {x_pct:>19.2f}% {y_pct:>19.2f}%\n\n"

            # ==== SHOW DIALOG TO SELECT METHOD ====
            msg = QMessageBox(self)
            msg.setWindowTitle("Apply Compensation Ratios")
            msg.setIcon(QMessageBox.Information)
            msg.setDetailedText(results_text)

            # Determine which method(s) to offer
            if x_ratio_ref is not None and x_ratio_cons is not None:
                # Both methods available
                msg.setText("Both registration methods calculated. Which would you like to apply?")
                ref_btn = msg.addButton(f"Reference-based\n(X={x_ratio_ref:.6f}, Y={y_ratio_ref:.6f})",
                                        QMessageBox.YesRole)
                cons_btn = msg.addButton(f"Consecutive pairs\n(X={x_ratio_cons:.6f}, Y={y_ratio_cons:.6f})",
                                         QMessageBox.YesRole)
                avg_btn = msg.addButton(
                    f"Average of both\n(X={(x_ratio_ref + x_ratio_cons) / 2:.6f}, Y={(y_ratio_ref + y_ratio_cons) / 2:.6f})",
                    QMessageBox.YesRole)
                cancel_btn = msg.addButton("Cancel", QMessageBox.NoRole)

                msg.exec_()
                clicked = msg.clickedButton()

                if clicked == ref_btn:
                    x_final, y_final = x_ratio_ref, y_ratio_ref
                    method_name = "Reference-based"
                elif clicked == cons_btn:
                    x_final, y_final = x_ratio_cons, y_ratio_cons
                    method_name = "Consecutive pairs"
                elif clicked == avg_btn:
                    x_final = (x_ratio_ref + x_ratio_cons) / 2
                    y_final = (y_ratio_ref + y_ratio_cons) / 2
                    method_name = "Average"
                else:
                    return

            elif x_ratio_ref is not None:
                # Only reference-based available
                msg.setText(f"Apply reference-based compensation ratios?\n\nX: {x_ratio_ref:.8f}\nY: {y_ratio_ref:.8f}")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                if msg.exec_() == QMessageBox.Yes:
                    x_final, y_final = x_ratio_ref, y_ratio_ref
                    method_name = "Reference-based"
                else:
                    return

            elif x_ratio_cons is not None:
                # Only consecutive pairs available
                msg.setText(
                    f"Apply consecutive pair compensation ratios?\n\nX: {x_ratio_cons:.8f} ± {x_ratio_cons_std:.8f}\nY: {y_ratio_cons:.8f} ± {y_ratio_cons_std:.8f}")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                if msg.exec_() == QMessageBox.Yes:
                    x_final, y_final = x_ratio_cons, y_ratio_cons
                    method_name = "Consecutive pairs"
                else:
                    return
            else:
                QMessageBox.warning(self, "No Data", "No compensation ratios available")
                return

            # Emit the selected ratios
            self.compensation_ratios_calculated.emit(x_final, y_final)
            self.status_label.setText(f"Applied {method_name}: X={x_final:.6f}, Y={y_final:.6f}")

            # Update statistics display
            current_stats = self.stats_text.toPlainText()
            self.stats_text.setPlainText(
                f"APPLIED: {method_name} - X={x_final:.8f}, Y={y_final:.8f}\n"
                f"{'=' * 70}\n\n{current_stats}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Calculation Error",
                                 f"Failed to calculate compensation ratios:\n{str(e)}\n\n{traceback.format_exc()}")

    def clear_results(self):
        """Clear all results displays"""
        self.drift_table.setRowCount(0)
        self.plot_fig.clear()
        self.plot_canvas.draw()
        self.visual_fig.clear()
        self.visual_canvas.draw()
        self.stats_text.clear()

class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()

        # Core components
        # Remove QThread for MQTT - keep it simple, paho handles threading
        self.mqtt_controller = MQTTController()
        self.data_processor = DataProcessor()
        self.image_reconstructor = ImageReconstructor()
        self.line_reconstructor = LineReconstructor()
        self.scan_controller = ScanController()

        self.line_scan_controller = LineScanController()
        self.line_scan_display = None  # Will be created in UI setup
        self.line_scan_active = False  # Track if line scan is active

        self.data_storage = DataStorage()
        # Give the scan controller access to live positions
        self.scan_controller.set_data_processor(self.data_processor)
        self.scan_controller.image_reconstructor = self.image_reconstructor

        # Give line scan controller access to reconstructor
        self.line_scan_controller.line_reconstructor = self.line_reconstructor
        self.line_scan_controller.set_data_processor(self.data_processor)

        # Connect data processor to storage for timestamp parsing
        self.data_processor.set_data_storage(self.data_storage)
        self.data_processor.scan_controller = self.scan_controller
        self.data_processor.line_scan_controller = self.line_scan_controller

        # Add after existing controllers
        self.z_series_controller = ZSeriesController()
        self.z_series_controller.set_data_processor(self.data_processor)
        self.z_series_controller.set_data_storage(self.data_storage)

        # Add Z-scan controller for 1D Z-series
        self.z_scan_controller = ZScanController()
        self.z_scan_controller.set_data_processor(self.data_processor)
        self.z_scan_controller.line_reconstructor = self.line_reconstructor
        # Make the processor aware of the Z-scan controller (REQUIRED for gating)
        self.data_processor.z_scan_controller = self.z_scan_controller

        # Track Z-series state
        self.z_series_active = False
        self.z_series_scan_type = None
        self.z_scan_mode = False  # True when doing 1D Z-scan vs regular line scan

        # Add R-series controller
        self.r_series_controller = RSeriesController()
        self.r_series_controller.set_data_processor(self.data_processor)
        self.r_series_controller.set_data_storage(self.data_storage)

        # Add RZ-series controller for combined R+Z scans
        self.r_series_active = False
        self.r_series_scan_type = None

        # Current scan data
        self.current_scan_id = None
        self.current_hdf5_file = None
        self.data_batch = []
        self.batch_size = 1000

        # Timers
        self.statistics_timer = QTimer()
        self.statistics_timer.timeout.connect(self.data_processor.emit_statistics)
        self.statistics_timer.start(5000)  # Every 5 seconds

        self.image_update_timer = QTimer()
        self.image_update_timer.timeout.connect(self.image_reconstructor.emit_full_image)
        self.image_update_timer.start(1000)  # Every 1 second

        #Add line update timer
        self.line_update_timer = QTimer()
        self.line_update_timer.timeout.connect(self.line_reconstructor.emit_full_line)
        self.line_update_timer.start(100)  # Every 100ms for smoother line updates

        # Create HDF5 writer thread
        self.hdf5_writer = HDF5WriterThread(self.data_storage)
        self.hdf5_writer.error_occurred.connect(self.add_error_message)
        self.hdf5_writer.start()

        self.enable_interactive_view = True  # Could make this a checkbox in UI
        self._open_viewers = []  # Keep references to prevent GC
        self._image_viewers = []

        self.setup_ui()
        self.connect_signals()
        QTimer.singleShot(200, self.load_settings)


    def setup_ui(self):
        """Setup the main UI"""
        self.setWindowTitle("Scanning Microscope System v1.0")
        self.setGeometry(100, 100, 1800, 1200)

        # Create menu bar
        self.create_menu_bar()

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout with splitter
        main_layout = QHBoxLayout()
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - Tabbed display for image and line scans
        right_tabs = QTabWidget()

        # Tab 1: 2D Image display (existing)
        self.image_widget = ImageDisplayWidget()
        right_tabs.addTab(self.image_widget, "2D Image")

        # Tab 2: Line scan display (new)
        self.line_scan_display = LineScanDisplayWidget()
        right_tabs.addTab(self.line_scan_display, "1D Plot")

        # Tab 3: R+Z series display
        self.rz_series_display = RZSeriesDisplayWidget()
        right_tabs.addTab(self.rz_series_display, "R+Z Heatmap")

        splitter.addWidget(right_tabs)

        # Set splitter sizes (30% control, 70% image)
        splitter.setSizes([540, 1260])

        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)

        # Status bar
        self.status_bar = self.statusBar()

        # Connection status
        self.connection_label = QLabel("Disconnected")
        self.connection_label.setStyleSheet("QLabel { color: red; padding: 3px; }")
        self.status_bar.addPermanentWidget(self.connection_label)

        # Scan status
        self.scan_status_label = QLabel("Idle")
        self.scan_status_label.setStyleSheet("QLabel { padding: 3px; }")
        self.status_bar.addPermanentWidget(self.scan_status_label)

        # Create docked windows
        self.create_docked_windows()

    def create_menu_bar(self):
        """Create the main menu bar"""
        menubar = self.menuBar()

        # Windows menu
        windows_menu = menubar.addMenu('Windows')

        # Image Viewer action
        self.show_viewer_action = windows_menu.addAction('Image Viewer')
        self.show_viewer_action.triggered.connect(self.open_image_viewer)

        windows_menu.addSeparator()

        # System Status action
        self.show_status_action = windows_menu.addAction('System Status')
        self.show_status_action.setCheckable(True)
        self.show_status_action.setChecked(False)  # Start unchecked
        self.show_status_action.triggered.connect(self.toggle_status_dock)

        # System Log action
        self.show_log_action = windows_menu.addAction('System Log')
        self.show_log_action.setCheckable(True)
        self.show_log_action.setChecked(False)  # Start unchecked
        self.show_log_action.triggered.connect(self.toggle_log_dock)

        windows_menu.addSeparator()

        # Reset Windows Layout action
        reset_layout_action = windows_menu.addAction('Reset Window Layout')
        reset_layout_action.triggered.connect(self.reset_window_layout)

        # Tools menu (NEW)
        tools_menu = menubar.addMenu('Tools')

        self.show_registration_action = tools_menu.addAction('Image Registration...')
        self.show_registration_action.triggered.connect(self.open_image_registration)

        settings_menu = menubar.addMenu("Settings")
        act_save = QAction("Save Settings Now", self)
        act_save.triggered.connect(self.save_settings)
        settings_menu.addAction(act_save)

        act_load = QAction("Load Settings", self)
        act_load.triggered.connect(self.load_settings)
        settings_menu.addAction(act_load)

        act_loc = QAction("Show Settings File Location", self)
        act_loc.triggered.connect(self.show_settings_location)
        settings_menu.addAction(act_loc)

    def create_left_panel(self) -> QWidget:
        """Create the left control panel with tabbed scan/manual controls"""
        panel = QWidget()
        layout = QVBoxLayout()

        # Compact Connection & Storage group - HORIZONTAL LAYOUT
        conn_group = QGroupBox("Connection & Storage Settings")
        conn_layout = QGridLayout()
        conn_layout.setSpacing(5)  # Reduce spacing for compactness

        # LEFT SIDE - MQTT and Storage (Column 0-1)
        # Row 0: MQTT Settings
        conn_layout.addWidget(QLabel("MQTT:"), 0, 0)
        mqtt_layout = QHBoxLayout()
        mqtt_layout.setSpacing(5)
        self.broker_edit = QLineEdit("localhost")
        self.broker_edit.setPlaceholderText("Broker address")
        mqtt_layout.addWidget(self.broker_edit)
        mqtt_layout.addWidget(QLabel("Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(1883)
        self.port_spin.setMaximumWidth(80)
        mqtt_layout.addWidget(self.port_spin)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setMaximumWidth(100)
        self.connect_btn.clicked.connect(self.toggle_mqtt_connection)
        mqtt_layout.addWidget(self.connect_btn)
        conn_layout.addLayout(mqtt_layout, 0, 1)

        # Row 1: Storage Settings
        conn_layout.addWidget(QLabel("Path:"), 1, 0)
        storage_layout = QHBoxLayout()
        storage_layout.setSpacing(5)
        self.save_path_edit = QLineEdit("scan_data")
        self.save_path_edit.setPlaceholderText("Save directory for all data files")
        storage_layout.addWidget(self.save_path_edit)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setMaximumWidth(100)
        self.browse_btn.clicked.connect(self.browse_save_path)
        storage_layout.addWidget(self.browse_btn)
        conn_layout.addLayout(storage_layout, 1, 1)

        # Row 2: Auto-export option
        self.auto_export_cb = QCheckBox("Auto-export PNG/CSV after scan")
        self.auto_export_cb.setChecked(True)  # Default to enabled
        self.auto_export_cb.setToolTip("Automatically export image/data when scan completes")
        conn_layout.addWidget(self.auto_export_cb, 2, 0, 1, 2)

        # VERTICAL SEPARATOR (Column 2)
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("QFrame { color: #cccccc; }")
        conn_layout.addWidget(separator, 0, 2, 2, 1)

        # RIGHT SIDE - Detector Settings (Column 3)
        # Create a container for detector settings
        detector_container = QWidget()
        detector_layout = QVBoxLayout()
        detector_layout.setContentsMargins(10, 0, 0, 0)  # Add left margin after separator

        # Detector lag row
        detector_row = QHBoxLayout()
        detector_row.addWidget(QLabel("Detector Lag:"))
        self.detector_lag_spin = QDoubleSpinBox()
        self.detector_lag_spin.setRange(0.0, 5.0)
        self.detector_lag_spin.setValue(0.350)
        self.detector_lag_spin.setSingleStep(0.010)
        self.detector_lag_spin.setDecimals(3)
        self.detector_lag_spin.setSuffix(" s")
        self.detector_lag_spin.setMaximumWidth(80)
        self.detector_lag_spin.setToolTip("Detector response time before data acquisition")
        detector_row.addWidget(self.detector_lag_spin)
        detector_layout.addLayout(detector_row)

        # Info label
        info_label = QLabel("Wait time after settling\nbefore acquiring data")
        info_label.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        detector_layout.addWidget(info_label)

        detector_container.setLayout(detector_layout)
        conn_layout.addWidget(detector_container, 0, 3, 2, 1)

        # Set column stretches - give more space to left side
        conn_layout.setColumnStretch(1, 3)  # MQTT/Storage column
        conn_layout.setColumnStretch(3, 1)  # Detector settings column

        conn_group.setLayout(conn_layout)
        conn_group.setMaximumHeight(100)  # Keep compact height
        layout.addWidget(conn_group)

        # CREATE TAB WIDGET FOR SCAN/MANUAL CONTROL (rest remains the same)
        self.control_tabs = QTabWidget()

        # Tab 1: Manual Control
        self.manual_control = ManualControlWidget()
        self.control_tabs.addTab(self.manual_control, "Manual Control")

        # Tab 2: Scan Control
        self.scan_control = ScanControlWidget()
        self.control_tabs.addTab(self.scan_control, "2D Control")

        # Tab 3: Line Scan Control
        self.line_scan_control = LineScanControlWidget()
        self.control_tabs.addTab(self.line_scan_control, "1D Control")

        layout.addWidget(self.control_tabs)

        # Current readings group (rest remains the same)
        readings_group = QGroupBox("Current Readings")
        readings_layout = QGridLayout()

        readings_layout.addWidget(QLabel("Current:"), 0, 0)
        self.current_label = QLabel("N/A")
        self.current_label.setFont(QFont("Monospace", 10))
        self.current_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 3px; }")
        readings_layout.addWidget(self.current_label, 0, 1)

        readings_layout.addWidget(QLabel("X Position:"), 1, 0)
        self.x_pos_label = QLabel("N/A")
        self.x_pos_label.setFont(QFont("Monospace", 10))
        self.x_pos_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 3px; }")
        readings_layout.addWidget(self.x_pos_label, 1, 1)

        readings_layout.addWidget(QLabel("Y Position:"), 2, 0)
        self.y_pos_label = QLabel("N/A")
        self.y_pos_label.setFont(QFont("Monospace", 10))
        self.y_pos_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 3px; }")
        readings_layout.addWidget(self.y_pos_label, 2, 1)

        readings_layout.addWidget(QLabel("Z Position:"), 3, 0)
        self.z_pos_label = QLabel("N/A")
        self.z_pos_label.setFont(QFont("Monospace", 10))
        self.z_pos_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 3px; }")
        readings_layout.addWidget(self.z_pos_label, 3, 1)

        readings_layout.addWidget(QLabel("R Position:"), 4, 0)
        self.r_pos_label = QLabel("N/A")
        self.r_pos_label.setFont(QFont("Monospace", 10))
        self.r_pos_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 3px; }")
        readings_layout.addWidget(self.r_pos_label, 4, 1)

        readings_group.setLayout(readings_layout)
        layout.addWidget(readings_group)

        layout.addStretch()
        panel.setLayout(layout)

        return panel

    def browse_save_path(self):
        """Browse for save directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Save Directory",
            self.save_path_edit.text()
        )
        if directory:
            self.save_path_edit.setText(directory)

    def create_docked_windows(self):
        """Create dockable status and log windows (initially hidden)"""
        # Status dock
        self.status_dock = QDockWidget("System Status", self)
        self.status_dock.setAllowedAreas(
            Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea | Qt.TopDockWidgetArea)
        self.status_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)

        status_widget = QWidget()
        status_layout = QVBoxLayout()

        # Statistics display
        self.stats_table = QTableWidget(0, 2)
        self.stats_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.setMaximumHeight(200)

        status_layout.addWidget(QLabel("System Statistics:"))
        status_layout.addWidget(self.stats_table)

        status_widget.setLayout(status_layout)
        self.status_dock.setWidget(status_widget)

        # Connect close event to update menu
        self.status_dock.visibilityChanged.connect(self.on_status_dock_visibility_changed)

        # Add to main window but keep hidden initially
        self.addDockWidget(Qt.BottomDockWidgetArea, self.status_dock)
        self.status_dock.hide()

        # Log dock
        self.log_dock = QDockWidget("System Log", self)
        self.log_dock.setAllowedAreas(
            Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea | Qt.TopDockWidgetArea)
        self.log_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)

        self.log_dock.setWidget(self.log_text)

        # Connect close event to update menu
        self.log_dock.visibilityChanged.connect(self.on_log_dock_visibility_changed)

        # Add to main window but keep hidden initially
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()

    def toggle_status_dock(self, checked):
        """Toggle System Status dock visibility"""
        if checked:
            # Make it a standalone window when opened from menu
            if self.status_dock.isFloating():
                self.status_dock.show()
            else:
                self.status_dock.setFloating(True)
                self.status_dock.show()
                # Position it nicely
                self.status_dock.move(self.x() + 50, self.y() + 50)
                self.status_dock.resize(400, 300)
        else:
            self.status_dock.hide()

    def toggle_log_dock(self, checked):
        """Toggle System Log dock visibility"""
        if checked:
            # Make it a standalone window when opened from menu
            if self.log_dock.isFloating():
                self.log_dock.show()
            else:
                self.log_dock.setFloating(True)
                self.log_dock.show()
                # Position it nicely
                self.log_dock.move(self.x() + 100, self.y() + 100)
                self.log_dock.resize(600, 250)
        else:
            self.log_dock.hide()

    def on_status_dock_visibility_changed(self, visible):
        """Update menu when status dock visibility changes"""
        self.show_status_action.setChecked(visible)

    def on_log_dock_visibility_changed(self, visible):
        """Update menu when log dock visibility changes"""
        self.show_log_action.setChecked(visible)

    def reset_window_layout(self):
        """Reset all dock widgets to default positions"""
        # Hide both docks
        self.status_dock.hide()
        self.log_dock.hide()

        # Reset to docked state
        self.status_dock.setFloating(False)
        self.log_dock.setFloating(False)

        # Reset menu checkboxes
        self.show_status_action.setChecked(False)
        self.show_log_action.setChecked(False)

        # Re-add to default positions
        self.addDockWidget(Qt.BottomDockWidgetArea, self.status_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)

    def open_image_viewer(self, filepath=None):
        """Open a new Image Viewer window"""
        # Create viewer WITHOUT parent to make it truly independent
        viewer = UnifiedImageViewer(filepath=filepath, parent=None)

        # NEW: Connect ROI signal to scan control
        viewer.roi_selected.connect(self.on_roi_selected_from_viewer)

        viewer.show()

        # Track viewer to prevent garbage collection
        if not hasattr(self, '_image_viewers'):
            self._image_viewers = []

        self._image_viewers.append(viewer)
        viewer.destroyed.connect(lambda: self._image_viewers.remove(viewer) if viewer in self._image_viewers else None)

        return viewer

    def on_roi_selected_from_viewer(self, x_start, x_end, y_start, y_end):
        """Handle ROI selection from Image Viewer"""
        # Switch to 2D Control tab
        self.control_tabs.setCurrentWidget(self.scan_control)

        # Update scan parameters
        self.scan_control.x_start_spin.setValue(x_start)
        self.scan_control.x_end_spin.setValue(x_end)
        self.scan_control.y_start_spin.setValue(y_start)
        self.scan_control.y_end_spin.setValue(y_end)

        # Log the update
        self.add_log_message(
            f"Updated scan region from ROI: "
            f"X=[{x_start:.1f}, {x_end:.1f}], Y=[{y_start:.1f}, {y_end:.1f}]"
        )

        # Show confirmation message
        self.status_bar.showMessage(
            f"Scan region updated from Image Viewer ROI", 3000
        )

        # Flash the scan control to draw attention
        original_style = self.scan_control.styleSheet()
        self.scan_control.setStyleSheet("QWidget { background-color: #e8f5e9; }")
        QTimer.singleShot(500, lambda: self.scan_control.setStyleSheet(original_style))

    def open_image_registration(self, preload_files=None):
        """Open Image Registration window"""
        reg_window = ImageRegistrationWindow(parent=None)

        # Connect signal to receive compensation ratios
        reg_window.compensation_ratios_calculated.connect(self.apply_compensation_from_registration)

        if preload_files:
            reg_window.load_files(preload_files)

        reg_window.show()

        # Track to prevent GC
        if not hasattr(self, '_registration_windows'):
            self._registration_windows = []
        self._registration_windows.append(reg_window)
        reg_window.destroyed.connect(
            lambda: self._registration_windows.remove(reg_window)
            if reg_window in self._registration_windows else None
        )

        return reg_window

    def apply_compensation_from_registration(self, x_ratio, y_ratio):
        """Apply compensation ratios from registration analysis to scan settings"""
        # Update Z-series compensation in 2D scan control
        if hasattr(self, 'scan_control'):
            self.scan_control.x_compensation_spin.setValue(x_ratio)
            self.scan_control.y_compensation_spin.setValue(y_ratio)

            # Show confirmation
            self.add_log_message(
                f"Applied drift compensation from registration: "
                f"X={x_ratio:.6f}, Y={y_ratio:.6f}"
            )

            # Flash the Z-series panel to draw attention
            if hasattr(self.scan_control, 'z_series_enable_cb'):
                original_style = self.scan_control.styleSheet()
                self.scan_control.setStyleSheet("QWidget { background-color: #e8f5e9; }")
                QTimer.singleShot(1000, lambda: self.scan_control.setStyleSheet(original_style))

    def auto_open_completed_scan(self, filepath: str, scan_type: str):
        """Automatically open completed scan in Image Viewer"""
        try:
            viewer = self.open_image_viewer(filepath)
            self.add_log_message(f"Opened {scan_type} scan in Image Viewer: {os.path.basename(filepath)}")
        except Exception as e:
            self.add_log_message(f"Failed to auto-open viewer: {e}")

    def connect_signals(self):
        """Connect all signals and slots - UNIFIED PIPELINE VERSION"""
        # MQTT signals
        self.mqtt_controller.connected.connect(self.on_mqtt_connected)
        self.mqtt_controller.position_data_received.connect(self.data_processor.process_position_data)
        self.mqtt_controller.current_data_received.connect(self.data_processor.process_current_data)
        self.mqtt_controller.status_update.connect(self.add_log_message)
        self.mqtt_controller.error_occurred.connect(self.add_error_message)

        # Data processor signals - UNIFIED PIPELINE
        self.data_processor.statistics_update.connect(self.update_statistics)
        self.data_processor.live_readings.connect(self.update_live_readings)

        # Connect to BOTH reconstructors for unified pipeline
        self.data_processor.new_data_point.connect(self.image_reconstructor.add_data_point)
        self.data_processor.new_data_point.connect(self.line_reconstructor.add_data_point)
        self.data_processor.new_data_point.connect(self.on_new_data_point)  # For HDF5 storage

        # Image reconstructor signals
        self.image_reconstructor.image_updated.connect(self.image_widget.update_image)

        # Line reconstructor signals
        self.line_reconstructor.line_updated.connect(self.line_scan_display.update_line)
        self.line_reconstructor.point_added.connect(self.line_scan_display.update_single_point)
        # NEW: Connect line reconstructor's averaged signal to R+Z display
        self.line_reconstructor.point_added.connect(self._on_averaged_point_for_rz)
        self.line_reconstructor.scan_completed.connect(self.line_scan_display.on_scan_completed)

        # Scan control signals (unchanged)
        self.scan_control.scan_parameters_changed.connect(self.on_scan_parameters_changed)
        self.scan_control.start_scan_requested.connect(self.start_scan)
        self.scan_control.pause_scan_requested.connect(self.scan_controller.pause_scan)
        self.scan_control.stop_scan_requested.connect(self.stop_scan)

        # Manual control signals (unchanged)
        self.manual_control.move_requested.connect(self.on_manual_move_requested)
        self.manual_control.stop_axis_requested.connect(self.on_manual_stop_axis)
        self.manual_control.stop_all_requested.connect(self.on_manual_stop_all)

        # Connect scan path preview safely
        try:
            self.scan_control.scan_path_preview.connect(self.image_widget.show_scan_preview)
        except AttributeError as e:
            print(f"Warning: Could not connect scan preview: {e}")

        # Scan controller signals (unchanged)
        self.scan_controller.scan_started.connect(self.on_scan_started)
        self.scan_controller.scan_completed.connect(self.on_scan_completed)
        self.scan_controller.scan_progress.connect(self.scan_control.update_progress)
        self.scan_controller.movement_command.connect(self.mqtt_controller.send_command)
        self.scan_controller.status_update.connect(self.add_log_message)
        self.scan_controller.initialization_error.connect(self.show_initialization_error)

        # Line scan control signals
        self.line_scan_control.parameters_changed.connect(self.on_line_scan_parameters_changed)
        self.line_scan_control.start_scan_requested.connect(self.start_line_scan)
        self.line_scan_control.stop_scan_requested.connect(self.stop_line_scan)

        # Line scan controller signals
        self.line_scan_controller.scan_started.connect(self.on_line_scan_started)
        self.line_scan_controller.scan_completed.connect(self.on_line_scan_completed)
        self.line_scan_controller.scan_progress.connect(self.line_scan_control.update_progress)
        self.line_scan_controller.movement_command.connect(self.mqtt_controller.send_command)
        self.line_scan_controller.status_update.connect(self.add_log_message)

        # Connect line scan controller to data processor
        self.line_scan_controller.set_data_processor(self.data_processor)

        # Trigger initial preview after all connections are made
        if hasattr(self.scan_control, 'emit_scan_parameters'):
            self.scan_control.emit_scan_parameters()

        self.z_series_controller.z_series_started.connect(self.on_z_series_started)
        self.z_series_controller.z_series_completed.connect(self.on_z_series_completed)
        self.z_series_controller.z_slice_started.connect(self.on_z_slice_started)
        self.z_series_controller.z_slice_completed.connect(self.on_z_slice_completed)
        self.z_series_controller.z_series_progress.connect(self.on_z_series_progress)
        self.z_series_controller.movement_command.connect(self.mqtt_controller.send_command)
        self.z_series_controller.status_update.connect(self.add_log_message)

        # Z-scan controller signals (for 1D Z-series)
        self.z_scan_controller.scan_started.connect(self.on_z_scan_started)
        self.z_scan_controller.scan_completed.connect(self.on_z_scan_completed)
        self.z_scan_controller.scan_progress.connect(self.line_scan_control.update_progress)
        self.z_scan_controller.movement_command.connect(self.mqtt_controller.send_command)
        self.z_scan_controller.status_update.connect(self.add_log_message)
        self.z_scan_controller.current_position_index.connect(self.on_z_scan_position_update)

        # Add R-series controller signals
        self.r_series_controller.r_series_started.connect(self.on_r_series_started)
        self.r_series_controller.r_series_completed.connect(self.on_r_series_completed)
        self.r_series_controller.r_slice_started.connect(self.on_r_slice_started)
        self.r_series_controller.r_slice_completed.connect(self.on_r_slice_completed)
        self.r_series_controller.r_series_progress.connect(self.on_r_series_progress)
        self.r_series_controller.movement_command.connect(self.mqtt_controller.send_command)
        self.r_series_controller.status_update.connect(self.add_log_message)

    @pyqtSlot(dict)
    def update_live_readings(self, d: Dict):
        """Update the sidebar labels with the freshest MQTT readings (always-on)."""
        def fmt(val, unit=""):
            if val is None:
                return "N/A"
            # Show 3–6 sig figs; switch to scientific for very small numbers
            text = f"{val:.3g}" if abs(val) >= 1e-3 else f"{val:.3e}"
            return f"{text} {unit}".strip()

        # Current (no unit specified—adjust if you know it, e.g., "nA")
        self.current_label.setText(fmt(d.get("current"), "nA"))

        # Positions (nm)
        self.x_pos_label.setText(fmt(d.get("X"), "nm"))
        self.y_pos_label.setText(fmt(d.get("Y"), "nm"))
        self.z_pos_label.setText(fmt(d.get("Z"), "nm"))
        # R unit depends on your stage (deg? nm? rad?). Put the right one:
        self.r_pos_label.setText(fmt(d.get("R"), "μdeg"))

    @pyqtSlot(str)
    def show_initialization_error(self, error_message: str):
        """Show initialization error message box"""
        QMessageBox.critical(self, "Scan Initialization Error", error_message)
        self.add_error_message(error_message)

    @pyqtSlot()
    def toggle_mqtt_connection(self):
        """Toggle MQTT connection"""
        if not self.mqtt_controller.connected_status:
            # Connect
            host = self.broker_edit.text()
            port = self.port_spin.value()

            self.mqtt_controller.setup_mqtt(host, port)
            #  Direct call since we're not using separate thread
            self.mqtt_controller.connect_mqtt()

            self.connect_btn.setText("Connecting...")
            self.connect_btn.setEnabled(False)
        else:
            # Disconnect
            self.mqtt_controller.disconnect_mqtt()

    @pyqtSlot(bool)
    def on_mqtt_connected(self, connected: bool):
        """Handle MQTT connection status"""
        if connected:
            self.connect_btn.setText("Disconnect")
            self.connect_btn.setEnabled(True)
            self.connection_label.setText("Connected")
            self.connection_label.setStyleSheet("QLabel { color: green; padding: 3px; }")
        else:
            self.connect_btn.setText("Connect")
            self.connect_btn.setEnabled(True)
            self.connection_label.setText("Disconnected")
            self.connection_label.setStyleSheet("QLabel { color: red; padding: 3px; }")

    @pyqtSlot(object)
    def on_line_scan_parameters_changed(self, params: LineScanParameters):
        """Handle line scan parameter changes"""
        if self.line_scan_controller.scan_state != ScanState.IDLE:
            QMessageBox.warning(self, "Scan Active",
                                "Cannot change parameters during scan")
            return

        self.line_scan_controller.set_parameters(params)

    @pyqtSlot()
    def start_line_scan(self):
        """Start a line scan, Z-scan, or R+Z series depending on parameters"""

        # Force UI → params sync
        self.line_scan_control.update_parameters()
        params = self.line_scan_controller.line_params

        if not self.mqtt_controller.connected_status:
            QMessageBox.warning(self, "Warning", "MQTT not connected")
            return

        # Check no other scan is running
        if self.scan_controller.scan_state != ScanState.IDLE:
            QMessageBox.warning(self, "Warning", "2D scan is active")
            return

        try:
            if not params:
                QMessageBox.warning(self, "Warning", "No line scan parameters set")
                return

            # Set detector lag from UI for all possible controllers
            detector_lag = self.detector_lag_spin.value()
            self.line_scan_controller.set_detector_lag(detector_lag)
            self.z_scan_controller.set_detector_lag(detector_lag)

            # Check for R+Z series mode
            if hasattr(params, 'r_series') and params.r_series and params.r_series.enabled:
                # Start R+Z series
                self.start_rz_series(params)
                return

            # Check if this is a Z-scan (existing code)
            if hasattr(params, 'z_series') and params.z_series.enabled:
                # Start Z-scan (existing code)
                self.z_scan_mode = True
                self.line_scan_active = True

                # Initialize line reconstructor for Z vs signal plot
                z_line_params = LineScanParameters(
                    scan_axis="Z",
                    scan_start=params.z_series.z_start,
                    scan_end=params.z_series.z_end,
                    num_points=params.z_series.z_numbers,
                    step_size=params.z_series.z_step,
                    dwell_time=params.z_series.z_dwell_time
                )

                self.line_reconstructor.initialize_scan(z_line_params)
                if self.line_scan_display:
                    self.line_scan_display.set_scan_parameters(z_line_params)
                    self.line_scan_display.live_plot.setData([], [])

                # Restart line update timer
                if not self.line_update_timer.isActive():
                    self.line_update_timer.start(100)

                # Create HDF5 file for Z-scan
                save_path = self.save_path_edit.text().strip() or "scan_data"

                self.data_storage.current_detector_lag = self.detector_lag_spin.value()

                # Get current R position before creating file
                x, y, z, r, tx, ty, tz, tr, now = self.data_processor.get_position_snapshot()
                current_r = r if r is not None else 0.0

                self.current_hdf5_file = self.data_storage.create_hdf5_file(
                    "Z",  # Scan type
                    save_path,
                    scan_params=None,
                    line_params=params,
                    actual_r_position=current_r
                )

                if self.current_hdf5_file:
                    self.add_log_message(f"Created Z-scan file: {os.path.basename(self.current_hdf5_file)}")

                # Clear data batch
                self.data_batch.clear()

                # Set Z-scan parameters and start
                self.z_scan_controller.set_parameters(params.z_series, params.base_z_position)
                self.z_scan_controller.start_scan()

            else:
                # Regular line scan (existing code continues...)
                self.z_scan_mode = False

                # Get current positions
                x, y, z, r, tx, ty, tz, tr, now = self.data_processor.get_position_snapshot()
                current_z = z if z is not None else 0.0
                current_r = r if r is not None else 0.0

                # Initialize line reconstructor
                self.line_reconstructor.initialize_scan(params)
                self.line_scan_display.set_scan_parameters(params)

                # Clear the display
                self.line_scan_display.live_plot.setData([], [])

                # Restart the line update timer
                if not self.line_update_timer.isActive():
                    self.line_update_timer.start(100)

                # Create HDF5 file for line scan
                save_path = self.save_path_edit.text().strip() or "scan_data"

                self.data_storage.current_detector_lag = self.detector_lag_spin.value()

                self.current_hdf5_file = self.data_storage.create_hdf5_file(
                    "LINE",  # Scan type
                    save_path,
                    scan_params=None,
                    line_params=params,
                    actual_z_position=current_z,
                    actual_r_position=current_r
                )

                if self.current_hdf5_file:
                    self.add_log_message(f"Created data file: {os.path.basename(self.current_hdf5_file)}")

                # Clear data batch
                self.data_batch.clear()

                # Set flag for line scan mode
                self.line_scan_active = True

                # Start regular line scan
                self.line_scan_controller.start_scan()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start scan: {e}")
            self.add_error_message(f"Scan start error: {e}")

    def start_rz_series(self, params):
        """Start R+Z series scan"""
        self.r_series_active = True
        self.r_series_scan_type = "RZ"
        self.line_scan_active = True

        # Initialize R+Z display properly
        if hasattr(self, 'rz_series_display'):
            self.rz_series_display.initialize_rz_scan(params.r_series, params.z_series)

        # Initialize line reconstructor for Z-scan data at each R position
        z_line_params = LineScanParameters(
            scan_axis="Z",
            scan_start=params.z_series.z_start,
            scan_end=params.z_series.z_end,
            num_points=params.z_series.z_numbers,
            step_size=params.z_series.z_step,
            dwell_time=params.z_series.z_dwell_time
        )

        # CRITICAL: Initialize the line reconstructor
        self.line_reconstructor.initialize_scan(z_line_params)
        if self.line_scan_display:
            self.line_scan_display.set_scan_parameters(z_line_params)
            self.line_scan_display.live_plot.setData([], [])

        # Create data point handler for R+Z matrix updates
        self.current_rz_r_index = 0
        self.current_rz_z_index = 0

        # Restart line update timer
        if not self.line_update_timer.isActive():
            self.line_update_timer.start(100)

        # Clear data batch
        self.data_batch.clear()

        self.data_storage.current_detector_lag = self.detector_lag_spin.value()

        self.add_log_message("Starting R+Z series scan")

        # Start R-series controller with special handling for Z-scan
        # The R-series controller will coordinate with Z-scan controller
        self.r_series_controller.scan_type = "RZ"
        self.r_series_controller.inner_scan_controller = self.z_scan_controller
        self.r_series_controller.inner_scan_params = params
        self.r_series_controller.current_r_index = 0
        self.r_series_controller.r_series_state = ScanState.SCANNING

        # Get R-series parameters
        self.r_series_controller.r_params = params.r_series

        self.r_series_controller.r_series_started.emit()
        self.r_series_controller.status_update.emit(
            f"Starting R+Z series: {params.r_series.r_numbers} R positions × {params.z_series.z_numbers} Z positions"
        )

        # Start the R-series movement
        self.r_series_controller._move_to_next_r_slice()

    def update_rz_display_point(self, r_index: int, z_index: int, value: float):
        """Update R+Z display with new data point"""
        if hasattr(self, 'rz_series_display'):
            self.rz_series_display.update_point(r_index, z_index, value)

    @pyqtSlot(int, float, float)
    def _on_averaged_point_for_rz(self, z_idx: int, z_pos: float, avg_value: float):
        """Update R+Z heatmap with AVERAGED value from line reconstructor"""
        # Only update if we're in R+Z series mode
        if (self.r_series_active and self.r_series_scan_type == "RZ" and
                hasattr(self, 'current_rz_r_index')):
            # Use the averaged value instead of raw
            self.rz_series_display.update_point(
                self.current_rz_r_index,
                z_idx,
                avg_value  # This is the dwell-averaged value
            )

    @pyqtSlot()
    def stop_line_scan(self):
        """Stop line scan, Z-scan, or R+Z series"""
        if self.r_series_active and self.r_series_scan_type == "RZ":
            self.r_series_controller.stop_r_series()
        elif self.z_scan_mode:
            self.z_scan_controller.stop_scan()
        else:
            self.line_scan_controller.stop_scan()

    @pyqtSlot()
    def on_line_scan_started(self):
        """Handle line scan started"""
        self.line_scan_control.set_scan_state(ScanState.SCANNING)
        self.scan_status_label.setText("Line Scanning")
        self.scan_status_label.setStyleSheet("QLabel { color: green; padding: 3px; }")
        self.add_log_message("Line scan started")

        # Disable other controls
        self.scan_control.setEnabled(False)
        self.manual_control.set_enabled(False)

    @pyqtSlot()
    def on_line_scan_completed(self):
        """Handle line scan completed with detailed position tracking"""
        self.line_scan_control.set_scan_state(ScanState.IDLE)
        self.scan_status_label.setText("Idle")
        self.scan_status_label.setStyleSheet("QLabel { padding: 3px; }")

        # Stop the line update timer to freeze the display
        self.line_update_timer.stop()

        # Save remaining data batch
        if self.data_batch and self.current_hdf5_file:
            self.data_storage.save_raw_data_batch(self.current_hdf5_file, self.data_batch)
            self.data_batch.clear()

        # Auto-export if enabled
        if getattr(self, "auto_export_cb", None) and self.auto_export_cb.isChecked():
            if self.line_reconstructor and self.line_reconstructor.positions is not None:
                save_path = self.save_path_edit.text().strip() or "scan_data"

                _x, _y, z_snap, r_snap, *_ = self.data_processor.get_position_snapshot()

                if self.z_scan_mode:
                    z_export = z_snap
                    r_export = r_snap if r_snap is not None else getattr(self.line_scan_controller.line_params,
                                                                         "base_r_position", 0.0)
                else:
                    z_export = z_snap if z_snap is not None else getattr(self.line_scan_controller.line_params,
                                                                         "base_z_position", 0.0)
                    r_export = r_snap if r_snap is not None else getattr(self.line_scan_controller.line_params,
                                                                         "base_r_position", 0.0)

                self.data_storage.export_line_scan_data(
                    self.line_reconstructor.positions,
                    self.line_reconstructor.signals,
                    self.line_scan_controller.line_params,
                    save_path,
                    z_position=z_export,
                    r_position=r_export
                )

        # REPLACE THE OLD Interactive1DViewer CODE WITH THIS:
        # Auto-open in UnifiedImageViewer
        if self.enable_interactive_view and self.current_hdf5_file:
            self.auto_open_completed_scan(self.current_hdf5_file, "LINE")

        # Clear current point in reconstructor
        self.line_reconstructor.clear_current_point()

        # Reset line scan flag
        self.line_scan_active = False

        # Re-enable controls
        self.scan_control.setEnabled(True)
        self.manual_control.set_enabled(True)

        self.add_log_message(f"Line scan completed. Data saved to: {self.current_hdf5_file}")
        self.current_hdf5_file = None

    @pyqtSlot(object)
    def on_new_data_point(self, dp):
        """Handle new data point - now with non-blocking writes"""
        if not self.current_hdf5_file:
            return

        try:
            self.data_batch.append(dp)

            # Queue batch for background writing when full
            if len(self.data_batch) >= self.batch_size:
                # Send to writer thread instead of blocking here
                self.hdf5_writer.enqueue_write(self.current_hdf5_file, self.data_batch)
                self.data_batch.clear()

        except Exception as e:
            self.add_error_message(f"Data point error: {e}")

    @pyqtSlot(object)
    def on_scan_parameters_changed(self, scan_params: ScanParameters):
        """Handle scan parameter changes"""
        # Don't allow parameter changes during active scan
        if self.scan_controller.scan_state != ScanState.IDLE:
            if self.scan_controller.scan_state == ScanState.PAUSED:
                QMessageBox.warning(self, "Scan Paused",
                                    "Cannot change parameters while scan is paused. Please stop the scan first.")
            else:
                QMessageBox.warning(self, "Scan Active",
                                    "Cannot change parameters during scan. Please wait for completion or stop the scan.")
            return

        self.scan_controller.set_scan_parameters(scan_params)
        #self.image_reconstructor.initialize_scan(scan_params)
        #self.image_widget.set_scan_parameters(scan_params)

        # Auto-generate and display preview path if checkbox is checked
        if self.image_widget.show_preview_cb.isChecked():
            preview_path = self.scan_controller.get_scan_preview_path()
            self.image_widget.show_scan_preview(preview_path)

    @pyqtSlot()
    def start_scan(self):
        """Start a new scan (2D with optional Z-series or R-series)"""
        if not self.mqtt_controller.connected_status:
            QMessageBox.warning(self, "Warning", "MQTT not connected")
            return

        try:
            scan_params = self.scan_controller.scan_params
            if not scan_params:
                QMessageBox.warning(self, "Warning", "No scan parameters set")
                return

            # Check for mutual exclusivity
            if scan_params.z_series.enabled and scan_params.r_series.enabled:
                QMessageBox.warning(self, "Warning",
                                    "Cannot enable both Z-series and R-series simultaneously")
                return

            # Check if R-series is enabled
            if scan_params.r_series.enabled:
                # Start R-series scan
                self.r_series_active = True
                self.r_series_scan_type = "2D"

                # Initialize reconstructor for first slice
                self.image_reconstructor.initialize_scan(scan_params)
                self.image_widget.set_scan_parameters(scan_params)

                # Clear data batch
                self.data_batch.clear()

                # Pass detector lag to storage for metadata
                self.data_storage.current_detector_lag = self.detector_lag_spin.value()

                # Start R-series
                self.r_series_controller.start_r_series("2D", self.scan_controller, scan_params)

            elif scan_params.z_series.enabled:
                # Start Z-series scan (existing code)
                self.z_series_active = True
                self.z_series_scan_type = "2D"

                # Initialize reconstructor for first slice
                self.image_reconstructor.initialize_scan(scan_params)
                self.image_widget.set_scan_parameters(scan_params)

                # Clear data batch
                self.data_batch.clear()

                # Pass detector lag to storage for metadata
                self.data_storage.current_detector_lag = self.detector_lag_spin.value()

                # Start Z-series
                self.z_series_controller.start_z_series("2D", self.scan_controller, scan_params)

            else:
                # Regular single 2D scan (existing code)
                self.z_series_active = False
                self.r_series_active = False

                x, y, z, r, tx, ty, tz, tr, now = self.data_processor.get_position_snapshot()
                current_z = z if z is not None else 0.0
                current_r = r if r is not None else 0.0

                # Initialize the reconstructor
                self.image_reconstructor.initialize_scan(scan_params)
                self.image_widget.set_scan_parameters(scan_params)

                # Get save settings from UI
                save_path = self.save_path_edit.text().strip() or "scan_data"

                # Pass detector lag to storage for metadata
                self.data_storage.current_detector_lag = self.detector_lag_spin.value()

                # Create HDF5 file with actual positions
                self.current_hdf5_file = self.data_storage.create_hdf5_file(
                    "2D",  # Scan type for naming
                    save_path,
                    scan_params,
                    actual_z_position=current_z,
                    actual_r_position=current_r
                )

                if self.current_hdf5_file:
                    self.add_log_message(f"Created data file: {os.path.basename(self.current_hdf5_file)}")

                # Clear data batch
                self.data_batch.clear()

                detector_lag = self.detector_lag_spin.value()
                self.scan_controller.set_detector_lag(detector_lag)

                # Start regular scan
                self.scan_controller.start_scan()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start scan: {e}")
            self.add_error_message(f"Scan start error: {e}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start scan: {e}")
            self.add_error_message(f"Scan start error: {e}")

    @pyqtSlot()
    def stop_scan(self):
        """Stop the current scan"""
        # Check if R-series is active and stop it
        if self.r_series_active:
            self.r_series_controller.stop_r_series()
        # Check if Z-series is active and stop it
        elif self.z_series_active:
            self.z_series_controller.stop_z_series()
        else:
            # Regular scan stop
            self.scan_controller.stop_scan()

    @pyqtSlot()
    def on_scan_started(self):
        """Handle scan started"""
        self.scan_control.set_scan_state(ScanState.SCANNING)
        self.scan_status_label.setText("Scanning")
        self.scan_status_label.setStyleSheet("QLabel { color: green; padding: 3px; }")
        self.add_log_message("Scan started")

        # Disable manual control during scan (keeps emergency stop active)
        self.manual_control.set_enabled(False)

    @pyqtSlot()
    def on_scan_completed(self):
        """Handle scan completed with detailed position tracking"""
        # Set UI state
        self.scan_control.set_scan_state(ScanState.IDLE)
        self.scan_status_label.setText("Idle")
        self.scan_status_label.setStyleSheet("QLabel { padding: 3px; }")

        # Re-enable manual control
        self.manual_control.set_enabled(True)

        # Save remaining data batch
        if self.data_batch and self.current_hdf5_file:
            self.data_storage.save_raw_data_batch(self.current_hdf5_file, self.data_batch)
            self.data_batch.clear()

        # Save final image with position mapping
        if self.current_hdf5_file and self.image_reconstructor.image is not None:
            self.data_storage.save_final_image(
                self.current_hdf5_file,
                self.image_reconstructor.image,
                self.scan_controller.scan_params
            )

            # Auto-export if enabled
            if getattr(self, "auto_export_cb", None) and self.auto_export_cb.isChecked():
                save_path = self.save_path_edit.text().strip() or "scan_data"

                z_idx = None
                r_idx = None
                if getattr(self, "z_series_active", False):
                    z_idx = self.z_series_controller.current_z_index
                if getattr(self, "r_series_active", False):
                    r_idx = self.r_series_controller.current_r_index

                z_export = None
                r_export = None

                if getattr(self, "z_series_active", False):
                    try:
                        z_positions = self.scan_controller.scan_params.z_series.z_positions
                        if z_idx is not None and 0 <= z_idx < len(z_positions):
                            z_export = z_positions[z_idx]
                    except Exception:
                        pass

                if getattr(self, "r_series_active", False):
                    try:
                        r_positions = self.scan_controller.scan_params.r_series.r_positions
                        if r_idx is not None and 0 <= r_idx < len(r_positions):
                            r_export = r_positions[r_idx]
                    except Exception:
                        pass

                if z_export is None or r_export is None:
                    _x, _y, z_snap, r_snap, *_ = self.data_processor.get_position_snapshot()
                    if z_export is None and (z_snap is not None):
                        z_export = z_snap
                    if r_export is None and (r_snap is not None):
                        r_export = r_snap

                self.data_storage.export_reconstructed_image(
                    self.image_reconstructor.image,
                    self.scan_controller.scan_params,
                    save_path,
                    z_idx,
                    r_idx,
                    z_export,
                    r_export
                )

        # REPLACE THE OLD Interactive2DViewer CODE WITH THIS:
        # Auto-open in UnifiedImageViewer
        if self.enable_interactive_view and self.current_hdf5_file:
            self.auto_open_completed_scan(self.current_hdf5_file, "2D")

        # Clear current pixel indices to prevent stray writes
        self.image_reconstructor.clear_current_pixel()

        self.add_log_message(f"Scan completed. Data saved to: {self.current_hdf5_file}")
        self.current_hdf5_file = None

    # Helper method to create and track viewers:
    def _create_viewer(self, viewer_class, *args, **kwargs):
        """Create viewer and track it to prevent garbage collection"""
        viewer = viewer_class(*args, **kwargs, parent=self)
        viewer.destroyed.connect(lambda: self._open_viewers.remove(viewer))
        self._open_viewers.append(viewer)
        return viewer

    @pyqtSlot(str, float)
    def on_manual_move_requested(self, axis: str, position: float):
        """Handle manual movement request"""
        if self.scan_controller.scan_state != ScanState.IDLE:
            QMessageBox.warning(self, "Warning", "Cannot move during active scan")
            return

        if not self.mqtt_controller.connected_status:
            QMessageBox.warning(self, "Warning", "MQTT not connected")
            return

        # Send movement command
        command = f"MOVE/{axis}/{position:.0f}"
        self.mqtt_controller.send_command(command)
        self.add_log_message(f"Manual move: {command}")

        # Disable scan controls during manual movement
        self.scan_control.setEnabled(False)

        # Set a timer to re-enable after movement (simplified approach)
        QTimer.singleShot(3000, lambda: self.on_manual_move_complete())

    @pyqtSlot(str)
    def on_manual_stop_axis(self, axis: str):
        """Handle stop request for specific axis"""
        if not self.mqtt_controller.connected_status:
            QMessageBox.warning(self, "Warning", "MQTT not connected")
            return

        command = f"STOP/{axis}"
        self.mqtt_controller.send_command(command)
        self.add_log_message(f"Manual stop: {command}")

        # Re-enable controls
        self.on_manual_move_complete()

    @pyqtSlot()
    def on_manual_stop_all(self):
        """Handle emergency stop all axes"""
        if not self.mqtt_controller.connected_status:
            QMessageBox.warning(self, "Warning", "MQTT not connected")
            return

        # Stop all axes
        for axis in ['X', 'Y', 'Z', 'R']:
            command = f"STOP/{axis}"
            self.mqtt_controller.send_command(command)

        self.add_log_message("EMERGENCY STOP - All axes stopped")

        # Re-enable controls
        self.on_manual_move_complete()

        # Also stop any active scan
        if self.scan_controller.scan_state != ScanState.IDLE:
            self.scan_controller.stop_scan()

    def on_manual_move_complete(self):
        """Re-enable controls after manual movement"""
        self.scan_control.setEnabled(True)
        self.manual_control.set_movement_complete()
        self.add_log_message("Manual movement complete")

    @pyqtSlot()
    def on_z_series_started(self):
        """Handle Z-series started"""
        if self.z_series_scan_type == "2D":
            self.scan_control.set_scan_state(ScanState.SCANNING)
            self.scan_status_label.setText("Z-Series 2D Scanning")
        else:
            self.line_scan_control.set_scan_state(ScanState.SCANNING)
            self.scan_status_label.setText("Z-Series 1D Scanning")

        self.scan_status_label.setStyleSheet("QLabel { color: blue; padding: 3px; }")
        self.add_log_message("Z-series scan started")

        # Disable other controls
        if self.z_series_scan_type == "2D":
            self.line_scan_control.setEnabled(False)
        else:
            self.scan_control.setEnabled(False)
        self.manual_control.set_enabled(False)

    @pyqtSlot()
    def on_z_series_completed(self):
        """Handle Z-series completed"""
        if self.z_series_scan_type == "2D":
            self.scan_control.set_scan_state(ScanState.IDLE)
        else:
            self.line_scan_control.set_scan_state(ScanState.IDLE)

        self.scan_status_label.setText("Idle")
        self.scan_status_label.setStyleSheet("QLabel { padding: 3px; }")

        # Save remaining data batch
        if self.data_batch and self.current_hdf5_file:
            self.data_storage.save_raw_data_batch(self.current_hdf5_file, self.data_batch)
            self.data_batch.clear()

        # Reset flags
        self.z_series_active = False
        self.line_scan_active = False

        # Re-enable controls
        self.scan_control.setEnabled(True)
        self.line_scan_control.setEnabled(True)
        self.manual_control.set_enabled(True)

        self.add_log_message("Z-series scan completed")
        self.current_hdf5_file = None

    @pyqtSlot(int, float)
    def on_z_slice_started(self, z_index: int, z_position: float):
        """Handle Z slice started"""
        # Clear and reinitialize reconstructor for new Z-slice
        if self.z_series_scan_type == "2D":
            # Reinitialize image reconstructor for this Z-slice
            scan_params = self.scan_controller.scan_params
            self.image_reconstructor.initialize_scan(scan_params)
            self.image_widget.set_scan_parameters(scan_params)

        # Create new HDF5 file for this Z slice
        save_path = self.save_path_edit.text().strip() or "scan_data"
        if self.z_series_scan_type == "2D":
            scan_params = self.scan_controller.scan_params
            self.current_hdf5_file = self.data_storage.create_hdf5_file(
                "2D", save_path, scan_params, None, z_index, z_position
            )
        else:
            line_params = self.line_scan_controller.line_params
            self.current_hdf5_file = self.data_storage.create_hdf5_file(
                "LINE", save_path, None, line_params, z_index, z_position
            )

        if self.current_hdf5_file:
            self.add_log_message(f"Created Z-slice file: {os.path.basename(self.current_hdf5_file)}")

    @pyqtSlot(int)
    def on_z_slice_completed(self, z_index: int):
        """Handle Z slice completed"""
        # Save final data for this slice
        if self.data_batch and self.current_hdf5_file:
            self.data_storage.save_raw_data_batch(self.current_hdf5_file, self.data_batch)
            self.data_batch.clear()

        # Save final image/line data
        if self.current_hdf5_file:
            if self.z_series_scan_type == "2D" and self.image_reconstructor.image is not None:
                self.data_storage.save_final_image(
                    self.current_hdf5_file,
                    self.image_reconstructor.image,
                    self.scan_controller.scan_params
                )
            # For line scans, the data is already saved in the raw data

        self.add_log_message(f"Completed Z-slice {z_index + 1}")

    @pyqtSlot(int, int)
    def on_z_series_progress(self, current_z: int, total_z: int):
        """Handle Z-series progress update"""
        if self.z_series_scan_type == "2D":
            # Update 2D scan progress bar with Z-series info
            progress = int((current_z / total_z) * 100)
            self.scan_control.progress_bar.setValue(progress)
            self.scan_control.progress_bar.setFormat(f"Z-slice {current_z}/{total_z} ({progress}%)")
        else:
            # Update 1D Z-scan progress bar - use "Z-point" not "Z-slice"
            progress = int((current_z / total_z) * 100)
            self.line_scan_control.progress_bar.setValue(progress)
            self.line_scan_control.progress_bar.setFormat(f"Z-point {current_z}/{total_z} ({progress}%)")

    @pyqtSlot()
    def on_z_scan_started(self):
        """Handle Z-scan started"""
        self.line_scan_control.set_scan_state(ScanState.SCANNING)
        self.scan_status_label.setText("Z-Scanning")
        self.scan_status_label.setStyleSheet("QLabel { color: purple; padding: 3px; }")
        self.add_log_message("Z-scan started")

        # Disable other controls
        self.scan_control.setEnabled(False)
        self.manual_control.set_enabled(False)

    @pyqtSlot()
    def on_z_scan_completed(self):
        """Handle Z-scan completed with detailed position tracking"""
        # Guard re-entry
        if not getattr(self, 'z_scan_mode', False):
            return

        # UI state
        self.line_scan_control.set_scan_state(ScanState.IDLE)
        self.scan_status_label.setText("Idle")
        self.scan_status_label.setStyleSheet("QLabel { padding: 3px; }")

        # Save any remaining raw-data first
        if self.data_batch and self.current_hdf5_file:
            try:
                self.data_storage.save_raw_data_batch(self.current_hdf5_file, self.data_batch)
            except Exception as e:
                self.add_log_message(f"[ERROR] Final Z-scan batch save failed: {e}")
            finally:
                self.data_batch.clear()

        # Auto-export if enabled
        _x, _y, z_snap, r_snap, *_ = self.data_processor.get_position_snapshot()
        r_export = r_snap if r_snap is not None else 0.0

        if getattr(self, "auto_export_cb", None) and self.auto_export_cb.isChecked():
            if self.line_reconstructor and self.line_reconstructor.positions is not None:
                save_path = self.save_path_edit.text().strip() or "scan_data"

                # Use the actual line_params with z_series information instead of creating new ones
                self.data_storage.export_line_scan_data(
                    self.line_reconstructor.positions,
                    self.line_reconstructor.signals,
                    self.line_scan_controller.line_params,  # CHANGED: Use actual params with fixed X/Y
                    save_path,
                    z_position=None,
                    r_position=r_export
                )

                self.add_log_message(f"Exported Z-scan data to {save_path}/")

        # Auto-open in UnifiedImageViewer
        if self.enable_interactive_view and self.current_hdf5_file:
            self.auto_open_completed_scan(self.current_hdf5_file, "Z")

        # Force a final render of the line trace
        if hasattr(self.line_reconstructor, 'emit_full_line'):
            self.line_reconstructor.emit_full_line()

        # Reset flags
        self.z_scan_mode = False
        self.line_scan_active = False

        # Re-enable controls
        self.scan_control.setEnabled(True)
        self.manual_control.set_enabled(True)

        # Log & release file handle
        if self.current_hdf5_file:
            self.add_log_message(f"Z-scan completed. Data saved to: {self.current_hdf5_file}")
            self.current_hdf5_file = None

    @pyqtSlot(int)
    def on_z_scan_position_update(self, z_index: int):
        """Handle Z-scan position update for display"""
        # This helps track current Z position for the reconstructor
        if self.r_series_active and self.r_series_scan_type == "RZ":
            self.current_rz_z_index = z_index

    @pyqtSlot()
    def on_r_series_started(self):
        """Handle R-series started"""
        if self.r_series_scan_type == "2D":
            self.scan_control.set_scan_state(ScanState.SCANNING)
            self.scan_status_label.setText("R-Series 2D Scanning")
        else:  # "RZ"
            self.line_scan_control.set_scan_state(ScanState.SCANNING)
            self.scan_status_label.setText("R+Z Series Scanning")

        self.scan_status_label.setStyleSheet("QLabel { color: purple; padding: 3px; }")
        self.add_log_message("R-series scan started")

        # Disable other controls
        if self.r_series_scan_type == "2D":
            self.line_scan_control.setEnabled(False)
        else:
            self.scan_control.setEnabled(False)
        self.manual_control.set_enabled(False)

    @pyqtSlot()
    def on_r_series_completed(self):
        """Handle R-series completed"""
        if self.r_series_scan_type == "2D":
            self.scan_control.set_scan_state(ScanState.IDLE)
        else:
            self.line_scan_control.set_scan_state(ScanState.IDLE)

        self.scan_status_label.setText("Idle")
        self.scan_status_label.setStyleSheet("QLabel { padding: 3px; }")

        # Save remaining data batch
        if self.data_batch and self.current_hdf5_file:
            self.data_storage.save_raw_data_batch(self.current_hdf5_file, self.data_batch)
            self.data_batch.clear()

        # Reset flags
        self.r_series_active = False
        self.line_scan_active = False

        # Re-enable controls
        self.scan_control.setEnabled(True)
        self.line_scan_control.setEnabled(True)
        self.manual_control.set_enabled(True)

        self.add_log_message("R-series scan completed")
        self.current_hdf5_file = None

    @pyqtSlot(int, float)
    def on_r_slice_started(self, r_index: int, r_position: float):
        """Handle R slice started"""

        # Track current R index for R+Z display
        if self.r_series_scan_type == "RZ":
            self.current_rz_r_index = r_index
            self.current_rz_z_index = 0

        # Clear and reinitialize reconstructor for new R-slice
        if self.r_series_scan_type == "2D":
            # Reinitialize image reconstructor for this R-slice
            # Use the updated scan parameters from R-series controller (transformed area)
            scan_params = self.r_series_controller.inner_scan_params
            self.image_reconstructor.initialize_scan(scan_params)
            self.image_widget.set_scan_parameters(scan_params)

        # Create new HDF5 file for this R slice
        save_path = self.save_path_edit.text().strip() or "scan_data"

        if self.r_series_scan_type == "2D":
            scan_params = self.scan_controller.scan_params

            # Get current Z position to record with this R slice
            x, y, z, r, tx, ty, tz, tr, now = self.data_processor.get_position_snapshot()
            current_z = z if z is not None else scan_params.base_z_position

            self.current_hdf5_file = self.data_storage.create_hdf5_file(
                "2D", save_path, scan_params, None,
                r_index=r_index, r_position=r_position,
                actual_z_position=current_z  # ADD THIS - record Z at this R position
            )
        else:  # "RZ"
            line_params = self.line_scan_controller.line_params
            self.current_hdf5_file = self.data_storage.create_hdf5_file(
                "Z", save_path, None, line_params,
                r_index=r_index, r_position=r_position
            )

        if self.current_hdf5_file:
            self.add_log_message(f"Created R-slice file: {os.path.basename(self.current_hdf5_file)}")

    @pyqtSlot(int)
    def on_r_slice_completed(self, r_index: int):
        """Handle R slice completed"""
        # Save final data for this slice
        if self.data_batch and self.current_hdf5_file:
            self.data_storage.save_raw_data_batch(self.current_hdf5_file, self.data_batch)
            self.data_batch.clear()

        # Save final image/data
        if self.current_hdf5_file:
            if self.r_series_scan_type == "2D" and self.image_reconstructor.image is not None:
                self.data_storage.save_final_image(
                    self.current_hdf5_file,
                    self.image_reconstructor.image,
                    self.scan_controller.scan_params
                )

        self.add_log_message(f"Completed R-slice {r_index + 1}")

    @pyqtSlot(int, int)
    def on_r_series_progress(self, current_r: int, total_r: int):
        """Handle R-series progress update"""
        # Get current R position for display
        r_pos = self.r_series_controller.r_params.r_positions[current_r - 1] if current_r > 0 else 0

        if self.r_series_scan_type == "2D":
            # Update 2D scan progress bar with detailed R-series info
            progress = int((current_r / total_r) * 100)
            self.scan_control.progress_bar.setValue(progress)
            self.scan_control.progress_bar.setFormat(
                f"R {current_r}/{total_r} ({r_pos / 1000:.1f}°) - {progress}%"
            )

            # Update status bar with current R position
            self.scan_status_label.setText(f"R-Series 2D: R={r_pos / 1000:.1f}°")

        else:  # "RZ"
            # For R+Z series, show both R and Z progress
            progress = int((current_r / total_r) * 100)
            self.line_scan_control.progress_bar.setValue(progress)

            # If Z-scan is active, show combined progress
            if self.z_scan_controller.scan_state == ScanState.SCANNING:
                z_current = self.z_scan_controller.current_z_index + 1
                z_total = len(self.z_scan_controller.z_positions)
                total_measurements = total_r * z_total
                current_measurement = (current_r - 1) * z_total + z_current

                self.line_scan_control.progress_bar.setFormat(
                    f"R{current_r}/{total_r} Z{z_current}/{z_total} "
                    f"({current_measurement}/{total_measurements})"
                )
            else:
                self.line_scan_control.progress_bar.setFormat(
                    f"R {current_r}/{total_r} ({r_pos / 1000:.1f}°) - {progress}%"
                )

    @pyqtSlot(dict)
    def update_statistics(self, stats: dict):
        """Update statistics display with improved metrics"""
        self.stats_table.setRowCount(0)

        display_stats = [
            ("Runtime", f"{stats.get('runtime', 0):.1f} s"),
            ("Position Messages", f"{stats.get('position_messages', 0):,}"),
            ("Current Messages", f"{stats.get('current_messages', 0):,}"),
            ("Data Points Created", f"{stats.get('data_points_created', 0):,}"),
            ("Position Rate", f"{stats.get('position_rate', 0):.1f} Hz"),
            ("Current Rate", f"{stats.get('current_rate', 0):.1f} Hz"),
            ("DataPoint Rate", f"{stats.get('datapoint_rate', 0):.1f} Hz"),
            ("Data Batch Size", f"{len(self.data_batch)}/{self.batch_size}"),
        ]

        if self.scan_controller.scan_state == ScanState.SCANNING:
            completion = self.image_reconstructor.get_completion_percentage()
            display_stats.append(("Scan Progress", f"{completion:.1f}%"))

        for i, (param, value) in enumerate(display_stats):
            self.stats_table.insertRow(i)
            self.stats_table.setItem(i, 0, QTableWidgetItem(param))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(value)))

    def add_log_message(self, message: str):
        """Add message to log"""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        formatted_message = f"[{timestamp}] {message}"

        self.log_text.append(formatted_message)

        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)

        # Limit log size
        if self.log_text.document().blockCount() > 1000:
            cursor.movePosition(QTextCursor.Start)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()

    def add_error_message(self, message: str):
        """Add error message to log"""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        formatted_message = f'<span style="color: red;">[{timestamp}] ERROR: {message}</span>'

        self.log_text.append(formatted_message)

        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)

    # --- Settings: save/load everything to scanner_settings.ini (same folder as scanner.py) ---

    def get_settings_path(self):
        """Settings file path in the same directory as scanner.py"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, "scanner_settings.ini")

    def save_settings(self):
        """Persist all current parameters (2D, Z-series, R-series, Line/Z/RZ, connection/storage, display)."""
        try:
            settings_file = self.get_settings_path()
            s = QSettings(settings_file, QSettings.IniFormat)
            self.add_log_message(f"Saving settings to: {settings_file}")

            # Wipe existing keys so removed items don't linger
            s.clear()

            # -------- General / Connection & Storage --------
            s.beginGroup("General")
            s.setValue("version", "2")  # bump when you extend
            s.endGroup()

            s.beginGroup("Connection")
            s.setValue("broker_address", self.broker_edit.text())
            s.setValue("broker_port", int(self.port_spin.value()))
            s.endGroup()

            s.beginGroup("Storage")
            s.setValue("save_path", self.save_path_edit.text())
            if hasattr(self, "auto_export_cb"):
                s.setValue("auto_export", bool(self.auto_export_cb.isChecked()))
            s.endGroup()

            s.beginGroup("Detector")
            if hasattr(self, "detector_lag_spin"):
                s.setValue("lag_seconds", float(self.detector_lag_spin.value()))
            s.endGroup()

            # -------- 2D Scan (ScanControlWidget) --------
            sc = self.scan_control  # shorthand

            s.beginGroup("Scan2D")
            s.setValue("x_start", float(sc.x_start_spin.value()))
            s.setValue("x_end", float(sc.x_end_spin.value()))
            s.setValue("y_start", float(sc.y_start_spin.value()))
            s.setValue("y_end", float(sc.y_end_spin.value()))
            s.setValue("x_pixels", int(sc.x_pixels_spin.value()))
            s.setValue("y_pixels", int(sc.y_pixels_spin.value()))
            s.setValue("x_step_input", float(sc.x_step_spin.value()))
            s.setValue("y_step_input", float(sc.y_step_spin.value()))
            # Mode/pattern saved by text to avoid enum mismatches
            s.setValue("mode_text", sc.mode_combo.currentText())
            s.setValue("pattern_text", sc.pattern_combo.currentText())
            s.setValue("dwell_time", float(sc.dwell_time_spin.value()))
            s.setValue("scan_speed", float(sc.scan_speed_spin.value()))
            s.setValue("base_z_position", float(sc.base_z_spin.value()))
            s.setValue("base_r_position", float(sc.base_r_spin.value()))
            s.endGroup()

            # -------- 2D: Z-series --------
            s.beginGroup("Scan2D_ZSeries")
            s.setValue("enabled", bool(sc.z_series_enable_cb.isChecked()))
            s.setValue("z_start", float(sc.z_start_spin.value()))
            s.setValue("z_end", float(sc.z_end_spin.value()))
            s.setValue("z_numbers", int(sc.z_numbers_spin.value()))
            s.setValue("z_step_input", float(sc.z_step_spin.value()))
            s.setValue("x_comp_ratio", float(sc.x_compensation_spin.value()))
            s.setValue("y_comp_ratio", float(sc.y_compensation_spin.value()))
            s.endGroup()

            # -------- 2D: R-series --------
            s.beginGroup("Scan2D_RSeries")
            s.setValue("enabled", bool(sc.r_series_enable_cb.isChecked()))
            s.setValue("r_start", float(sc.r_start_spin.value()))
            s.setValue("r_end", float(sc.r_end_spin.value()))
            s.setValue("r_numbers", int(sc.r_numbers_spin.value()))
            s.setValue("r_step_input", float(sc.r_step_spin.value()))
            s.setValue("base_r_position", float(sc.base_r_spin.value()))
            s.setValue("cor_enabled", bool(sc.r_mode_combo.currentText().lower() != "none"))
            s.setValue("cor_mode_text", sc.r_mode_combo.currentText())
            s.setValue("cor_x", float(sc.cor_x_spin.value()))
            s.setValue("cor_y", float(sc.cor_y_spin.value()))
            s.setValue("cor_base_z", float(sc.cor_base_z_spin.value()))
            s.setValue("cor_x_comp_ratio", float(sc.cor_x_comp_spin.value()))
            s.setValue("cor_y_comp_ratio", float(sc.cor_y_comp_spin.value()))
            s.setValue("transform_text", sc.r_transform_combo.currentText())
            s.endGroup()

            # -------- Line / Z / R+Z (LineScanControlWidget) --------
            lc = self.line_scan_control

            s.beginGroup("LineCommon")
            s.setValue("mode_text", lc.z_mode_combo.currentText())  # "Line Scan", "Z-Scan", "R+Z Series"
            s.endGroup()

            # Line Scan settings
            s.beginGroup("LineScan")
            s.setValue("fixed_axis", lc.fixed_axis_combo.currentText())
            s.setValue("fixed_position", float(lc.fixed_position_spin.value()))
            s.setValue("scan_start", float(lc.scan_start_spin.value()))
            s.setValue("scan_end", float(lc.scan_end_spin.value()))
            s.setValue("num_points", int(lc.num_points_spin.value()))
            s.setValue("step_size", float(lc.step_size_spin.value()))
            s.setValue("dwell_time", float(lc.dwell_time_spin.value()))
            s.setValue("base_z_position", float(lc.z_base_z_spin.value()))
            s.endGroup()

            # Z-Scan settings
            s.beginGroup("ZScan")
            s.setValue("fixed_x", float(lc.z_fixed_x_spin.value()))
            s.setValue("fixed_y", float(lc.z_fixed_y_spin.value()))
            s.setValue("z_start", float(lc.z_start_spin.value()))
            s.setValue("z_end", float(lc.z_end_spin.value()))
            s.setValue("z_numbers", int(lc.z_numbers_spin.value()))
            s.setValue("z_step_input", float(lc.z_step_spin.value()))
            s.setValue("z_dwell_time", float(lc.z_dwell_spin.value()))
            s.setValue("x_comp_ratio", float(lc.z_x_compensation_spin.value()))
            s.setValue("y_comp_ratio", float(lc.z_y_compensation_spin.value()))
            s.setValue("base_z_position", float(lc.z_base_z_spin.value()))
            s.endGroup()

            # R+Z series (within LineScanControlWidget)
            s.beginGroup("RZSeries")
            s.setValue("r_start", float(lc.rz_r_start_spin.value()))
            s.setValue("r_end", float(lc.rz_r_end_spin.value()))
            s.setValue("r_numbers", int(lc.rz_r_numbers_spin.value()))
            s.setValue("r_step_input", float(lc.rz_r_step_spin.value()))
            s.setValue("base_r_position", float(lc.rz_base_r_spin.value()))
            s.setValue("cor_x", float(lc.rz_cor_x_spin.value()))
            s.setValue("cor_y", float(lc.rz_cor_y_spin.value()))
            s.setValue("cor_base_z", float(lc.rz_cor_base_z_spin.value()))
            s.setValue("cor_x_comp_ratio", float(lc.rz_cor_x_comp_spin.value()))
            s.setValue("cor_y_comp_ratio", float(lc.rz_cor_y_comp_spin.value()))
            s.setValue("transform_text", lc.rz_transform_combo.currentText())
            s.endGroup()

            # -------- Display prefs (lightweight) --------
            s.beginGroup("Display")
            try:
                # 2D image widget has a 'current_colormap' attribute
                s.setValue("colormap", getattr(self.image_widget, "current_colormap", "viridis"))
            except Exception:
                pass
            s.endGroup()

            self.add_log_message("Settings saved ✅")
        except Exception as e:
            self.add_error_message(f"Failed to save settings: {e}")

    def load_settings(self):
        """Load settings back into the UI safely (ignore missing keys)."""
        try:
            settings_file = self.get_settings_path()
            s = QSettings(settings_file, QSettings.IniFormat)
            self.add_log_message(f"Loading settings from: {settings_file}")

            # Helpers
            def set_text_combo(combo, text):
                if not hasattr(combo, "findText"): return
                idx = combo.findText(str(text))
                if idx >= 0:
                    combo.setCurrentIndex(idx)

            # -------- General / Connection & Storage --------
            s.beginGroup("Connection")
            if s.contains("broker_address"):
                self.broker_edit.setText(s.value("broker_address", type=str))
            if s.contains("broker_port"):
                self.port_spin.setValue(s.value("broker_port", type=int))
            s.endGroup()

            s.beginGroup("Storage")
            if s.contains("save_path"):
                self.save_path_edit.setText(s.value("save_path", type=str))
            if s.contains("auto_export") and hasattr(self, "auto_export_cb"):
                self.auto_export_cb.setChecked(s.value("auto_export", type=bool))
            s.endGroup()

            s.beginGroup("Detector")
            if s.contains("lag_seconds") and hasattr(self, "detector_lag_spin"):
                self.detector_lag_spin.setValue(float(s.value("lag_seconds", type=float)))
            s.endGroup()

            # -------- 2D Scan --------
            sc = self.scan_control

            s.beginGroup("Scan2D")
            for key, wname, typ in [
                ("x_start", "x_start_spin", float),
                ("x_end", "x_end_spin", float),
                ("y_start", "y_start_spin", float),
                ("y_end", "y_end_spin", float),
                ("x_pixels", "x_pixels_spin", int),
                ("y_pixels", "y_pixels_spin", int),
                ("x_step_input", "x_step_spin", float),
                ("y_step_input", "y_step_spin", float),
                ("dwell_time", "dwell_time_spin", float),
                ("scan_speed", "scan_speed_spin", float),
                ("base_z_position", "base_z_spin", float),
                ("base_r_position", "base_r_spin", float),
            ]:
                if s.contains(key):
                    getattr(sc, wname).setValue(s.value(key, type=typ))
            if s.contains("mode_text"):
                set_text_combo(sc.mode_combo, s.value("mode_text", type=str))
            if s.contains("pattern_text"):
                set_text_combo(sc.pattern_combo, s.value("pattern_text", type=str))
            s.endGroup()

            # -------- 2D: Z-series --------
            s.beginGroup("Scan2D_ZSeries")
            if s.contains("enabled"):
                sc.z_series_enable_cb.setChecked(s.value("enabled", type=bool))
            for key, wname, typ in [
                ("z_start", "z_start_spin", float),
                ("z_end", "z_end_spin", float),
                ("z_numbers", "z_numbers_spin", int),
                ("z_step_input", "z_step_spin", float),
                ("x_comp_ratio", "x_compensation_spin", float),
                ("y_comp_ratio", "y_compensation_spin", float),
            ]:
                if s.contains(key):
                    getattr(sc, wname).setValue(s.value(key, type=typ))
            s.endGroup()

            # -------- 2D: R-series --------
            s.beginGroup("Scan2D_RSeries")
            if s.contains("enabled"):
                sc.r_series_enable_cb.setChecked(s.value("enabled", type=bool))
            for key, wname, typ in [
                ("r_start", "r_start_spin", float),
                ("r_end", "r_end_spin", float),
                ("r_numbers", "r_numbers_spin", int),
                ("r_step_input", "r_step_spin", float),
                ("base_r_position", "base_r_spin", float),
                ("cor_x", "cor_x_spin", float),
                ("cor_y", "cor_y_spin", float),
                ("cor_base_z", "cor_base_z_spin", float),
                ("cor_x_comp_ratio", "cor_x_comp_spin", float),
                ("cor_y_comp_ratio", "cor_y_comp_spin", float),
            ]:
                if s.contains(key):
                    getattr(sc, wname).setValue(s.value(key, type=typ))
            if s.contains("cor_mode_text"):
                set_text_combo(sc.r_mode_combo, s.value("cor_mode_text", type=str))
            if s.contains("transform_text"):
                set_text_combo(sc.r_transform_combo, s.value("transform_text", type=str))
            s.endGroup()

            # -------- Line / Z / R+Z --------
            lc = self.line_scan_control

            s.beginGroup("LineCommon")
            if s.contains("mode_text"):
                set_text_combo(lc.z_mode_combo, s.value("mode_text", type=str))
            s.endGroup()

            s.beginGroup("LineScan")
            if s.contains("fixed_axis"):
                set_text_combo(lc.fixed_axis_combo, s.value("fixed_axis", type=str))
            for key, wname, typ in [
                ("fixed_position", "fixed_position_spin", float),
                ("scan_start", "scan_start_spin", float),
                ("scan_end", "scan_end_spin", float),
                ("num_points", "num_points_spin", int),
                ("step_size", "step_size_spin", float),
                ("dwell_time", "dwell_time_spin", float),
                ("base_z_position", "z_base_z_spin", float),
            ]:
                if s.contains(key):
                    getattr(lc, wname).setValue(s.value(key, type=typ))
            s.endGroup()

            s.beginGroup("ZScan")
            for key, wname, typ in [
                ("fixed_x", "z_fixed_x_spin", float),
                ("fixed_y", "z_fixed_y_spin", float),
                ("z_start", "z_start_spin", float),
                ("z_end", "z_end_spin", float),
                ("z_numbers", "z_numbers_spin", int),
                ("z_step_input", "z_step_spin", float),
                ("z_dwell_time", "z_dwell_spin", float),
                ("x_comp_ratio", "z_x_compensation_spin", float),
                ("y_comp_ratio", "z_y_compensation_spin", float),
                ("base_z_position", "z_base_z_spin", float),
            ]:
                if s.contains(key):
                    getattr(lc, wname).setValue(s.value(key, type=typ))
            s.endGroup()

            s.beginGroup("RZSeries")
            for key, wname, typ in [
                ("r_start", "rz_r_start_spin", float),
                ("r_end", "rz_r_end_spin", float),
                ("r_numbers", "rz_r_numbers_spin", int),
                ("r_step_input", "rz_r_step_spin", float),
                ("base_r_position", "rz_base_r_spin", float),
                ("cor_x", "rz_cor_x_spin", float),
                ("cor_y", "rz_cor_y_spin", float),
                ("cor_base_z", "rz_cor_base_z_spin", float),
                ("cor_x_comp_ratio", "rz_cor_x_comp_spin", float),
                ("cor_y_comp_ratio", "rz_cor_y_comp_spin", float),
            ]:
                if s.contains(key):
                    getattr(lc, wname).setValue(s.value(key, type=typ))
            if s.contains("transform_text"):
                set_text_combo(lc.rz_transform_combo, s.value("transform_text", type=str))
            s.endGroup()

            # -------- Display --------
            s.beginGroup("Display")
            if s.contains("colormap") and hasattr(self.image_widget, "set_colormap"):
                self.image_widget.set_colormap(s.value("colormap", type=str))
            s.endGroup()

            self.add_log_message("Settings loaded ✅")
        except Exception as e:
            self.add_error_message(f"Failed to load settings: {e}")

    def show_settings_location(self):
        """Show where settings are stored (same folder as scanner.py)."""
        settings_file = self.get_settings_path()
        exists = "✓ EXISTS" if os.path.exists(settings_file) else "✗ NOT FOUND"
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setText(f"Settings file:\n\n{settings_file}\n\nStatus: {exists}")
        msg.setWindowTitle("Settings Location")
        msg.setDetailedText(f"Platform: {sys.platform}\nFile format: INI")
        msg.exec_()
        self.add_log_message(f"Settings location: {settings_file} ({exists})")

    def closeEvent(self, event):
        """Handle application close"""
        try:

            # Save settings before closing
            self.save_settings()

            # Stop any running R-series scan
            if hasattr(self, 'r_series_controller') and self.r_series_active:
                self.r_series_controller.stop_r_series()

            # Stop any running scan
            if self.scan_controller.scan_state != ScanState.IDLE:
                self.scan_controller.stop_scan()

            # Stop any running line scan
            if self.line_scan_controller.scan_state != ScanState.IDLE:
                self.line_scan_controller.stop_scan()

                # Save any remaining data synchronously (final flush)
            if self.data_batch and self.current_hdf5_file:
                self.data_storage.save_raw_data_batch(self.current_hdf5_file, self.data_batch)

                # Stop HDF5 writer thread
            if hasattr(self, 'hdf5_writer'):
                self.hdf5_writer.stop()

                # Close all image viewers
            if hasattr(self, '_image_viewers'):
                for viewer in self._image_viewers:
                    if viewer and not viewer.isHidden():
                        viewer.close()

            # Save final image if available (2D scan)
            if self.current_hdf5_file and hasattr(self.image_reconstructor, 'image'):
                if self.image_reconstructor.image is not None:
                    self.data_storage.save_final_image(
                        self.current_hdf5_file,
                        self.image_reconstructor.image,
                        self.scan_controller.scan_params
                    )

            # Stop all timers
            self.statistics_timer.stop()
            self.image_update_timer.stop()
            self.line_update_timer.stop()  # NEW: Stop line update timer

            # Disconnect MQTT
            self.mqtt_controller.disconnect_mqtt()

        except Exception as e:
            print(f"Error during cleanup: {e}")

        event.accept()

class ScanningMicroscopeApp(QApplication):
    """Main application class"""

    def __init__(self, argv):
        super().__init__(argv)

        # Set application properties
        self.setApplicationName("Scanning Microscope")
        self.setApplicationVersion("1.0")
        self.setOrganizationName("Research Lab")

        # Set application style
        self.setStyle('Fusion')

        # Optional: Set custom palette for modern look
        self.set_modern_style()

    def set_modern_style(self):
        """Set modern application style"""
        palette = QPalette()

        # Window colors
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))

        # Base colors
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))

        # Text colors
        palette.setColor(QPalette.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))

        # Button colors
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))

        # Highlight colors
        palette.setColor(QPalette.Highlight, QColor(76, 163, 224))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

        self.setPalette(palette)

def main():
    """Main entry point"""
    # Enable high DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Create application
    app = ScanningMicroscopeApp(sys.argv)

    # Create and show main window
    window = MainWindow()
    window.show()

    # Print startup information
    print("=" * 80)
    print("Scanning Microscope System v1.0")
    print("=" * 80)
    print("Features:")
    print("  ✓ High-frequency MQTT data acquisition (6kHz+ ECC, 40Hz picoammeter)")
    print("  ✓ Step-and-stop and continuous scanning modes")
    print("  ✓ Real-time image reconstruction and display")
    print("  ✓ HDF5 raw data storage with SQLite metadata")
    print("  ✓ Multi-threaded architecture for optimal performance")
    print("  ✓ Professional Qt-based GUI with dockable panels")
    print("=" * 80)
    print("System Requirements:")
    print("  • MQTT broker (mosquitto) running on localhost:1883")
    print("  • ECC100 controllers connected and streaming position data")
    print("  • Picoammeter 9103 connected and streaming current data")
    print("  • Sufficient disk space for data storage")
    if not HDF5_AVAILABLE:
        print("  ! h5py not installed - raw data storage disabled")
    if not SCIPY_AVAILABLE:
        print("  ! scipy not installed - advanced interpolation disabled")
    print("=" * 80)
    print("Quick Start:")
    print("  1. Ensure MQTT broker is running")
    print("  2. Connect to MQTT using the connection panel")
    print("  3. Set scan parameters (area, resolution, mode)")
    print("  4. Click 'Start Scan' to begin image acquisition")
    print("=" * 80)
    print("Data Storage:")
    print(f"  • Scan metadata: {os.path.abspath('scan_data/scan_metadata.db')}")
    print(f"  • Raw data: {os.path.abspath('scan_data/')}scan_XXXXXX_*.h5")
    print("=" * 80)
    print()

    # Run application
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("\nShutting down...")
        window.close()
        app.quit()

if __name__ == "__main__":
    main()
