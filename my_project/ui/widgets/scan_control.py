"""Scan control widget."""
from __future__ import annotations

from config.constants import *
from core.models import ScanMode, ScanParameters, ScanState
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ScanControlWidget(QWidget):
    """Widget for scan parameter control."""

    # Signals
    scan_parameters_changed = pyqtSignal(object)  # ScanParameters
    start_scan_requested = pyqtSignal()
    pause_scan_requested = pyqtSignal()
    stop_scan_requested = pyqtSignal()
    scan_path_preview = pyqtSignal(list)  # List of (x, y) tuples

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.setup_ui()
        self.connect_signals()
        self.load_defaults()

    def setup_ui(self) -> None:
        """Setup the scan control UI."""
        layout = QVBoxLayout()

        # Scan area group
        layout.addWidget(self._create_scan_area_group())

        # Resolution group
        layout.addWidget(self._create_resolution_group())

        # Scan mode group
        layout.addWidget(self._create_scan_mode_group())

        # Scan estimate group
        layout.addWidget(self._create_estimate_group())

        # Control buttons
        layout.addLayout(self._create_control_buttons())

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def _create_scan_area_group(self) -> QGroupBox:
        """Create scan area group."""
        group = QGroupBox("Scan Area")
        layout = QGridLayout()

        # X parameters
        layout.addWidget(QLabel("X Start (nm):"), 0, 0)
        self.x_start_spin = QDoubleSpinBox()
        self.x_start_spin.setRange(MIN_POSITION_NM, MAX_POSITION_NM)
        self.x_start_spin.setValue(0)
        layout.addWidget(self.x_start_spin, 0, 1)

        layout.addWidget(QLabel("X End (nm):"), 0, 2)
        self.x_end_spin = QDoubleSpinBox()
        self.x_end_spin.setRange(MIN_POSITION_NM, MAX_POSITION_NM)
        self.x_end_spin.setValue(10000)
        layout.addWidget(self.x_end_spin, 0, 3)

        # Y parameters
        layout.addWidget(QLabel("Y Start (nm):"), 1, 0)
        self.y_start_spin = QDoubleSpinBox()
        self.y_start_spin.setRange(MIN_POSITION_NM, MAX_POSITION_NM)
        self.y_start_spin.setValue(0)
        layout.addWidget(self.y_start_spin, 1, 1)

        layout.addWidget(QLabel("Y End (nm):"), 1, 2)
        self.y_end_spin = QDoubleSpinBox()
        self.y_end_spin.setRange(MIN_POSITION_NM, MAX_POSITION_NM)
        self.y_end_spin.setValue(10000)
        layout.addWidget(self.y_end_spin, 1, 3)

        group.setLayout(layout)
        return group

    def _create_resolution_group(self) -> QGroupBox:
        """Create resolution group."""
        group = QGroupBox("Resolution")
        layout = QGridLayout()

        layout.addWidget(QLabel("X Pixels:"), 0, 0)
        self.x_pixels_spin = QSpinBox()
        self.x_pixels_spin.setRange(MIN_PIXELS, MAX_PIXELS)
        self.x_pixels_spin.setValue(self.config.default_x_pixels)
        layout.addWidget(self.x_pixels_spin, 0, 1)

        layout.addWidget(QLabel("Y Pixels:"), 0, 2)
        self.y_pixels_spin = QSpinBox()
        self.y_pixels_spin.setRange(MIN_PIXELS, MAX_PIXELS)
        self.y_pixels_spin.setValue(self.config.default_y_pixels)
        layout.addWidget(self.y_pixels_spin, 0, 3)

        # Step size display
        layout.addWidget(QLabel("X Step (nm):"), 1, 0)
        self.x_step_label = QLabel("100.0")
        self.x_step_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 3px; }")
        layout.addWidget(self.x_step_label, 1, 1)

        layout.addWidget(QLabel("Y Step (nm):"), 1, 2)
        self.y_step_label = QLabel("100.0")
        self.y_step_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 3px; }")
        layout.addWidget(self.y_step_label, 1, 3)

        group.setLayout(layout)
        return group

    def _create_scan_mode_group(self) -> QGroupBox:
        """Create scan mode group."""
        group = QGroupBox("Scan Mode")
        layout = QGridLayout()

        layout.addWidget(QLabel("Mode:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Step and Stop", ScanMode.STEP_STOP)
        self.mode_combo.addItem("Continuous (Under Construction)", ScanMode.CONTINUOUS)
        layout.addWidget(self.mode_combo, 0, 1, 1, 2)

        # Bidirectional checkbox
        layout.addWidget(QLabel("Bidirectional:"), 1, 0)
        self.bidirectional_cb = QCheckBox()
        self.bidirectional_cb.setChecked(True)
        layout.addWidget(self.bidirectional_cb, 1, 1)

        # Preview button
        self.preview_btn = QPushButton("Preview Path")
        self.preview_btn.clicked.connect(self.preview_scan_path)
        layout.addWidget(self.preview_btn, 1, 2, 1, 2)

        # Mode-specific parameters
        layout.addWidget(QLabel("Dwell Time (s):"), 2, 0)
        self.dwell_time_spin = QDoubleSpinBox()
        self.dwell_time_spin.setRange(MIN_DWELL_TIME, MAX_DWELL_TIME)
        self.dwell_time_spin.setSingleStep(0.1)
        self.dwell_time_spin.setValue(self.config.default_dwell_time)
        layout.addWidget(self.dwell_time_spin, 2, 1)

        layout.addWidget(QLabel("Scan Speed (nm/s):"), 2, 2)
        self.scan_speed_spin = QDoubleSpinBox()
        self.scan_speed_spin.setRange(MIN_SCAN_SPEED, MAX_SCAN_SPEED)
        self.scan_speed_spin.setValue(self.config.default_scan_speed)
        self.scan_speed_spin.setEnabled(False)
        layout.addWidget(self.scan_speed_spin, 2, 3)

        # Notice label
        self.mode_notice_label = QLabel("")
        self.mode_notice_label.setStyleSheet("QLabel { color: orange; padding: 5px; }")
        self.mode_notice_label.setWordWrap(True)
        layout.addWidget(self.mode_notice_label, 3, 0, 1, 4)

        group.setLayout(layout)
        return group

    def _create_estimate_group(self) -> QGroupBox:
        """Create scan estimate group."""
        group = QGroupBox("Scan Estimate")
        layout = QGridLayout()

        layout.addWidget(QLabel("Total Pixels:"), 0, 0)
        self.total_pixels_label = QLabel("10,000")
        layout.addWidget(self.total_pixels_label, 0, 1)

        layout.addWidget(QLabel("Estimated Time:"), 0, 2)
        self.estimated_time_label = QLabel("2h 5m")
        layout.addWidget(self.estimated_time_label, 0, 3)

        group.setLayout(layout)
        return group

    def _create_control_buttons(self) -> QHBoxLayout:
        """Create control buttons."""
        layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Scan")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
        )
        self.start_btn.clicked.connect(self.start_scan_requested.emit)
        layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.pause_scan_requested.emit)
        layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; }"
        )
        self.stop_btn.clicked.connect(self.stop_scan_requested.emit)
        layout.addWidget(self.stop_btn)

        return layout

    def connect_signals(self) -> None:
        """Connect internal signals."""
        # Update calculations when parameters change
        for widget in [
            self.x_start_spin, self.x_end_spin,
            self.y_start_spin, self.y_end_spin,
            self.x_pixels_spin, self.y_pixels_spin,
            self.dwell_time_spin, self.scan_speed_spin
        ]:
            widget.valueChanged.connect(self.update_calculations)

        self.mode_combo.currentTextChanged.connect(self.update_mode_controls)
        self.bidirectional_cb.stateChanged.connect(self.update_calculations)

    def load_defaults(self) -> None:
        """Load default values from config."""
        self.x_pixels_spin.setValue(self.config.default_x_pixels)
        self.y_pixels_spin.setValue(self.config.default_y_pixels)
        self.dwell_time_spin.setValue(self.config.default_dwell_time)
        self.scan_speed_spin.setValue(self.config.default_scan_speed)
        self.update_calculations()

    def update_calculations(self) -> None:
        """Update step size and time estimates."""
        params = self.get_scan_parameters()

        # Update step sizes
        self.x_step_label.setText(f"{params.x_step:.1f}")
        self.y_step_label.setText(f"{params.y_step:.1f}")

        # Update total pixels
        self.total_pixels_label.setText(f"{params.total_pixels:,}")

        # Update time estimate
        total_time = params.estimated_time()
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
        self.scan_parameters_changed.emit(params)

    def update_mode_controls(self) -> None:
        """Update controls based on scan mode."""
        mode = self.mode_combo.currentData()
        is_step_stop = (mode == ScanMode.STEP_STOP)

        self.dwell_time_spin.setEnabled(is_step_stop)
        self.scan_speed_spin.setEnabled(not is_step_stop)

        if mode == ScanMode.CONTINUOUS:
            self.mode_notice_label.setText(
                "⚠️ Continuous mode is under construction. "
                "The scan will run in step-and-stop mode."
            )
        else:
            self.mode_notice_label.setText("")

    def get_scan_parameters(self) -> ScanParameters:
        """Get current scan parameters."""
        return ScanParameters(
            x_start=self.x_start_spin.value(),
            x_end=self.x_end_spin.value(),
            y_start=self.y_start_spin.value(),
            y_end=self.y_end_spin.value(),
            x_pixels=self.x_pixels_spin.value(),
            y_pixels=self.y_pixels_spin.value(),
            mode=self.mode_combo.currentData(),
            dwell_time=self.dwell_time_spin.value(),
            scan_speed=self.scan_speed_spin.value(),
            bidirectional=self.bidirectional_cb.isChecked()
        )

    def preview_scan_path(self) -> None:
        """Preview the scan path."""
        from controllers.movement_patterns import RasterPattern

        params = self.get_scan_parameters()
        pattern_gen = RasterPattern(params.bidirectional)
        pattern = pattern_gen.generate(params)

        # Extract just x, y coordinates
        path = [(x, y) for x, y, _, _ in pattern]
        self.scan_path_preview.emit(path)

    def set_scan_state(self, state: ScanState) -> None:
        """Update UI based on scan state."""
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
    def update_progress(self, current: int, total: int) -> None:
        """Update scan progress."""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
            self.progress_bar.setFormat(f"{current}/{total} ({progress}%)")
