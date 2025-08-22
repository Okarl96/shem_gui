from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QSpinBox, QCheckBox
)
from PyQt5.QtCore import pyqtSlot, QTimer
import pyqtgraph as pg
import numpy as np
from collections import deque
import time


class SignalTimePlugin:
    """Real-time current vs time plotting"""
    
    @property
    def name(self) -> str:
        return "Signal vs Time"
    
    @property
    def icon(self) -> str:
        return ""  # Optional icon path
    
    def __init__(self):
        self.widget = None
        self.plot_widget = None
        self.curve = None
        
        # Data buffers
        self.time_buffer = deque(maxlen=1000)  # Keep last 1000 points
        self.current_buffer = deque(maxlen=1000)
        self.position_x_buffer = deque(maxlen=1000)
        self.position_y_buffer = deque(maxlen=1000)
        
        # Start time reference
        self.start_time = None
        
        # Update timer
        self.update_timer = None
        
    def create_widget(self, data_source) -> QWidget:
        """Create the signal vs time widget"""
        self.widget = QWidget()
        layout = QVBoxLayout()
        
        # === Controls ===
        controls = QHBoxLayout()
        
        # Buffer size control
        controls.addWidget(QLabel("Buffer Size:"))
        self.buffer_spin = QSpinBox()
        self.buffer_spin.setRange(100, 10000)
        self.buffer_spin.setValue(1000)
        self.buffer_spin.setSuffix(" points")
        self.buffer_spin.valueChanged.connect(self.update_buffer_size)
        controls.addWidget(self.buffer_spin)
        
        # Auto-scroll checkbox
        self.auto_scroll_cb = QCheckBox("Auto Scroll")
        self.auto_scroll_cb.setChecked(True)
        controls.addWidget(self.auto_scroll_cb)
        
        # Clear button
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_data)
        controls.addWidget(self.clear_btn)
        
        # Pause button
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        controls.addWidget(self.pause_btn)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # === Plot Widget ===
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Current', units='nA')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setTitle('Current vs Time')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Create plot curves
        self.curve = self.plot_widget.plot(
            pen=pg.mkPen(color='y', width=2),
            name='Current'
        )
        
        # Add legend
        self.plot_widget.addLegend()
        
        layout.addWidget(self.plot_widget)
        
        # === Statistics Display ===
        stats_layout = QHBoxLayout()
        
        self.stats_label = QLabel("No data")
        self.stats_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; }"
        )
        stats_layout.addWidget(self.stats_label)
        
        layout.addLayout(stats_layout)
        
        self.widget.setLayout(layout)
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        self.update_timer.start(100)  # Update every 100ms
        
        return self.widget
    
    def on_data_update(self, data_point):
        """Handle new data points"""
        if self.pause_btn and self.pause_btn.isChecked():
            return  # Skip if paused
            
        # Initialize start time on first data point
        if self.start_time is None:
            self.start_time = data_point.timestamp
        
        # Calculate relative time
        relative_time = data_point.timestamp - self.start_time
        
        # Add to buffers
        self.time_buffer.append(relative_time)
        self.current_buffer.append(data_point.current)
        
        if data_point.x_pos is not None:
            self.position_x_buffer.append(data_point.x_pos)
        if data_point.y_pos is not None:
            self.position_y_buffer.append(data_point.y_pos)
        
        # Update statistics
        self.update_statistics()
    
    def update_plot(self):
        """Update the plot display"""
        if not self.time_buffer or not self.current_buffer:
            return
            
        # Convert to arrays
        times = np.array(self.time_buffer)
        currents = np.array(self.current_buffer)
        
        # Update curve
        self.curve.setData(times, currents)
        
        # Auto-scroll if enabled
        if self.auto_scroll_cb.isChecked() and len(times) > 0:
            # Show last N seconds
            window_size = 10.0  # seconds
            x_max = times[-1]
            x_min = max(0, x_max - window_size)
            self.plot_widget.setXRange(x_min, x_max, padding=0)
    
    def update_statistics(self):
        """Update statistics display"""
        if not self.current_buffer:
            self.stats_label.setText("No data")
            return
            
        currents = np.array(self.current_buffer)
        
        stats_text = (
            f"Points: {len(self.current_buffer)} | "
            f"Current - Mean: {np.mean(currents):.4f} nA, "
            f"Std: {np.std(currents):.4f} nA, "
            f"Min: {np.min(currents):.4f} nA, "
            f"Max: {np.max(currents):.4f} nA"
        )
        
        self.stats_label.setText(stats_text)
    
    def update_buffer_size(self, size):
        """Update buffer size"""
        # Create new buffers with new size
        new_time = deque(self.time_buffer, maxlen=size)
        new_current = deque(self.current_buffer, maxlen=size)
        new_x = deque(self.position_x_buffer, maxlen=size)
        new_y = deque(self.position_y_buffer, maxlen=size)
        
        self.time_buffer = new_time
        self.current_buffer = new_current
        self.position_x_buffer = new_x
        self.position_y_buffer = new_y
    
    def clear_data(self):
        """Clear all data buffers"""
        self.time_buffer.clear()
        self.current_buffer.clear()
        self.position_x_buffer.clear()
        self.position_y_buffer.clear()
        self.start_time = None
        self.curve.setData([], [])
        self.stats_label.setText("No data")
    
    def on_scan_started(self):
        """Called when scan starts"""
        # Optionally clear previous data
        if hasattr(self, 'auto_clear_cb') and self.auto_clear_cb.isChecked():
            self.clear_data()
    
    def on_scan_completed(self):
        """Called when scan completes"""
        # Could add a marker or save data
        pass
