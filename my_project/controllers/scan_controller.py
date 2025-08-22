"""Scan execution controller"""

from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot
from typing import List, Tuple, Optional

from core.models import ScanParameters, ScanMode, ScanState
from .command_builder import CommandBuilder
from .movement_patterns import RasterPattern


class ScanController(QObject):
    """Controls scan execution and movement patterns"""
    
    # Signals
    scan_started = pyqtSignal()
    scan_completed = pyqtSignal()
    scan_progress = pyqtSignal(int, int)  # current_pixel, total_pixels
    movement_command = pyqtSignal(str)    # MQTT command to send
    status_update = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.scan_params: Optional[ScanParameters] = None
        self.scan_state = ScanState.IDLE
        self.current_pixel = 0
        self.scan_pattern: List[Tuple[float, float, int, int]] = []
        
        # Command builder
        self.command_builder = CommandBuilder()
        
        # Movement pattern generator
        self.pattern_generator = RasterPattern()
        
        # Timers
        self.movement_timer = QTimer()
        self.movement_timer.timeout.connect(self.execute_next_movement)
        
        self.step_timer = QTimer()
        self.step_timer.setSingleShot(True)
        self.step_timer.timeout.connect(self.on_dwell_completed)
        
    def set_scan_parameters(self, scan_params: ScanParameters):
        """Set scan parameters"""
        self.scan_params = scan_params
        self.pattern_generator.bidirectional = scan_params.bidirectional
        self.generate_scan_pattern()
        
    def generate_scan_pattern(self):
        """Generate scan pattern"""
        if not self.scan_params:
            return
            
        self.scan_pattern = self.pattern_generator.generate(self.scan_params)
        
    def get_scan_preview_path(self) -> List[Tuple[float, float]]:
        """Get scan path for preview visualization"""
        if not self.scan_params:
            return []
            
        return [(x, y) for x, y, _, _ in self.scan_pattern]
                
    @pyqtSlot()
    def start_scan(self):
        """Start the scan"""
        if self.scan_state != ScanState.IDLE or not self.scan_params:
            return
            
        self.scan_state = ScanState.SCANNING
        self.current_pixel = 0
        self.scan_started.emit()
        
        if self.scan_params.mode == ScanMode.STEP_STOP:
            self.start_step_stop_scan()
        else:
            self.start_continuous_scan()
            
    def start_step_stop_scan(self):
        """Start step-and-stop scan"""
        self.status_update.emit("Starting step-and-stop scan")
        self.execute_next_movement()
        
    def start_continuous_scan(self):
        """Start continuous scan"""
        # Currently falls back to step-and-stop
        self.status_update.emit("⚠️ Continuous mode is under construction - using step-and-stop mode")
        self.start_step_stop_scan()
        
    @pyqtSlot()
    def execute_next_movement(self):
        """Execute next movement in scan pattern"""
        if (self.scan_state != ScanState.SCANNING or 
            self.current_pixel >= len(self.scan_pattern)):
            self.complete_scan()
            return
            
        x_pos, y_pos, x_idx, y_idx = self.scan_pattern[self.current_pixel]
        
        # Send movement commands
        self.movement_command.emit(self.command_builder.move_x(x_pos))
        self.movement_command.emit(self.command_builder.move_y(y_pos))
        
        self.scan_progress.emit(self.current_pixel + 1, len(self.scan_pattern))
        
        if self.scan_params.mode == ScanMode.STEP_STOP:
            # Wait for dwell time
            dwell_ms = int(self.scan_params.dwell_time * 1000)
            self.step_timer.start(dwell_ms)
        else:
            # Continuous mode - move to next immediately
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
            self.status_update.emit("Scan paused")
            
    @pyqtSlot()
    def resume_scan(self):
        """Resume the scan"""
        if self.scan_state == ScanState.PAUSED:
            self.scan_state = ScanState.SCANNING
            if self.scan_params and self.scan_params.mode == ScanMode.STEP_STOP:
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
        
        # Send stop commands
        self.movement_command.emit(self.command_builder.stop_x())
        self.movement_command.emit(self.command_builder.stop_y())
        
        self.complete_scan()
        
    def complete_scan(self):
        """Complete the scan"""
        self.scan_state = ScanState.IDLE
        self.movement_timer.stop()
        self.step_timer.stop()
        self.scan_completed.emit()
        self.status_update.emit("Scan completed")
