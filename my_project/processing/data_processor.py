"""Data processor for synchronizing position and current data"""

import time
from PyQt5.QtCore import QObject, QMutex, QMutexLocker, pyqtSignal, pyqtSlot
from typing import Dict, Optional, Any

from core.models import DataPoint
from core.interfaces import IStorage


class DataProcessor(QObject):
    """Process and synchronize high-frequency data"""
    
    # Signals
    new_data_point = pyqtSignal(object)  # DataPoint object
    statistics_update = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        
        # Storage reference
        self.data_storage: Optional[IStorage] = None
        
        # Current state
        self.latest_positions = {"X": None, "Y": None, "Z": None, "R": None}
        self.latest_current = None
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
        
        # Configuration
        self.freshness_window = 0.200  # seconds
        
    def set_storage(self, data_storage: IStorage):
        """Set reference to data storage"""
        self.data_storage = data_storage
        
    def set_freshness_window(self, window: float):
        """Set position freshness window in seconds"""
        self.freshness_window = window
        
    @pyqtSlot(str)
    def process_position_data(self, payload: str):
        """Process ECC100 position data"""
        try:
            current_time = time.time()
            lines = payload.strip().split('\n')
            
            for line in lines:
                if not line:
                    continue
                    
                parts = line.split('/')
                if len(parts) == 5:
                    timestamp_str, x_str, y_str, z_str, r_str = parts
                    
                    # Parse device timestamp
                    device_time = self._parse_timestamp(timestamp_str, current_time)
                    
                    # Parse positions
                    positions = {}
                    for axis, val_str in zip(['X', 'Y', 'Z', 'R'], 
                                           [x_str, y_str, z_str, r_str]):
                        if val_str != "NaN":
                            positions[axis] = float(val_str)
                        else:
                            positions[axis] = None
                    
                    with QMutexLocker(self.mutex):
                        # Update positions and timestamps
                        for axis, val in positions.items():
                            self.latest_positions[axis] = val
                            self.latest_pos_time[axis] = device_time
                            
                        self.stats['position_messages'] += 1
                        self.stats['last_position_time'] = device_time
                        
        except Exception as e:
            print(f"Error processing position data: {e}")
            
    @pyqtSlot(str)
    def process_current_data(self, payload: str):
        """Process picoammeter current data"""
        try:
            current_time = time.time()
            parts = payload.strip().split('/')
            if len(parts) == 2:
                timestamp_str, current_str = parts
                current_value = float(current_str)
                
                # Parse device timestamp
                device_time = self._parse_timestamp(timestamp_str, current_time)
                
                with QMutexLocker(self.mutex):
                    self.latest_current = current_value
                    self.stats['current_messages'] += 1
                    self.stats['last_current_time'] = device_time
                    
                    # Check if positions are fresh
                    x = self.latest_positions.get('X')
                    y = self.latest_positions.get('Y')
                    tx = self.latest_pos_time.get('X')
                    ty = self.latest_pos_time.get('Y')
                    
                    if (x is not None and y is not None and
                        tx is not None and ty is not None and
                        abs(device_time - tx) < self.freshness_window and 
                        abs(device_time - ty) < self.freshness_window):
                        
                        data_point = DataPoint(
                            timestamp=device_time,
                            x_pos=x,
                            y_pos=y,
                            z_pos=self.latest_positions.get('Z'),
                            r_pos=self.latest_positions.get('R'),
                            current=current_value
                        )
                        self.new_data_point.emit(data_point)
                        self.stats['data_points_created'] += 1
                        
        except Exception as e:
            print(f"Error processing current data: {e}")
            
    @pyqtSlot()
    def emit_statistics(self):
        """Emit current statistics with proper thread safety"""
        with QMutexLocker(self.mutex):
            # Create a deep copy of stats to avoid race conditions
            stats_copy = {
                'position_messages': self.stats['position_messages'],
                'current_messages': self.stats['current_messages'],
                'data_points_created': self.stats['data_points_created'],
                'start_time': self.stats['start_time'],
                'last_position_time': self.stats['last_position_time'],
                'last_current_time': self.stats['last_current_time']
            }
            
        # Calculate rates outside the mutex lock
        runtime = time.time() - stats_copy['start_time']
        stats_copy['runtime'] = runtime
        
        if runtime > 0:
            stats_copy['position_rate'] = stats_copy['position_messages'] / runtime
            stats_copy['current_rate'] = stats_copy['current_messages'] / runtime
            stats_copy['datapoint_rate'] = stats_copy['data_points_created'] / runtime
        else:
            stats_copy['position_rate'] = 0
            stats_copy['current_rate'] = 0
            stats_copy['datapoint_rate'] = 0
            
        self.statistics_update.emit(stats_copy)
            
    def _parse_timestamp(self, timestamp_str: str, fallback_time: float) -> float:
        """Parse device timestamp with fallback"""
        try:
            ts = float(timestamp_str)
            # Convert to seconds based on magnitude
            if ts > 1e12:  # microseconds
                ts *= 1e-6
            elif ts > 1e10:  # milliseconds  
                ts *= 1e-3
            # Check if timestamp is reasonable
            if abs(ts - fallback_time) < 86400:  # within 1 day
                return ts
            else:
                return fallback_time
        except (ValueError, TypeError):
            return fallback_time
            
    def reset_statistics(self):
        """Reset statistics"""
        with QMutexLocker(self.mutex):
            self.stats = {
                'position_messages': 0,
                'current_messages': 0,
                'data_points_created': 0,
                'start_time': time.time(),
                'last_position_time': None,
                'last_current_time': None
            }
