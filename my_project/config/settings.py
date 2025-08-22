"""Application configuration and settings management"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any
from PyQt5.QtCore import QSettings


@dataclass
class AppConfig:
    """Application configuration"""
    
    # MQTT Settings
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_keepalive: int = 60
    
    # Data Settings
    data_path: str = field(default_factory=lambda: str(Path.home() / "ScanningMicroscope" / "data"))
    data_batch_size: int = 1000
    
    # Timing Settings (ms)
    image_update_interval: int = 1000
    stats_update_interval: int = 5000
    
    # Scan Defaults
    default_dwell_time: float = 0.75  # seconds
    default_scan_speed: float = 1000.0  # nm/s
    default_x_pixels: int = 100
    default_y_pixels: int = 100
    
    # UI Settings
    theme: str = "fusion_light"
    window_geometry: Dict[str, int] = field(default_factory=lambda: {
        "x": 100, "y": 100, "width": 1800, "height": 1200
    })
    
    # Processing Settings
    position_freshness_window: float = 0.200  # seconds
    min_samples_per_pixel: int = 1
    
    @classmethod
    def load(cls) -> 'AppConfig':
        """Load configuration from QSettings with directory creation"""
        settings = QSettings('ResearchLab', 'ScanningMicroscope')
        config = cls()
        
        # Load MQTT settings
        config.mqtt_host = settings.value('mqtt/host', config.mqtt_host)
        config.mqtt_port = int(settings.value('mqtt/port', config.mqtt_port))
        
        # Load data settings
        config.data_path = settings.value('data/path', config.data_path)
        config.data_batch_size = int(settings.value('data/batch_size', config.data_batch_size))
        
        # Load timing settings
        config.image_update_interval = int(settings.value('timing/image_update', config.image_update_interval))
        config.stats_update_interval = int(settings.value('timing/stats_update', config.stats_update_interval))
        
        # Load scan defaults
        config.default_dwell_time = float(settings.value('scan/default_dwell_time', config.default_dwell_time))
        config.default_scan_speed = float(settings.value('scan/default_scan_speed', config.default_scan_speed))
        
        # Load UI settings
        config.theme = settings.value('ui/theme', config.theme)
        
        # Ensure all required directories exist
        directories_to_create = [
            config.data_path,
            os.path.join(os.path.dirname(__file__), '..', 'resources', 'icons'),
            os.path.join(os.path.dirname(__file__), '..', 'resources', 'qss')
        ]
        
        for directory in directories_to_create:
            os.makedirs(directory, exist_ok=True)
        
        return config
    
    def save(self):
        """Save configuration to QSettings"""
        settings = QSettings('ResearchLab', 'ScanningMicroscope')
        
        # Save MQTT settings
        settings.setValue('mqtt/host', self.mqtt_host)
        settings.setValue('mqtt/port', self.mqtt_port)
        
        # Save data settings
        settings.setValue('data/path', self.data_path)
        settings.setValue('data/batch_size', self.data_batch_size)
        
        # Save timing settings
        settings.setValue('timing/image_update', self.image_update_interval)
        settings.setValue('timing/stats_update', self.stats_update_interval)
        
        # Save scan defaults
        settings.setValue('scan/default_dwell_time', self.default_dwell_time)
        settings.setValue('scan/default_scan_speed', self.default_scan_speed)
        
        # Save UI settings
        settings.setValue('ui/theme', self.theme)
