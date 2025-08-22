"""Plugin interface definition"""

from typing import Protocol
from PyQt5.QtWidgets import QWidget


class ITabPlugin(Protocol):
    """Interface for tab plugins"""
    
    @property
    def name(self) -> str:
        """Display name for the tab"""
        ...
        
    @property
    def icon(self) -> str:
        """Path to icon file (optional)"""
        ...
        
    def create_widget(self, data_source) -> QWidget:
        """Create the plugin's widget
        
        Args:
            data_source: Reference to data processor or reconstructor
            
        Returns:
            QWidget to be added as a tab
        """
        ...
        
    def on_data_update(self, data_point):
        """Handle new data points
        
        Args:
            data_point: DataPoint object
        """
        ...
        
    def on_scan_started(self):
        """Called when scan starts"""
        ...
        
    def on_scan_completed(self):
        """Called when scan completes"""
        ...
