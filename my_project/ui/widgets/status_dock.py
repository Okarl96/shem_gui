"""Status dock widget"""

from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import pyqtSlot


class StatusDockWidget(QDockWidget):
    """System status dock widget"""
    
    def __init__(self):
        super().__init__("System Status")
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("System Statistics:"))
        
        self.stats_table = QTableWidget(0, 2)
        self.stats_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.setMaximumHeight(200)
        
        layout.addWidget(self.stats_table)
        
        widget.setLayout(layout)
        self.setWidget(widget)
        
    @pyqtSlot(dict)
    def update_statistics(self, stats: dict):
        """Update statistics display"""
        self.stats_table.setRowCount(0)
        
        display_stats = [
            ("Runtime", f"{stats.get('runtime', 0):.1f} s"),
            ("Position Messages", f"{stats.get('position_messages', 0):,}"),
            ("Current Messages", f"{stats.get('current_messages', 0):,}"),
            ("Data Points Created", f"{stats.get('data_points_created', 0):,}"),
            ("Position Rate", f"{stats.get('position_rate', 0):.1f} Hz"),
            ("Current Rate", f"{stats.get('current_rate', 0):.1f} Hz"),
            ("DataPoint Rate", f"{stats.get('datapoint_rate', 0):.1f} Hz"),
        ]
        
        for i, (param, value) in enumerate(display_stats):
            self.stats_table.insertRow(i)
            self.stats_table.setItem(i, 0, QTableWidgetItem(param))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(value)))
