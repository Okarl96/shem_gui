"""Status dock widget."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSlot  # type: ignore library
from PyQt5.QtWidgets import (
    QDockWidget,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class StatusDockWidget(QDockWidget):
    """System status dock widget."""

    def __init__(self) -> None:
        super().__init__("System Status")
        self.setup_ui()

    def setup_ui(self) -> None:
        """Sets up the UI."""  # noqa: D401
        widget = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("System Statistics:"))

        self.stats_table = QTableWidget(0, 2)
        self.stats_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)  # type: ignore library
        self.stats_table.setMaximumHeight(200)

        layout.addWidget(self.stats_table)

        widget.setLayout(layout)
        self.setWidget(widget)

    @pyqtSlot(dict)
    def update_statistics(self, stats: dict[str, int]) -> None:
        """Update statistics display."""
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
