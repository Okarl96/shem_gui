"""MQTT connection panel widget."""
from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
)


class ConnectionPanel(QGroupBox):
    """MQTT connection panel."""

    connect_requested = pyqtSignal()
    disconnect_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__("MQTT Connection")
        self.setup_ui()
        self._connected = False

    def setup_ui(self) -> None:
        """Setup the UI."""
        layout = QGridLayout()

        layout.addWidget(QLabel("Broker:"), 0, 0)
        self.broker_edit = QLineEdit("localhost")
        layout.addWidget(self.broker_edit, 0, 1)

        layout.addWidget(QLabel("Port:"), 1, 0)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(1883)
        layout.addWidget(self.port_spin, 1, 1)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        layout.addWidget(self.connect_btn, 2, 0, 1, 2)

        self.setLayout(layout)

    def _on_connect_clicked(self) -> None:
        """Handle connect button click."""
        if self._connected:
            self.disconnect_requested.emit()
        else:
            self.connect_requested.emit()

    def set_connected(self, connected: bool) -> None:
        """Update connection state."""
        self._connected = connected
        if connected:
            self.connect_btn.setText("Disconnect")
            self.broker_edit.setEnabled(False)
            self.port_spin.setEnabled(False)
        else:
            self.connect_btn.setText("Connect")
            self.broker_edit.setEnabled(True)
            self.port_spin.setEnabled(True)

    def get_host(self) -> str:
        """Get broker host."""
        return self.broker_edit.text()

    def get_port(self) -> int:
        """Get broker port."""
        return self.port_spin.value()
