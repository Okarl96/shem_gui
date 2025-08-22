"""UI widgets module."""
from __future__ import annotations

from .connection_panel import ConnectionPanel
from .image_display import ImageDisplayWidget
from .log_dock import LogDockWidget
from .scan_control import ScanControlWidget
from .status_dock import StatusDockWidget

__all__ = [
    "ConnectionPanel",
    "ImageDisplayWidget",
    "LogDockWidget",
    "ScanControlWidget",
    "StatusDockWidget"
]
