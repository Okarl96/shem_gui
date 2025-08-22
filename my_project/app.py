#!/usr/bin/env python3
"""Scanning Microscope System - Application Entry Point."""
from __future__ import annotations

import sys
from pathlib import Path

# Set High DPI attributes BEFORE any Qt imports
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

# Set these as early as possible
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

# Now import the rest
sys.path.insert(0, str(Path(__file__).parent))

from comms.mqtt_client import MQTTController
from config.settings import AppConfig
from controllers.scan_controller import ScanController
from processing.data_processor import DataProcessor
from processing.reconstructor import ImageReconstructor
from storage.hdf5_raw import HDF5RawStorage
from storage.sqlite_meta import SQLiteMetaStorage
from ui.main_window import MainWindow
from ui.styles import StyleManager


class DependencyContainer:
    """Dependency injection container."""

    def __init__(self) -> None:
        # Load configuration
        self.config = AppConfig.load()

        # Create core components
        self.mqtt_client = MQTTController()
        self.data_processor = DataProcessor()
        self.image_reconstructor = ImageReconstructor()
        self.scan_controller = ScanController()

        # Create storage components
        self.meta_storage = SQLiteMetaStorage(self.config.data_path)
        self.raw_storage = HDF5RawStorage(self.config.data_path)

        # Wire dependencies
        self.data_processor.set_storage(self.meta_storage)

    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self.meta_storage, "close"):
            self.meta_storage.close()
        if hasattr(self.raw_storage, "close"):
            self.raw_storage.close()


class ScanningMicroscopeApp(QApplication):
    """Main application class."""

    def __init__(self, argv) -> None:
        super().__init__(argv)

        # Set application properties
        self.setApplicationName("Scanning Microscope")
        self.setApplicationVersion("1.0.0")
        self.setOrganizationName("Research Lab")

        # Apply style
        self.style_manager = StyleManager()
        self.style_manager.apply_theme(self)


def main() -> None:
    """Main entry point."""
    # Create application (attributes already set at module level)
    app = ScanningMicroscopeApp(sys.argv)

    # Create dependency container
    container = DependencyContainer()

    # Create and show main window
    window = MainWindow(container)
    window.show()

    # Run application
    try:
        result = app.exec_()
    finally:
        container.cleanup()

    sys.exit(result)


if __name__ == "__main__":
    main()
