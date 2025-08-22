from __future__ import annotations

from datetime import datetime

from core.models import ScanState
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDockWidget,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .widgets.connection_panel import ConnectionPanel
from .widgets.image_display import ImageDisplayWidget
from .widgets.log_dock import LogDockWidget
from .widgets.scan_control import ScanControlWidget
from .widgets.status_dock import StatusDockWidget


class MainWindow(QMainWindow):
    """Main application window with plugin support."""

    def __init__(self, container) -> None:
        super().__init__()

        # Store dependency container
        self.container = container

        # Current scan state
        self.current_scan_id = None
        self.current_hdf5_file = None
        self.data_batch = []

        # Plugin storage
        self.plugins = {}
        self.plugin_instances = []

        # Setup timers
        self.setup_timers()

        # Setup UI
        self.setup_ui()

        # Setup plugins AFTER UI is created
        self.setup_plugins()

        # Connect signals
        self.connect_signals()

    def setup_ui(self) -> None:
        """Setup the main UI."""
        self.setWindowTitle("Scanning Microscope System v1.0")
        self.setGeometry(100, 100, 1800, 1200)

        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - Image display
        self.image_widget = ImageDisplayWidget()
        splitter.addWidget(self.image_widget)

        # Set splitter sizes (30% control, 70% image)
        splitter.setSizes([540, 1260])

        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)

        # Create docked windows
        self.create_docks()

        # Status bar
        self.setup_status_bar()

    def setup_plugins(self) -> None:
        """Setup plugin system."""
        import importlib
        import sys

        from PyQt5.QtCore import Qt

        # Create plugin dock
        self.plugin_dock = QDockWidget("Analysis Plugins")
        self.plugin_tabs = QTabWidget()

        # Get the plugins/builtin directory path
        import pathlib
        app_dir = pathlib.Path(__file__).parent.parent
        builtin_plugin_dir = app_dir / "plugins" / "builtin"

        # Make sure the directory exists
        builtin_plugin_dir.mkdir(parents=True, exist_ok=True)

        # Add plugins directory to Python path if not already there
        plugins_dir = str(app_dir / "plugins")
        if plugins_dir not in sys.path:
            sys.path.insert(0, plugins_dir)

        # List of built-in plugins to load
        builtin_plugins = [
            "signal_vs_time",

        ]

        # Load each plugin
        for plugin_name in builtin_plugins:
            try:
                # Check if plugin file exists
                plugin_file = builtin_plugin_dir / f"{plugin_name}.py"
                if not plugin_file.exists():
                    print(f"Plugin file not found: {plugin_file}")
                    continue

                # Import the plugin module
                module = importlib.import_module(f"builtin.{plugin_name}")

                # Find the plugin class
                plugin_class = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and
                        hasattr(attr, "create_widget") and
                        hasattr(attr, "name")):
                        plugin_class = attr
                        break

                if plugin_class:
                    # Create plugin instance
                    plugin = plugin_class()
                    self.plugin_instances.append(plugin)

                    # Create widget
                    widget = plugin.create_widget(self.container.image_reconstructor)

                    # Add to tabs
                    tab_name = plugin.name
                    self.plugin_tabs.addTab(widget, tab_name)

                    # Connect signals
                    if hasattr(plugin, "on_data_update"):
                        self.container.data_processor.new_data_point.connect(
                            plugin.on_data_update
                        )

                    if hasattr(plugin, "on_scan_started"):
                        self.container.scan_controller.scan_started.connect(
                            plugin.on_scan_started
                        )

                    if hasattr(plugin, "on_scan_completed"):
                        self.container.scan_controller.scan_completed.connect(
                            plugin.on_scan_completed
                        )

                    print(f"✓ Loaded plugin: {tab_name}")

            except ImportError as e:
                print(f"Failed to import plugin {plugin_name}: {e}")
            except Exception as e:
                print(f"Failed to load plugin {plugin_name}: {e}")

        # Only add the dock if we have plugins
        if self.plugin_tabs.count() > 0:
            self.plugin_dock.setWidget(self.plugin_tabs)
            self.addDockWidget(Qt.RightDockWidgetArea, self.plugin_dock)
            print(f"Added plugin dock with {self.plugin_tabs.count()} plugins")
        else:
            print("No plugins loaded, plugin dock not added")

    # ... rest of the MainWindow implementation remains the same ...
    def create_left_panel(self) -> QWidget:
        """Create left control panel."""
        panel = QWidget()
        layout = QVBoxLayout()

        # Connection panel
        self.connection_panel = ConnectionPanel()
        layout.addWidget(self.connection_panel)

        # Scan control
        self.scan_control = ScanControlWidget(self.container.config)
        layout.addWidget(self.scan_control)

        # Current readings display
        self.current_readings_widget = self.create_current_readings_widget()
        layout.addWidget(self.current_readings_widget)

        layout.addStretch()
        panel.setLayout(layout)

        return panel

    def create_current_readings_widget(self) -> QGroupBox:
        """Create current readings display widget."""
        group = QGroupBox("Current Readings")
        layout = QGridLayout()

        # Current reading
        layout.addWidget(QLabel("Current:"), 0, 0)
        self.current_label = QLabel("N/A")
        self.current_label.setFont(QFont("Monospace", 10))
        self.current_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 3px; }")
        layout.addWidget(self.current_label, 0, 1)

        # X Position
        layout.addWidget(QLabel("X Position:"), 1, 0)
        self.x_pos_label = QLabel("N/A")
        self.x_pos_label.setFont(QFont("Monospace", 10))
        self.x_pos_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 3px; }")
        layout.addWidget(self.x_pos_label, 1, 1)

        # Y Position
        layout.addWidget(QLabel("Y Position:"), 2, 0)
        self.y_pos_label = QLabel("N/A")
        self.y_pos_label.setFont(QFont("Monospace", 10))
        self.y_pos_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 3px; }")
        layout.addWidget(self.y_pos_label, 2, 1)

        group.setLayout(layout)
        return group

    def create_docks(self) -> None:
        """Create dockable windows."""
        # Status dock
        self.status_dock = StatusDockWidget()
        self.addDockWidget(Qt.BottomDockWidgetArea, self.status_dock)

        # Log dock
        self.log_dock = LogDockWidget()
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)

    def setup_status_bar(self) -> None:
        """Setup status bar."""
        self.status_bar = self.statusBar()

        # Connection status
        self.connection_label = QLabel("Disconnected")
        self.connection_label.setStyleSheet("QLabel { color: red; padding: 3px; }")
        self.status_bar.addPermanentWidget(self.connection_label)

        # Scan status
        self.scan_status_label = QLabel("Idle")
        self.status_bar.addPermanentWidget(self.scan_status_label)

    def setup_timers(self) -> None:
        """Setup update timers."""
        # Statistics timer
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(
            self.container.data_processor.emit_statistics
        )
        self.stats_timer.start(self.container.config.stats_update_interval)

        # Image update timer
        self.image_timer = QTimer()
        self.image_timer.timeout.connect(
            self.container.image_reconstructor.emit_full_image
        )
        self.image_timer.start(self.container.config.image_update_interval)

    def connect_signals(self) -> None:
        """Connect all signals."""
        # Connection panel - FIX: Add disconnect signal
        self.connection_panel.connect_requested.connect(self.connect_mqtt)
        self.connection_panel.disconnect_requested.connect(self.disconnect_mqtt)

        # MQTT signals
        mqtt = self.container.mqtt_client
        mqtt.connected.connect(self.on_mqtt_connected)
        mqtt.position_data_received.connect(
            self.container.data_processor.process_position_data
        )
        mqtt.current_data_received.connect(
            self.container.data_processor.process_current_data
        )
        mqtt.status_update.connect(self.log_dock.add_message)
        mqtt.error_occurred.connect(self.log_dock.add_error)

        # Data processor signals
        proc = self.container.data_processor
        proc.new_data_point.connect(self.on_new_data_point)
        proc.statistics_update.connect(self.status_dock.update_statistics)

        # Image reconstructor signals
        recon = self.container.image_reconstructor
        recon.image_updated.connect(self.image_widget.update_image)
        # Connect processor's signal to reconstructor's slot
        proc.new_data_point.connect(recon.add_data_point)

        # Scan control signals
        self.scan_control.scan_parameters_changed.connect(self.on_scan_parameters_changed)
        self.scan_control.start_scan_requested.connect(self.start_scan)
        self.scan_control.pause_scan_requested.connect(self.container.scan_controller.pause_scan)
        self.scan_control.stop_scan_requested.connect(self.container.scan_controller.stop_scan)

        # Connect scan path preview
        self.scan_control.scan_path_preview.connect(self.image_widget.show_scan_preview)

        # Scan controller signals
        ctrl = self.container.scan_controller
        ctrl.movement_command.connect(mqtt.send_command)
        ctrl.scan_started.connect(self.on_scan_started)
        ctrl.scan_completed.connect(self.on_scan_completed)
        ctrl.scan_progress.connect(self.scan_control.update_progress)
        ctrl.status_update.connect(self.log_dock.add_message)

    @pyqtSlot()
    def connect_mqtt(self) -> None:
        """Connect to MQTT broker."""
        host = self.connection_panel.get_host()
        port = self.connection_panel.get_port()

        self.container.mqtt_client.setup_mqtt(host, port)
        self.container.mqtt_client.connect_mqtt()

    @pyqtSlot()
    def disconnect_mqtt(self) -> None:
        """Disconnect from MQTT broker."""
        self.container.mqtt_client.disconnect_mqtt()
        self.log_dock.add_message("Disconnecting from MQTT broker...")

    @pyqtSlot(bool)
    def on_mqtt_connected(self, connected: bool) -> None:
        """Handle MQTT connection status."""
        if connected:
            self.connection_label.setText("Connected")
            self.connection_label.setStyleSheet("QLabel { color: green; padding: 3px; }")
            self.connection_panel.set_connected(True)
        else:
            self.connection_label.setText("Disconnected")
            self.connection_label.setStyleSheet("QLabel { color: red; padding: 3px; }")
            self.connection_panel.set_connected(False)

    @pyqtSlot(object)
    def on_scan_parameters_changed(self, scan_params) -> None:
        """Handle scan parameter changes."""
        self.container.scan_controller.set_scan_parameters(scan_params)
        self.container.image_reconstructor.initialize_scan(scan_params)
        self.image_widget.set_scan_parameters(scan_params)

    @pyqtSlot()
    def on_scan_started(self) -> None:
        """Handle scan started."""
        self.scan_control.set_scan_state(ScanState.SCANNING)
        self.scan_status_label.setText("Scanning")
        self.scan_status_label.setStyleSheet("QLabel { color: green; padding: 3px; }")
        self.log_dock.add_message("Scan started")

    @pyqtSlot()
    def start_scan(self) -> None:
        """Start a new scan with proper verification."""
        # Check MQTT connection
        if not self.container.mqtt_client.connected_status:
            QMessageBox.warning(self, "Warning", "MQTT not connected")
            return

        # Verify MQTT subscriptions
        all_subscribed, missing = self.container.mqtt_client.verify_subscriptions()
        if not all_subscribed:
            QMessageBox.warning(
                self,
                "Warning",
                f"Missing MQTT subscriptions: {', '.join(missing)}\n"
                "Please reconnect to the broker."
            )
            return

        # Get and validate scan parameters
        scan_params = self.container.scan_controller.scan_params
        if not scan_params:
            QMessageBox.warning(self, "Warning", "No scan parameters set")
            return

        # Validate scan parameters
        from utils.validators import InputValidator
        is_valid, error_msg = InputValidator.validate_scan_parameters(scan_params)
        if not is_valid:
            QMessageBox.warning(self, "Invalid Parameters", error_msg)
            return

        # Create scan record
        scan_name = f"Scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            self.current_scan_id = self.container.meta_storage.create_scan_record(
                scan_params, scan_name
            )

            # Create HDF5 file
            self.current_hdf5_file = self.container.raw_storage.create_file(
                self.current_scan_id, scan_params
            )

            # Update metadata with file path
            self.container.meta_storage.update_file_path(
                self.current_scan_id, self.current_hdf5_file
            )

            # Clear data batch
            self.data_batch.clear()

            # Start scan
            self.container.scan_controller.start_scan()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start scan: {e}")
            self.log_dock.add_error(f"Scan start failed: {e}")

    @pyqtSlot(object)
    def on_new_data_point(self, data_point) -> None:
        """Handle new data point."""
        # UPDATE: Display current readings
        if data_point.current is not None:
            self.current_label.setText(f"{data_point.current:.4f} nA")

        if data_point.x_pos is not None:
            self.x_pos_label.setText(f"{data_point.x_pos:.0f} nm")

        if data_point.y_pos is not None:
            self.y_pos_label.setText(f"{data_point.y_pos:.0f} nm")

        # Add to batch
        self.data_batch.append(data_point)

        # Save batch if full
        if len(self.data_batch) >= self.container.config.data_batch_size:
            self.save_data_batch()

    def save_data_batch(self) -> None:
        """Save current data batch with error handling."""
        if not self.data_batch or not self.current_hdf5_file:
            return

        # Create a copy of the batch to avoid data loss
        batch_copy = self.data_batch.copy()

        try:
            self.container.raw_storage.save_batch(batch_copy)
            # Only clear if save was successful
            self.data_batch.clear()
        except Exception as e:
            # Keep the data and log error
            self.log_dock.add_error(f"Failed to save data batch: {e}")
            # Optionally, try to save to a backup location
            try:
                backup_file = f"backup_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
                np.save(backup_file, [dp.to_dict() for dp in batch_copy])
                self.log_dock.add_message(f"Batch saved to backup: {backup_file}")
            except:
                pass  # Last resort failed

    @pyqtSlot()
    def on_scan_completed(self) -> None:
        """Handle scan completion."""
        # Save remaining data
        self.save_data_batch()

        # Save final image
        image = self.container.image_reconstructor.get_current_image()
        if image is not None:
            self.container.raw_storage.save_image(image)

        # Mark scan as completed
        if self.current_scan_id:
            self.container.meta_storage.complete_scan(self.current_scan_id)

        # Close HDF5 file
        self.container.raw_storage.close()

        # Update UI
        self.scan_control.set_scan_state(ScanState.IDLE)
        self.scan_status_label.setText("Idle")
        self.scan_status_label.setStyleSheet("QLabel { padding: 3px; }")
        self.log_dock.add_message("Scan completed")

    def closeEvent(self, event) -> None:
        """Handle application close."""
        # Stop any running scan
        if self.container.scan_controller.scan_state != ScanState.IDLE:
            self.container.scan_controller.stop_scan()

        # Save remaining data
        self.save_data_batch()

        # Stop timers
        self.stats_timer.stop()
        self.image_timer.stop()

        # Disconnect MQTT
        self.container.mqtt_client.disconnect_mqtt()

        event.accept()
