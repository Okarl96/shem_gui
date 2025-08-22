"""Image display widget"""

import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox,
    QCheckBox, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSlot

import pyqtgraph as pg
import pyqtgraph.exporters

from core.models import ScanParameters
from storage.csv_export import CSVExporter


class ImageDisplayWidget(QWidget):
    """Widget for displaying the reconstructed image"""
    
    def __init__(self):
        super().__init__()
        
        self.current_image = None
        self.scan_params = None
        self.current_colormap = 'viridis'
        self._last_preview_path = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the image display UI"""
        layout = QVBoxLayout()
        
        # Image and colorbar
        image_layout = QHBoxLayout()
        
        # Main plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setLabel('left', 'Y Position', units='nm')
        self.plot_widget.setLabel('bottom', 'X Position', units='nm')
        
        # Image item
        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)
        
        # Scan path preview
        self.path_line = pg.PlotDataItem(
            pen=pg.mkPen(color='red', width=2, style=Qt.DashLine),
            name='Scan Path'
        )
        self.plot_widget.addItem(self.path_line)
        self.path_line.hide()
        
        # Start/end markers
        self.start_marker = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen('green', width=2),
            brush=pg.mkBrush('green'), name='Start'
        )
        self.end_marker = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen('red', width=2),
            brush=pg.mkBrush('red'), name='End'
        )
        self.plot_widget.addItem(self.start_marker)
        self.plot_widget.addItem(self.end_marker)
        self.start_marker.hide()
        self.end_marker.hide()
        
        image_layout.addWidget(self.plot_widget)
        
        # Colorbar
        self.colorbar_widget = self._create_colorbar()
        image_layout.addWidget(self.colorbar_widget)
        
        # Add image layout
        image_widget = QWidget()
        image_widget.setLayout(image_layout)
        layout.addWidget(image_widget)
        
        # Controls
        layout.addLayout(self._create_controls())
        
        self.setLayout(layout)
        
    def _create_colorbar(self) -> pg.PlotWidget:
        """Create colorbar widget"""
        colorbar = pg.PlotWidget()
        colorbar.setMaximumWidth(50)
        colorbar.setMinimumWidth(50)
        colorbar.hideAxis('bottom')
        colorbar.hideAxis('left')
        colorbar.setMouseEnabled(x=False, y=False)
        
        self.colorbar_image = pg.ImageItem()
        colorbar.addItem(self.colorbar_image)
        
        # Initialize gradient
        gradient = np.linspace(0, 1, 256).reshape(256, 1)
        self.colorbar_image.setImage(gradient)
        
        return colorbar
        
    def _create_controls(self) -> QHBoxLayout:
        """Create control buttons"""
        layout = QHBoxLayout()
        
        layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        # Use 'grey' instead of 'gray' for pyqtgraph
        self.colormap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'CET-L1', 'CET-L2'])
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        layout.addWidget(self.colormap_combo)
        
        self.auto_levels_btn = QPushButton("Auto Levels")
        self.auto_levels_btn.clicked.connect(self.auto_levels)
        layout.addWidget(self.auto_levels_btn)
        
        self.save_btn = QPushButton("Save Image")
        self.save_btn.clicked.connect(self.save_image)
        layout.addWidget(self.save_btn)
        
        self.export_btn = QPushButton("Export Data")
        self.export_btn.clicked.connect(self.export_data)
        layout.addWidget(self.export_btn)
        
        self.show_preview_cb = QCheckBox("Show Path Preview")
        self.show_preview_cb.stateChanged.connect(self.toggle_preview)
        layout.addWidget(self.show_preview_cb)
        
        self.clear_preview_btn = QPushButton("Clear Preview")
        self.clear_preview_btn.clicked.connect(self.clear_preview)
        layout.addWidget(self.clear_preview_btn)
        
        layout.addStretch()
        
        return layout
        
    @pyqtSlot(list)
    def show_scan_preview(self, preview_path):
        """Show scan path preview"""
        if not preview_path:
            self.clear_preview()
            return
            
        self._last_preview_path = preview_path
        
        xs, ys = zip(*preview_path)
        self.path_line.setData(xs, ys)
        self.path_line.show()
        
        self.start_marker.setData([xs[0]], [ys[0]])
        self.end_marker.setData([xs[-1]], [ys[-1]])
        self.start_marker.show()
        self.end_marker.show()
        
        if not self.show_preview_cb.isChecked():
            self.clear_preview()
            
    def toggle_preview(self, state):
        """Toggle preview visibility"""
        if state == Qt.Checked and self._last_preview_path:
            self.show_scan_preview(self._last_preview_path)
        else:
            self.clear_preview()
            
    def clear_preview(self):
        """Clear scan path preview"""
        self.path_line.hide()
        self.start_marker.hide()
        self.end_marker.hide()
        
    @pyqtSlot(object)
    def update_image(self, image_data):
        """Update displayed image"""
        self.current_image = image_data
        
        if image_data is not None:
            # Calculate levels
            valid_data = image_data[~np.isnan(image_data)]
            if len(valid_data) > 0:
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                if min_val == max_val:
                    levels = [min_val - 0.1, max_val + 0.1]
                else:
                    levels = [min_val, max_val]
            else:
                levels = [0.0, 1.0]
                
            self.image_item.setImage(image_data, levels=levels)
            
            # Set coordinate transformation
            if self.scan_params:
                rect = pg.QtCore.QRectF(
                    self.scan_params.x_start,
                    self.scan_params.y_start,
                    self.scan_params.x_end - self.scan_params.x_start,
                    self.scan_params.y_end - self.scan_params.y_start
                )
                self.image_item.setRect(rect)
                
            # Update colormap
            self.update_colormap(self.current_colormap)
            
    def set_scan_parameters(self, scan_params: ScanParameters):
        """Set scan parameters for proper scaling"""
        self.scan_params = scan_params
        
        self.plot_widget.setLabel('left', 'Y Position (nm)')
        self.plot_widget.setLabel('bottom', 'X Position (nm)')
        self.plot_widget.setTitle(
            f'Scan Image ({scan_params.x_pixels}x{scan_params.y_pixels})'
        )
        
    def update_colormap(self, colormap_name: str):
        """Update the colormap"""
        self.current_colormap = colormap_name
        
        try:
            colormap = pg.colormap.get(colormap_name)
            
            if self.current_image is not None:
                self.image_item.setColorMap(colormap)
                
            # Update colorbar
            self.colorbar_image.setColorMap(colormap)
            
        except Exception as e:
            print(f"Error updating colormap: {e}")
            
    def auto_levels(self):
        """Auto-adjust image levels"""
        if self.current_image is not None:
            valid_data = self.current_image[~np.isnan(self.current_image)]
            if len(valid_data) > 0:
                min_val = np.percentile(valid_data, 1)
                max_val = np.percentile(valid_data, 99)
                
                if min_val == max_val:
                    min_val -= 0.1
                    max_val += 0.1
                    
                self.image_item.setLevels([min_val, max_val])
                
    def save_image(self):
        """Save the current image"""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image to save")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Image",
            f"scan_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG Files (*.png);;TIFF Files (*.tif);;All Files (*)"
        )
        
        if filename:
            try:
                exporter = pg.exporters.ImageExporter(self.plot_widget.plotItem)
                exporter.export(filename)
                QMessageBox.information(self, "Success", f"Image saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {e}")
                
    def export_data(self):
        """Export current image data"""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No data to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Data",
            f"scan_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv);;NumPy Files (*.npy);;All Files (*)"
        )
        
        if filename:
            try:
                if filename.endswith('.csv'):
                    # Check if CSVExporter exists, otherwise use numpy
                    try:
                        CSVExporter.export_image(self.current_image, filename)
                    except:
                        np.savetxt(filename, self.current_image, delimiter=',', fmt='%.6f')
                elif filename.endswith('.npy'):
                    np.save(filename, self.current_image)
                else:
                    # Default to CSV
                    np.savetxt(filename, self.current_image, delimiter=',', fmt='%.6f')
                    
                QMessageBox.information(self, "Success", f"Data exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data: {e}")
