"""Style manager for application theming"""

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt  # Add this import
from PyQt5.QtGui import QPalette, QColor


class StyleManager:
    """Manage application styles and themes"""
    
    def __init__(self):
        self.current_theme = "fusion_light"
        
    def apply_theme(self, app: QApplication, theme: str = None):
        """Apply theme to application"""
        if theme:
            self.current_theme = theme
            
        app.setStyle('Fusion')
        
        if self.current_theme == "fusion_light":
            self._apply_light_theme(app)
        elif self.current_theme == "fusion_dark":
            self._apply_dark_theme(app)
            
    def _apply_light_theme(self, app: QApplication):
        """Apply light theme"""
        palette = QPalette()
        
        # Window colors
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        
        # Base colors
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
        
        # Text colors
        palette.setColor(QPalette.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        
        # Button colors
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
        
        # Highlight colors
        palette.setColor(QPalette.Highlight, QColor(76, 163, 224))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        
        app.setPalette(palette)
        
    def _apply_dark_theme(self, app: QApplication):
        """Apply dark theme"""
        palette = QPalette()
        
        # Window colors
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        
        # Base colors
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        
        # Text colors
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        
        # Button colors
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        
        # Highlight colors
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        
        app.setPalette(palette)
