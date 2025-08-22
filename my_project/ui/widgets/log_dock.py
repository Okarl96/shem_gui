"""Log dock widget"""

from PyQt5.QtWidgets import QDockWidget, QTextEdit
from PyQt5.QtCore import QDateTime, pyqtSlot
from PyQt5.QtGui import QTextCursor


class LogDockWidget(QDockWidget):
    """System log dock widget"""
    
    def __init__(self):
        super().__init__("System Log")
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI"""
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.setWidget(self.log_text)
        
    @pyqtSlot(str)
    def add_message(self, message: str):
        """Add message to log"""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        formatted_message = f"[{timestamp}] {message}"
        
        self.log_text.append(formatted_message)
        self._scroll_to_bottom()
        self._limit_log_size()
        
    @pyqtSlot(str)
    def add_error(self, message: str):
        """Add error message to log"""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        formatted_message = f'<span style="color: red;">[{timestamp}] ERROR: {message}</span>'
        
        self.log_text.append(formatted_message)
        self._scroll_to_bottom()
        self._limit_log_size()
        
    def _scroll_to_bottom(self):
        """Scroll to bottom of log"""
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        
    def _limit_log_size(self):
        """Limit log size to prevent memory issues"""
        if self.log_text.document().blockCount() > 1000:
            cursor = self.log_text.textCursor()
            cursor.movePosition(QTextCursor.Start)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
