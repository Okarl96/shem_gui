"""SQLite metadata storage"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import QMutex, QMutexLocker

from core.models import DataPoint, ScanParameters
from .base import IStorageBackend


class SQLiteMetaStorage(IStorageBackend):
    """SQLite storage for scan metadata"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.base_path / "scan_metadata.db"
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.mutex = QMutex()
        
        self._init_database()
        
    def _init_database(self):
        """Initialize database tables"""
        with QMutexLocker(self.mutex):
            cursor = self.conn.cursor()
            
            # Scans table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    scan_name TEXT,
                    x_start REAL, x_end REAL,
                    y_start REAL, y_end REAL,
                    x_pixels INTEGER, y_pixels INTEGER,
                    mode TEXT,
                    dwell_time REAL,
                    scan_speed REAL,
                    bidirectional BOOLEAN,
                    file_path TEXT,
                    duration REAL,
                    completed BOOLEAN
                )
            """)
            
            # Summary statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_stats (
                    scan_id INTEGER PRIMARY KEY,
                    total_points INTEGER,
                    min_current REAL,
                    max_current REAL,
                    avg_current REAL,
                    std_current REAL,
                    FOREIGN KEY (scan_id) REFERENCES scans (id)
                )
            """)
            
            self.conn.commit()
            
    def create_scan_record(self, scan_params: ScanParameters, scan_name: str) -> int:
        """Create new scan record"""
        with QMutexLocker(self.mutex):
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO scans (
                    timestamp, scan_name,
                    x_start, x_end, y_start, y_end,
                    x_pixels, y_pixels,
                    mode, dwell_time, scan_speed, bidirectional,
                    completed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(), scan_name,
                scan_params.x_start, scan_params.x_end,
                scan_params.y_start, scan_params.y_end,
                scan_params.x_pixels, scan_params.y_pixels,
                scan_params.mode.value,
                scan_params.dwell_time,
                scan_params.scan_speed,
                scan_params.bidirectional,
                False
            ))
            self.conn.commit()
            return cursor.lastrowid
            
    def save_data_point(self, scan_id: int, data_point: DataPoint):
        """Save single data point (not typically used for high-frequency data)"""
        # For high-frequency data, use batch saving instead
        pass
        
    def save_batch(self, scan_id: int, data_points: List[DataPoint]):
        """Save batch of data points (delegates to HDF5 for raw data)"""
        # Metadata storage doesn't store raw data
        # This would be handled by HDF5RawStorage
        pass
        
    def complete_scan(self, scan_id: int):
        """Mark scan as completed"""
        with QMutexLocker(self.mutex):
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE scans SET completed = ? WHERE id = ?",
                (True, scan_id)
            )
            self.conn.commit()
            
    def update_file_path(self, scan_id: int, file_path: str):
        """Update HDF5 file path for scan"""
        with QMutexLocker(self.mutex):
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE scans SET file_path = ? WHERE id = ?",
                (file_path, scan_id)
            )
            self.conn.commit()
            
    def get_scan_list(self) -> List[dict]:
        """Get list of all scans"""
        with QMutexLocker(self.mutex):
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, scan_name, completed
                FROM scans
                ORDER BY id DESC
            """)
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
            
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
