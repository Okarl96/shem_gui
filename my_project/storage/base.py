"""Base storage interface"""

from abc import ABC, abstractmethod
from typing import List, Optional

from core.models import DataPoint, ScanParameters


class IStorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    def create_scan_record(self, scan_params: ScanParameters, scan_name: str) -> int:
        """Create a new scan record"""
        pass
    
    @abstractmethod
    def save_data_point(self, scan_id: int, data_point: DataPoint):
        """Save a single data point"""
        pass
    
    @abstractmethod
    def save_batch(self, scan_id: int, data_points: List[DataPoint]):
        """Save a batch of data points"""
        pass
    
    @abstractmethod
    def complete_scan(self, scan_id: int):
        """Mark scan as completed"""
        pass
    
    @abstractmethod
    def close(self):
        """Close storage connection"""
        pass
