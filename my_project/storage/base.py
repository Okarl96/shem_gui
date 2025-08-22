"""Base storage interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import DataPoint, ScanParameters


class IStorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def create_scan_record(self, scan_params: ScanParameters, scan_name: str) -> int:
        """Create a new scan record."""

    @abstractmethod
    def save_data_point(self, scan_id: int, data_point: DataPoint):
        """Save a single data point."""

    @abstractmethod
    def save_batch(self, scan_id: int, data_points: list[DataPoint]):
        """Save a batch of data points."""

    @abstractmethod
    def complete_scan(self, scan_id: int):
        """Mark scan as completed."""

    @abstractmethod
    def close(self):
        """Close storage connection."""
