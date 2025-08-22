"""Core data models."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ScanMode(Enum):
    """Scanning mode enumeration."""

    STEP_STOP = "step_stop"
    CONTINUOUS = "continuous"


class ScanState(Enum):
    """Scan state enumeration."""

    IDLE = "idle"
    SCANNING = "scanning"
    PAUSED = "paused"
    STOPPING = "stopping"


@dataclass
class ScanParameters:
    """Scan configuration parameters."""

    x_start: float = 0.0        # nm
    x_end: float = 10000.0      # nm
    y_start: float = 0.0        # nm
    y_end: float = 10000.0      # nm
    x_pixels: int = 100
    y_pixels: int = 100
    mode: ScanMode = ScanMode.STEP_STOP
    dwell_time: float = 0.75    # seconds (for step-stop)
    scan_speed: float = 1000.0  # nm/s (for continuous)
    bidirectional: bool = True

    @property
    def x_step(self) -> float:
        """Calculate X step size."""
        return (self.x_end - self.x_start) / (self.x_pixels - 1) if self.x_pixels > 1 else 0

    @property
    def y_step(self) -> float:
        """Calculate Y step size."""
        return (self.y_end - self.y_start) / (self.y_pixels - 1) if self.y_pixels > 1 else 0

    @property
    def total_pixels(self) -> int:
        """Total number of pixels."""
        return self.x_pixels * self.y_pixels

    @property
    def scan_area(self) -> tuple:
        """Return scan area as (width, height) in nm."""
        return (self.x_end - self.x_start, self.y_end - self.y_start)

    def estimated_time(self) -> float:
        """Estimate total scan time in seconds."""
        if self.mode == ScanMode.STEP_STOP:
            move_time = 0.1  # Assume 100ms per move
            return self.total_pixels * (self.dwell_time + move_time)
        x_range = self.x_end - self.x_start
        y_range = self.y_end - self.y_start
        total_distance = x_range * self.y_pixels + y_range * (self.y_pixels - 1)
        return total_distance / self.scan_speed


@dataclass
class DataPoint:
    """Single data point with timestamp and position."""

    timestamp: float
    x_pos: float
    y_pos: float
    z_pos: float | None
    r_pos: float | None
    current: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "x_pos": self.x_pos,
            "y_pos": self.y_pos,
            "z_pos": self.z_pos,
            "r_pos": self.r_pos,
            "current": self.current
        }
