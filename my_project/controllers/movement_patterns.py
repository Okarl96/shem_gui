
"""Movement pattern generators - FIXED VERSION"""

from abc import ABC, abstractmethod
from typing import List, Tuple

from core.models import ScanParameters
from core.exceptions import ValidationError


class MovementPattern(ABC):
    """Abstract base class for movement patterns"""
    
    @abstractmethod
    def generate(self, params: ScanParameters) -> List[Tuple[float, float, int, int]]:
        """Generate movement pattern
        
        Returns:
            List of (x_pos, y_pos, x_idx, y_idx) tuples
        """
        pass


class RasterPattern(MovementPattern):
    """Raster scan pattern generator with validation"""
    
    def __init__(self, bidirectional: bool = True):
        self.bidirectional = bidirectional
        
    def generate(self, params: ScanParameters) -> List[Tuple[float, float, int, int]]:
        """Generate raster scan pattern with validation"""
        # Validate parameters
        if params.x_pixels < 1 or params.y_pixels < 1:
            raise ValidationError("Pixel count must be at least 1")
            
        if params.x_pixels == 1 and params.y_pixels == 1:
            # Single point scan
            return [(params.x_start, params.y_start, 0, 0)]
            
        pattern = []
        
        # Calculate step sizes safely
        x_step = params.x_step if params.x_pixels > 1 else 0
        y_step = params.y_step if params.y_pixels > 1 else 0
        
        for y_idx in range(params.y_pixels):
            y_pos = params.y_start + y_idx * y_step
            
            # Determine X direction based on bidirectional setting
            if self.bidirectional and y_idx % 2 == 1:
                # Reverse direction for odd rows
                x_range = range(params.x_pixels - 1, -1, -1)
            else:
                # Normal direction
                x_range = range(params.x_pixels)
                
            for x_idx in x_range:
                x_pos = params.x_start + x_idx * x_step
                pattern.append((x_pos, y_pos, x_idx, y_idx))
                
        return pattern


class SpiralPattern(MovementPattern):
    """Spiral scan pattern generator (future implementation)"""
    
    def generate(self, params: ScanParameters) -> List[Tuple[float, float, int, int]]:
        """Generate spiral scan pattern"""
        raise NotImplementedError("Spiral pattern not yet implemented")


class LissajousPattern(MovementPattern):
    """Lissajous scan pattern generator (future implementation)"""
    
    def generate(self, params: ScanParameters) -> List[Tuple[float, float, int, int]]:
        """Generate Lissajous scan pattern"""
        raise NotImplementedError("Lissajous pattern not yet implemented")
