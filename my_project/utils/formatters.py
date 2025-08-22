"""Data formatting utilities"""

from typing import Union


class DataFormatter:
    """Format data for display"""
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time duration"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
            
    @staticmethod
    def format_position(position: float, precision: int = 1) -> str:
        """Format position value"""
        return f"{position:.{precision}f} nm"
        
    @staticmethod
    def format_current(current: float, precision: int = 4) -> str:
        """Format current value"""
        return f"{current:.{precision}f} nA"
        
    @staticmethod
    def format_rate(rate: float, precision: int = 1) -> str:
        """Format data rate"""
        return f"{rate:.{precision}f} Hz"
        
    @staticmethod
    def format_large_number(num: Union[int, float]) -> str:
        """Format large numbers with commas"""
        if isinstance(num, float):
            return f"{num:,.2f}"
        else:
            return f"{num:,}"
