"""Thread-safe utilities"""

from collections import deque
from PyQt5.QtCore import QMutex, QMutexLocker
from typing import List, Any, Optional


class ThreadSafeBuffer:
    """Thread-safe circular buffer"""
    
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.buffer = deque(maxlen=maxsize)
        self.mutex = QMutex()
        
    def append(self, item: Any):
        """Add item to buffer"""
        with QMutexLocker(self.mutex):
            self.buffer.append(item)
            
    def get_recent(self, n: Optional[int] = None) -> List[Any]:
        """Get recent items"""
        with QMutexLocker(self.mutex):
            if n is None:
                return list(self.buffer)
            else:
                return list(self.buffer)[-n:]
                
    def clear(self):
        """Clear buffer"""
        with QMutexLocker(self.mutex):
            self.buffer.clear()
            
    def __len__(self) -> int:
        with QMutexLocker(self.mutex):
            return len(self.buffer)
