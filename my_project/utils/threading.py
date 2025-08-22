"""Thread-safe utilities."""
from __future__ import annotations

from collections import deque
from typing import Any

from PyQt5.QtCore import QMutex, QMutexLocker


class ThreadSafeBuffer:
    """Thread-safe circular buffer."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self.buffer = deque(maxlen=maxsize)
        self.mutex = QMutex()

    def append(self, item: Any) -> None:
        """Add item to buffer."""
        with QMutexLocker(self.mutex):
            self.buffer.append(item)

    def get_recent(self, n: int | None = None) -> list[Any]:
        """Get recent items."""
        with QMutexLocker(self.mutex):
            if n is None:
                return list(self.buffer)
            return list(self.buffer)[-n:]

    def clear(self) -> None:
        """Clear buffer."""
        with QMutexLocker(self.mutex):
            self.buffer.clear()

    def __len__(self) -> int:
        with QMutexLocker(self.mutex):
            return len(self.buffer)
