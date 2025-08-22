"""Custom exceptions."""
from __future__ import annotations


class ScanningMicroscopeError(Exception):
    """Base exception for scanning microscope."""


class ConnectionError(ScanningMicroscopeError):
    """Connection-related errors."""


class ScanError(ScanningMicroscopeError):
    """Scan-related errors."""


class StorageError(ScanningMicroscopeError):
    """Storage-related errors."""


class ValidationError(ScanningMicroscopeError):
    """Validation errors."""
