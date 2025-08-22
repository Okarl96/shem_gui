"""Custom exceptions"""


class ScanningMicroscopeError(Exception):
    """Base exception for scanning microscope"""
    pass


class ConnectionError(ScanningMicroscopeError):
    """Connection-related errors"""
    pass


class ScanError(ScanningMicroscopeError):
    """Scan-related errors"""
    pass


class StorageError(ScanningMicroscopeError):
    """Storage-related errors"""
    pass


class ValidationError(ScanningMicroscopeError):
    """Validation errors"""
    pass
