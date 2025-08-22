"""Physical constants and system limits."""

# Physical Limits
from __future__ import annotations

MAX_POSITION_NM = 10000000  # Maximum position in nanometers
MIN_POSITION_NM = -10000000  # Minimum position in nanometers
MAX_PIXELS = 2000  # Maximum pixels per axis
MIN_PIXELS = 1  # Minimum pixels per axis

# Timing Constants
MIN_DWELL_TIME = 0.001  # Minimum dwell time in seconds
MAX_DWELL_TIME = 10.0  # Maximum dwell time in seconds
MIN_SCAN_SPEED = 1.0  # Minimum scan speed in nm/s
MAX_SCAN_SPEED = 10000.0  # Maximum scan speed in nm/s

# Data Rates
POSITION_DATA_RATE = 6000  # Hz (ECC100)
CURRENT_DATA_RATE = 40  # Hz (Picoammeter)

# MQTT Topics
MQTT_TOPICS = {
    "picoammeter": "picoammeter/current",
    "stage_position": "microscope/stage/position",
    "stage_command": "microscope/stage/command",
    "stage_result": "microscope/stage/result"
}

# File Formats
SUPPORTED_IMAGE_FORMATS = "PNG Files (*.png);;TIFF Files (*.tif);;All Files (*)"
SUPPORTED_DATA_FORMATS = "CSV Files (*.csv);;NumPy Files (*.npy);;All Files (*)"

# Colors
STATUS_COLORS = {
    "connected": "green",
    "disconnected": "red",
    "scanning": "green",
    "idle": "black",
    "error": "red",
    "warning": "orange"
}
