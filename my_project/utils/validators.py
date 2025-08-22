"""Input validation utilities."""
from __future__ import annotations

from typing import TYPE_CHECKING

from config.constants import *

if TYPE_CHECKING:
    from core.models import ScanParameters


class InputValidator:
    """Validate user inputs."""

    @staticmethod
    def validate_scan_parameters(params: ScanParameters) -> tuple[bool, str]:
        """Validate scan parameters.

        Returns
        -------
            (is_valid, error_message)
        """
        # Check position ranges
        if params.x_start >= params.x_end:
            return False, "X end must be greater than X start"

        if params.y_start >= params.y_end:
            return False, "Y end must be greater than Y start"

        # Check pixel counts
        if params.x_pixels < MIN_PIXELS or params.x_pixels > MAX_PIXELS:
            return False, f"X pixels must be between {MIN_PIXELS} and {MAX_PIXELS}"

        if params.y_pixels < MIN_PIXELS or params.y_pixels > MAX_PIXELS:
            return False, f"Y pixels must be between {MIN_PIXELS} and {MAX_PIXELS}"

        # Check timing parameters
        if params.dwell_time < MIN_DWELL_TIME or params.dwell_time > MAX_DWELL_TIME:
            return False, f"Dwell time must be between {MIN_DWELL_TIME} and {MAX_DWELL_TIME} seconds"

        if params.scan_speed < MIN_SCAN_SPEED or params.scan_speed > MAX_SCAN_SPEED:
            return False, f"Scan speed must be between {MIN_SCAN_SPEED} and {MAX_SCAN_SPEED} nm/s"

        return True, ""

    @staticmethod
    def validate_mqtt_settings(host: str, port: int) -> tuple[bool, str]:
        """Validate MQTT settings."""
        if not host:
            return False, "Host cannot be empty"

        if port < 1 or port > 65535:
            return False, "Port must be between 1 and 65535"

        return True, ""
