"""
PositionSignalReader

This class ingests high-frequency position and signal messages from MQTT
and exposes the latest known state in a thread-safe way.

Key design points:
- We only store the most recent position frame and most recent signal frame.
  Memory usage stays O(1) regardless of runtime length.
- Each frame carries exactly one device timestamp (from the hardware).
  We do NOT use time.time() from the host.
- We do NOT judge "freshness" here. Higher-level scan logic will do that.
- Optionally we keep short ring buffers (bounded history) for debugging/plotting.

Intended usage:
    reader = PositionSignalReader(history_size=100)

    # hook these to MQTTClient callbacks:
    mqtt.set_callbacks(
        on_position=reader.update_position,
        on_current=reader.update_signal
    )

    # later in scan logic:
    pos = reader.get_xyzr()
    sig = reader.get_signal()
    # pos["X_nm"], pos["t"], sig["value_na"], sig["t"], etc.

Message formats assumed:

Position topic payload (example):
    "1761611608272002000/5000/5000/0/0"
    [0] device timestamp in ns
    [1] X position in nm
    [2] Y position in nm
    [3] Z position in nm
    [4] R position in micro-deg

Signal / current topic payload (example):
    "1761611608272510400/100.000"
    [0] device timestamp in ns
    [1] detector reading in nA (or arbitrary unit you define)

All timestamps are stored internally as float seconds in device time.
"""

import threading
from collections import deque
from typing import Dict, Optional, List


class PositionSignalReader:
    """
    Thread-safe container for latest stage position and detector signal.

    - update_position(payload): called by MQTTClient when a new position frame arrives
    - update_signal(payload):   called by MQTTClient when a new signal frame arrives
    - get_xyzr():               read latest {X_nm, Y_nm, Z_nm, R_udeg, t}
    - get_signal():             read latest {value_na, t}
    - get_position_history():   recent position frames (optional ring buffer)
    - get_signal_history():     recent signal frames (optional ring buffer)
    - get_statistics():         counters for diagnostics

    We do NOT:
    - compute freshness / staleness
    - compute settle / tolerance
    - average the signal
    Those are responsibilities of higher-level scan logic.
    """

    def __init__(self, history_size: int = 0):
        """
        Args:
            history_size:
                0  -> no history (minimal memory, fastest)
                >0 -> keep up to N recent frames in ring buffers for debugging/plotting
        """

        # Latest known stage position frame
        # One device timestamp for all axes in that frame
        self._latest_position: Dict[str, Optional[float]] = {
            "X_nm": None,
            "Y_nm": None,
            "Z_nm": None,
            "R_udeg": None,
            "t": None,          # device time [s]
        }

        # Latest known detector signal frame
        self._latest_signal: Dict[str, Optional[float]] = {
            "value_na": None,
            "t": None,          # device time [s]
        }

        # Thread lock for all shared state
        self._lock = threading.Lock()

        # Optional bounded history
        self._history_size = history_size
        if history_size > 0:
            self._position_history: Optional[deque] = deque(maxlen=history_size)
            self._signal_history: Optional[deque] = deque(maxlen=history_size)
        else:
            self._position_history = None
            self._signal_history = None

        # Statistics / diagnostics
        self._stats: Dict[str, Optional[float]] = {
            "position_count": 0,
            "signal_count": 0,
            "parse_errors": 0,
            "last_position_t": None,   # last device time [s] we've seen for position
            "last_signal_t": None,     # last device time [s] we've seen for signal
        }

    # -------------------------------------------------------------------------
    # Update methods (called from MQTT callbacks)
    # -------------------------------------------------------------------------

    def update_position(self, payload: str) -> None:
        """
        Ingest a high-rate position update from MQTT.

        Expected format:
            "timestamp_ns/x_nm/y_nm/z_nm/r_udeg"

        Example:
            "1761611608272002000/5000/5000/0/0"

        We parse it, convert timestamp_ns -> seconds, and update the
        latest-known position frame.
        """
        try:
            parts = payload.strip().split('/')
            # Require at least timestamp + X + Y + Z + R
            if len(parts) < 5:
                return  # ignore malformed message silently

            # Convert timestamp (ns -> s, as float)
            device_ts_ns = int(parts[0])
            device_ts_s = device_ts_ns / 1e9

            # Parse numeric positions (None if NaN or invalid)
            x_nm = self._parse_float(parts[1])
            y_nm = self._parse_float(parts[2])
            z_nm = self._parse_float(parts[3])
            r_udeg = self._parse_float(parts[4])

            with self._lock:
                # Update latest frame
                self._latest_position["X_nm"] = x_nm
                self._latest_position["Y_nm"] = y_nm
                self._latest_position["Z_nm"] = z_nm
                self._latest_position["R_udeg"] = r_udeg
                self._latest_position["t"] = device_ts_s

                # Stats
                self._stats["position_count"] += 1
                self._stats["last_position_t"] = device_ts_s

                # Optional ring buffer
                if self._position_history is not None:
                    self._position_history.append({
                        "t": device_ts_s,
                        "X_nm": x_nm,
                        "Y_nm": y_nm,
                        "Z_nm": z_nm,
                        "R_udeg": r_udeg,
                    })

        except Exception:
            # Parsing failed, don't crash the high-rate callback
            with self._lock:
                self._stats["parse_errors"] += 1

    def update_signal(self, payload: str) -> None:
        """
        Ingest a detector / picoammeter frame from MQTT.

        Expected format:
            "timestamp_ns/value_na"

        Example:
            "1761611608272510400/100.000"

        We parse it, convert timestamp_ns -> seconds, and update the
        latest-known signal frame.
        """
        try:
            parts = payload.strip().split('/')
            # Require at least timestamp + value
            if len(parts) < 2:
                return  # ignore malformed

            device_ts_ns = int(parts[0])
            device_ts_s = device_ts_ns / 1e9

            value_na = self._parse_float(parts[1])

            with self._lock:
                # Update latest frame
                self._latest_signal["value_na"] = value_na
                self._latest_signal["t"] = device_ts_s

                # Stats
                self._stats["signal_count"] += 1
                self._stats["last_signal_t"] = device_ts_s

                # Optional ring buffer
                if self._signal_history is not None:
                    self._signal_history.append({
                        "t": device_ts_s,
                        "value_na": value_na,
                    })

        except Exception:
            with self._lock:
                self._stats["parse_errors"] += 1

    # -------------------------------------------------------------------------
    # Read methods (called by scan / control logic)
    # -------------------------------------------------------------------------

    def get_xyzr(self) -> Dict[str, Optional[float]]:
        """
        Get the most recent stage position frame.

        Returns a dict with:
            {
                "X_nm": float | None,
                "Y_nm": float | None,
                "Z_nm": float | None,
                "R_udeg": float | None,
                "t": float | None,   # device timestamp [s] for this frame
            }

        The timestamp applies to all four axes in this frame,
        because the controller provided them together.
        """
        with self._lock:
            return {
                "X_nm": self._latest_position["X_nm"],
                "Y_nm": self._latest_position["Y_nm"],
                "Z_nm": self._latest_position["Z_nm"],
                "R_udeg": self._latest_position["R_udeg"],
                "t": self._latest_position["t"],
            }

    def get_signal(self) -> Dict[str, Optional[float]]:
        """
        Get the most recent detector reading.

        Returns a dict with:
            {
                "value_na": float | None,
                "t": float | None,   # device timestamp [s] for this frame
            }
        """
        with self._lock:
            return {
                "value_na": self._latest_signal["value_na"],
                "t": self._latest_signal["t"],
            }

    # -------------------------------------------------------------------------
    # Optional history access
    # -------------------------------------------------------------------------

    def get_position_history(self) -> Optional[List[Dict[str, Optional[float]]]]:
        """
        Return a *copy* of the recent position frames if history is enabled.
        Each entry is:
            {
                "t": <device time s>,
                "X_nm": ...,
                "Y_nm": ...,
                "Z_nm": ...,
                "R_udeg": ...,
            }

        Returns:
            list[...] or None if history was disabled (history_size=0)
        """
        if self._position_history is None:
            return None
        with self._lock:
            return list(self._position_history)

    def get_signal_history(self) -> Optional[List[Dict[str, Optional[float]]]]:
        """
        Return a *copy* of the recent detector frames if history is enabled.
        Each entry is:
            {
                "t": <device time s>,
                "value_na": ...
            }

        Returns:
            list[...] or None if history was disabled (history_size=0)
        """
        if self._signal_history is None:
            return None
        with self._lock:
            return list(self._signal_history)

    # -------------------------------------------------------------------------
    # Diagnostics / stats
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Optional[float]]:
        """
        Return counters and last-seen timestamps.

        Returns a dict like:
            {
                "position_count": int,
                "signal_count": int,
                "parse_errors": int,
                "last_position_t": float | None,
                "last_signal_t": float | None,
            }
        """
        with self._lock:
            return dict(self._stats)

    def reset_statistics(self) -> None:
        """
        Reset message counters and parse error count.
        (Does not clear the latest position/signal frames.)
        """
        with self._lock:
            self._stats["position_count"] = 0
            self._stats["signal_count"] = 0
            self._stats["parse_errors"] = 0
            # we intentionally leave last_position_t / last_signal_t as-is

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_float(value: str) -> Optional[float]:
        """
        Convert string to float, returning None on 'NaN' or invalid.
        """
        if value in ("NaN", "nan", "None", ""):
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def __repr__(self) -> str:
        """
        Human-friendly preview of current state, useful in REPL/notebook.
        """
        pos = self.get_xyzr()
        sig = self.get_signal()
        return (
            "PositionSignalReader("
            f"X={pos['X_nm']}, "
            f"Y={pos['Y_nm']}, "
            f"Z={pos['Z_nm']}, "
            f"R={pos['R_udeg']}, "
            f"signal={sig['value_na']})"
        )
