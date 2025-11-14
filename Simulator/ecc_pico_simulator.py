#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECC + Picoammeter Simulator with Z-Stack, Rotation, and X-Z Compensation

NEW FEATURES:
  - Proper stage coordinate system
  - Sample positioning in stage space
  - 3D Center of Rotation (COR) support with cor_x, cor_y, cor_z
  - Smooth rotation movement
  - Out-of-bounds detection (returns 0 signal)
  - Dynamic COR adjustment via MQTT
  - X-Z Compensation: Image shifts in X as Z changes (independent of COR)
  - COR also shifts with Z using cor_z as reference point

Topics:
  - microscope/stage/position   : "timestamp_ns/X/Y/Z/R"
  - microscope/picoammeter      : "timestamp_ns/current_pA"
  - microscope/stage/command    : accepts MOVE, SET_RATE, SET_COR, STATUS
  - microscope/stage/result     : command results

Dependencies: pip install paho-mqtt pillow numpy
"""

import argparse, time, threading, signal
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image

try:
    import paho.mqtt.client as mqtt

    HAVE_MQTT = True
except Exception:
    HAVE_MQTT = False
    mqtt = None

AXES = ("X", "Y", "Z", "R")  # X/Y/Z in nm, R in micro-degrees


def now_ns() -> int:
    return time.time_ns()


class ImageSampler:
    """Single image sampler with pixel interpolation."""

    def __init__(self, path: str, fov_nm: Tuple[float, float] | None):
        img = Image.open(path).convert("L")
        self.w, self.h = img.size
        self.arr = np.asarray(img, dtype=np.float32) / 255.0

        if fov_nm is None:
            self.fov_x_nm = float(self.w)
            self.fov_y_nm = float(self.h)
        else:
            self.fov_x_nm, self.fov_y_nm = map(float, fov_nm)

        self.nm_per_px_x = self.fov_x_nm / self.w
        self.nm_per_px_y = self.fov_y_nm / self.h

    def sample_pixel(self, px: float, py: float) -> float:
        """Sample at pixel coordinates with bounds checking."""
        if px < 0 or px >= self.w or py < 0 or py >= self.h:
            return 0.0

        # Nearest neighbor sampling
        ipx = int(round(px))
        ipy = int(round(py))
        ipx = max(0, min(self.w - 1, ipx))
        ipy = max(0, min(self.h - 1, ipy))
        return float(self.arr[ipy, ipx])


class ZStackSampler:
    """
    Z-stack sampler with coordinate system, rotation, and X-Z compensation support.

    Image position: Slides in X based on Z directly
    COR position: Slides in X based on (Z - cor_z)
    """

    def __init__(self, image_paths: List[str], z_positions: List[float],
                 fov_nm: Tuple[float, float] | None,
                 sample_center_x_base: float = 0.0,
                 sample_center_y: float = 0.0,
                 x_per_z_ratio: float = 1.0):
        """
        Args:
            image_paths: List of image file paths
            z_positions: Z positions for each image
            fov_nm: Field of view (width, height)
            sample_center_x_base: X position of sample center at Z=0
            sample_center_y: Y position of sample center
            x_per_z_ratio: X shift per unit Z change (default 1.0 = 1nm X per 1nm Z)
        """
        if len(image_paths) != len(z_positions):
            raise ValueError(f"Number of images must match Z positions")

        self.samplers = []
        self.z_positions = sorted(z_positions)

        # Load all images
        for i, path in enumerate(image_paths):
            print(f"Loading image {i + 1}/{len(image_paths)}: {path} at Z={z_positions[i]} nm")
            self.samplers.append(ImageSampler(path, fov_nm))

        # Get FOV from first sampler
        self.fov_x_nm = self.samplers[0].fov_x_nm
        self.fov_y_nm = self.samplers[0].fov_y_nm

        # Store base sample position (at Z=0)
        self.sample_center_x_base = sample_center_x_base
        self.sample_center_y = sample_center_y

        # X-Z compensation parameters
        self.x_per_z_ratio = x_per_z_ratio

        print(f"Z-stack initialized: {len(self.samplers)} images")
        print(f"Z range: {self.z_positions[0]} to {self.z_positions[-1]} nm")
        print(f"Sample center at Z=0: ({sample_center_x_base}, {sample_center_y}) nm")
        print(f"X-Z compensation: {x_per_z_ratio} nm X shift per nm Z change")

    def get_image_x_offset(self, z_nm: float) -> float:
        """Calculate image X offset based on Z position (always relative to Z=0)."""
        return z_nm * self.x_per_z_ratio

    def get_cor_x_offset(self, z_nm: float, cor_z: float) -> float:
        """Calculate COR X offset based on Z position (relative to cor_z)."""
        return (z_nm - cor_z) * self.x_per_z_ratio

    def get_sample_center_at_z(self, z_nm: float) -> Tuple[float, float]:
        """Get image center position at given Z (independent of COR)."""
        x_offset = self.get_image_x_offset(z_nm)
        return self.sample_center_x_base + x_offset, self.sample_center_y

    def get_sample_bounds_at_z(self, z_nm: float) -> Tuple[float, float, float, float]:
        """Get sample boundaries at given Z position. Returns (left, right, bottom, top)."""
        center_x, center_y = self.get_sample_center_at_z(z_nm)
        left = center_x - self.fov_x_nm / 2.0
        right = center_x + self.fov_x_nm / 2.0
        bottom = center_y - self.fov_y_nm / 2.0
        top = center_y + self.fov_y_nm / 2.0
        return left, right, bottom, top

    def is_within_sample(self, x_nm: float, y_nm: float, z_nm: float) -> bool:
        """Check if stage position is within sample bounds at given Z."""
        left, right, bottom, top = self.get_sample_bounds_at_z(z_nm)
        return (left <= x_nm <= right and bottom <= y_nm <= top)

    def stage_to_image_coords(self, x_nm: float, y_nm: float, z_nm: float) -> Tuple[float, float]:
        """Convert stage coordinates to image pixel coords."""
        center_x, center_y = self.get_sample_center_at_z(z_nm)
        sample_left = center_x - self.fov_x_nm / 2.0
        sample_top = center_y + self.fov_y_nm / 2.0

        px = (x_nm - sample_left) / self.samplers[0].nm_per_px_x
        # flip Y: top (sample_top) → row 0, bottom (sample_bottom) → row h-1
        py = (sample_top - y_nm) / self.samplers[0].nm_per_px_y
        return px, py

    def apply_rotation_transform(self, probe_x: float, probe_y: float,
                                 rotation_udeg: float, cor_x: float, cor_y: float) -> Tuple[float, float]:
        """
        Apply inverse rotation to find what part of the sample is under the probe.
        When rotation_udeg increases, the sample rotates clockwise.
        """
        # Convert micro-degrees to radians (positive for clockwise rotation)
        theta = np.radians(rotation_udeg / 1000000.0)

        # Vector from COR to probe
        dx = probe_x - cor_x
        dy = probe_y - cor_y

        # Apply inverse clockwise rotation to find original sample position
        sample_x = np.cos(theta) * dx - np.sin(theta) * dy + cor_x
        sample_y = np.sin(theta) * dx + np.cos(theta) * dy + cor_y

        return sample_x, sample_y

    def sample_z_interpolated(self, px: float, py: float, z_nm: float) -> float:
        """Sample with Z interpolation at pixel coordinates."""
        # Handle Z outside range
        if z_nm <= self.z_positions[0]:
            return self.samplers[0].sample_pixel(px, py)
        if z_nm >= self.z_positions[-1]:
            return self.samplers[-1].sample_pixel(px, py)

        # Find bracketing Z planes
        for i in range(len(self.z_positions) - 1):
            z_lower = self.z_positions[i]
            z_upper = self.z_positions[i + 1]

            if z_lower <= z_nm <= z_upper:
                # Linear interpolation weight
                alpha = 0.0 if z_upper == z_lower else (z_nm - z_lower) / (z_upper - z_lower)

                # Sample from both planes
                intensity_lower = self.samplers[i].sample_pixel(px, py)
                intensity_upper = self.samplers[i + 1].sample_pixel(px, py)

                # Interpolate
                return (1.0 - alpha) * intensity_lower + alpha * intensity_upper

        # Fallback
        return self.samplers[len(self.samplers) // 2].sample_pixel(px, py)

    def sample_full(self, probe_x: float, probe_y: float, z_nm: float,
                    rotation_udeg: float, cor_x_base: float, cor_y: float, cor_z: float) -> float:
        """
        Complete sampling with rotation, bounds checking, and X-Z compensation.

        Image position: Determined by Z directly (at Z=0: center_x_base, at Z=1000: center_x_base+1000)
        COR position: Determined by (Z - cor_z) offset

        Args:
            probe_x, probe_y: Stage position of probe
            z_nm: Z position
            rotation_udeg: Rotation angle
            cor_x_base: Base X position of COR (at cor_z)
            cor_y: Y position of COR
            cor_z: Z position where COR has no X offset

        Returns:
            Intensity [0,1] if within sample, 0.0 if outside
        """
        # Calculate effective COR at this Z (COR slides based on cor_z reference)
        cor_x_offset = self.get_cor_x_offset(z_nm, cor_z)
        cor_x_effective = cor_x_base + cor_x_offset

        # Apply rotation if needed
        if abs(rotation_udeg) > 1.0:  # More than 0.001 degrees
            sample_x, sample_y = self.apply_rotation_transform(
                probe_x, probe_y, rotation_udeg, cor_x_effective, cor_y
            )
        else:
            sample_x, sample_y = probe_x, probe_y

        # Check bounds at this Z position
        if not self.is_within_sample(sample_x, sample_y, z_nm):
            return 0.0

        # Convert to pixel coordinates
        px, py = self.stage_to_image_coords(sample_x, sample_y, z_nm)

        # Sample with Z interpolation
        return self.sample_z_interpolated(px, py, z_nm)


class State:
    """
    Enhanced state with 3D rotation support and COR.

    NEW FEATURES:
    - 3D Center of Rotation (COR) coordinates: cor_x, cor_y, cor_z
    - Smooth rotation movement
    - Stage limits
    """

    def __init__(self, pos_rate_hz: int, sig_rate_hz: int,
                 speed_xy: float = 2000.0, speed_z: float = 1000.0,
                 speed_r: float = 45000.0,  # micro-degrees per second
                 cor_x: float = 0.0, cor_y: float = 0.0, cor_z: float = 0.0):
        self.lock = threading.Lock()

        # Current positions
        self.X = 0.0  # nm
        self.Y = 0.0  # nm
        self.Z = 0.0  # nm
        self.R = 0.0  # micro-degrees

        # Target positions
        self.Tx = 0.0
        self.Ty = 0.0
        self.Tz = 0.0
        self.Tr = 0.0

        # Speeds
        self.speed_xy = float(speed_xy)  # nm/s
        self.speed_z = float(speed_z)  # nm/s
        self.speed_r = float(speed_r)  # micro-deg/s

        # Center of Rotation (3D)
        self.cor_x = cor_x
        self.cor_y = cor_y
        self.cor_z = cor_z

        # Stage limits
        self.stage_x_min = -10000.0
        self.stage_x_max = 10000.0
        self.stage_y_min = -10000.0
        self.stage_y_max = 10000.0
        self.stage_z_min = -1000.0
        self.stage_z_max = 2000.0

        # Rates
        self.pos_rate = pos_rate_hz
        self.sig_rate = sig_rate_hz

    def set_target(self, axis: str, value: float):
        with self.lock:
            if axis == "X":
                self.Tx = np.clip(value, self.stage_x_min, self.stage_x_max)
            elif axis == "Y":
                self.Ty = np.clip(value, self.stage_y_min, self.stage_y_max)
            elif axis == "Z":
                self.Tz = np.clip(value, self.stage_z_min, self.stage_z_max)
            elif axis == "R":
                self.Tr = value  # No limits on rotation

    def set_cor(self, x: float, y: float, z: float):
        """Set 3D Center of Rotation."""
        with self.lock:
            self.cor_x = x
            self.cor_y = y
            self.cor_z = z

    def get_cor(self) -> Tuple[float, float, float]:
        with self.lock:
            return self.cor_x, self.cor_y, self.cor_z

    def get_axes(self) -> Tuple[float, float, float, float]:
        with self.lock:
            return self.X, self.Y, self.Z, self.R

    def step_motion(self, dt: float):
        """Move all axes smoothly toward targets."""
        with self.lock:
            # X movement
            dx = self.Tx - self.X
            max_step_x = self.speed_xy * dt
            if abs(dx) <= max_step_x:
                self.X = self.Tx
            else:
                self.X += max_step_x if dx > 0 else -max_step_x

            # Y movement
            dy = self.Ty - self.Y
            max_step_y = self.speed_xy * dt
            if abs(dy) <= max_step_y:
                self.Y = self.Ty
            else:
                self.Y += max_step_y if dy > 0 else -max_step_y

            # Z movement
            dz = self.Tz - self.Z
            max_step_z = self.speed_z * dt
            if abs(dz) <= max_step_z:
                self.Z = self.Tz
            else:
                self.Z += max_step_z if dz > 0 else -max_step_z

            # R movement (smooth rotation)
            dr = self.Tr - self.R
            max_step_r = self.speed_r * dt
            if abs(dr) <= max_step_r:
                self.R = self.Tr
            else:
                self.R += max_step_r if dr > 0 else -max_step_r

    def set_rate_both(self, hz: int):
        with self.lock:
            self.pos_rate = hz
            self.sig_rate = hz

    def get_pos_dt(self) -> float:
        with self.lock:
            return 1.0 / float(max(1, self.pos_rate))

    def get_sig_dt(self) -> float:
        with self.lock:
            return 1.0 / float(max(1, self.sig_rate))


class Bus:
    """MQTT client with command handling."""

    def __init__(self, broker: str, port: int):
        self.client = None
        self.connected = False
        if HAVE_MQTT:
            try:
                self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
            except Exception:
                self.client = mqtt.Client()
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.connect(broker, port, 60)
            self.client.loop_start()
        else:
            print("[WARN] paho-mqtt not installed; running in console mode.")
        self.cmd_handler = None

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        self.connected = True
        try:
            rc_val = getattr(reason_code, "value", reason_code)
        except Exception:
            rc_val = reason_code
        print(f"[MQTT] Connected rc={rc_val}")
        client.subscribe("microscope/stage/command")

    def _on_message(self, client, userdata, msg):
        payload = msg.payload.decode(errors="replace")
        if self.cmd_handler:
            self.cmd_handler(payload)

    def pub(self, topic: str, payload: str, qos: int = 0):
        if self.client and self.connected:
            self.client.publish(topic, payload, qos=qos, retain=False)
        else:
            print(f"[{topic}] {payload}")

    def stop(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


def make_status(state: State, sampler: ZStackSampler) -> str:
    """Generate comprehensive status with rotation and X-Z compensation info."""
    X, Y, Z, R = state.get_axes()
    cor_x, cor_y, cor_z = state.get_cor()

    # Calculate offsets
    image_x_offset = sampler.get_image_x_offset(Z)
    cor_x_offset = sampler.get_cor_x_offset(Z, cor_z)
    cor_x_effective = cor_x + cor_x_offset

    center_x, center_y = sampler.get_sample_center_at_z(Z)
    left, right, bottom, top = sampler.get_sample_bounds_at_z(Z)

    # Check if probe is within sample
    if abs(R) > 1.0:
        sample_x, sample_y = sampler.apply_rotation_transform(
            X, Y, R, cor_x_effective, cor_y
        )
    else:
        sample_x, sample_y = X, Y

    in_sample = sampler.is_within_sample(sample_x, sample_y, Z)

    # Find active Z planes
    active_planes = "Outside range"
    for i in range(len(sampler.z_positions) - 1):
        if sampler.z_positions[i] <= Z <= sampler.z_positions[i + 1]:
            if sampler.z_positions[i + 1] != sampler.z_positions[i]:
                weight = (Z - sampler.z_positions[i]) / (sampler.z_positions[i + 1] - sampler.z_positions[i])
                active_planes = f"Z{sampler.z_positions[i]}<->Z{sampler.z_positions[i + 1]} ({weight:.1%})"
            break

    lines = [
        "=== ECC Z-Stack Rotation Simulator with X-Z Compensation ===",
        f"MQTT Connected: YES",
        "",
        "== Sample Info ==",
        f"Sample Center (at Z=0): ({sampler.sample_center_x_base:.1f}, {sampler.sample_center_y:.1f}) nm",
        f"X-Z Compensation: {sampler.x_per_z_ratio} nm/nm",
        f"",
        f"Current Z: {Z:.1f} nm",
        f"  Image X offset: {image_x_offset:.1f} nm",
        f"  Image Center: ({center_x:.1f}, {center_y:.1f}) nm",
        f"  Image Bounds: X=[{left:.1f}, {right:.1f}], Y=[{bottom:.1f}, {top:.1f}] nm",
        f"FOV: {int(sampler.fov_x_nm)}x{int(sampler.fov_y_nm)} nm",
        f"Z-Stack: {len(sampler.samplers)} images [{sampler.z_positions[0]}-{sampler.z_positions[-1]} nm]",
        "",
        "== Stage Position ==",
        f"X: {X:.1f} nm",
        f"Y: {Y:.1f} nm",
        f"Z: {Z:.1f} nm ({active_planes})",
        f"R: {R:.1f} micro-deg ({R / 1000000:.2f} deg)",
        "",
        "== 3D Center of Rotation ==",
        f"COR (base at cor_z): ({cor_x:.1f}, {cor_y:.1f}, {cor_z:.1f}) nm",
        f"  COR X offset at current Z: {cor_x_offset:.1f} nm",
        f"  Effective COR: ({cor_x_effective:.1f}, {cor_y:.1f}, {cor_z:.1f}) nm",
        f"Effective sample pos under probe: ({sample_x:.1f}, {sample_y:.1f})",
        f"Probe in sample: {'YES' if in_sample else 'NO (signal=0)'}",
        "",
        "== Rates ==",
        f"Position: {int(1.0 / state.get_pos_dt())} Hz",
        f"Signal: {int(1.0 / state.get_sig_dt())} Hz",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="ECC + Pico Z-Stack Rotation Simulator with X-Z Compensation")

    # Image and Z configuration
    ap.add_argument("--images", nargs='+', required=True, metavar="IMG",
                    help="Image files for Z-stack (in Z order)")
    ap.add_argument("--z-positions", nargs='+', type=float,
                    metavar="Z", help="Z positions for each image in nm")

    # Sample positioning
    ap.add_argument("--sample-center-x", type=float, default=0.0,
                    help="X position of sample center at Z=0 (nm)")
    ap.add_argument("--sample-center-y", type=float, default=0.0,
                    help="Y position of sample center (nm)")

    # X-Z Compensation
    ap.add_argument("--x-per-z-nm", type=float, default=1.0,
                    help="X shift per unit Z change (default: 1.0 = 1nm X per 1nm Z)")

    # 3D Center of Rotation
    ap.add_argument("--cor-x", type=float, default=0.0,
                    help="X position of rotation center at cor_z (nm)")
    ap.add_argument("--cor-y", type=float, default=0.0,
                    help="Y position of rotation center (nm)")
    ap.add_argument("--cor-z", type=float, default=0.0,
                    help="Z position of rotation center - reference for COR X-Z compensation (nm)")

    # MQTT
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--port", type=int, default=1883)

    # Rates
    ap.add_argument("--pos-rate", type=int, default=100, help="position rate (Hz)")
    ap.add_argument("--sig-rate", type=int, default=100, help="signal rate (Hz)")

    # FOV
    ap.add_argument("--fov-x", type=float, help="FOV width in nm")
    ap.add_argument("--fov-y", type=float, help="FOV height in nm")

    # Signal
    ap.add_argument("--gain-pa", type=float, default=1000.0, help="pA per unit intensity")
    ap.add_argument("--offset-pa", type=float, default=100.0, help="baseline pA")

    # Speeds
    ap.add_argument("--speed-xy", type=float, default=2000.0, help="X/Y speed (nm/s)")
    ap.add_argument("--speed-z", type=float, default=1000.0, help="Z speed (nm/s)")
    ap.add_argument("--speed-r", type=float, default=45000000.0,
                    help="Rotation speed (micro-degrees/s, default 45 deg/s)")

    args = ap.parse_args()

    # Handle Z positions
    if args.z_positions:
        if len(args.z_positions) != len(args.images):
            ap.error(f"Number of Z positions must match number of images")
        z_positions = args.z_positions
    else:
        # Default Z spacing
        z_step = 250.0
        z_positions = [i * z_step for i in range(len(args.images))]

    # Create sampler with X-Z compensation
    fov = None if (args.fov_x is None or args.fov_y is None) else (args.fov_x, args.fov_y)
    sampler = ZStackSampler(
        args.images, z_positions, fov,
        args.sample_center_x, args.sample_center_y,
        args.x_per_z_nm
    )

    # Create state with 3D COR
    state = State(
        pos_rate_hz=args.pos_rate,
        sig_rate_hz=args.sig_rate,
        speed_xy=args.speed_xy,
        speed_z=args.speed_z,
        speed_r=args.speed_r,
        cor_x=args.cor_x,
        cor_y=args.cor_y,
        cor_z=args.cor_z
    )

    # MQTT bus
    bus = Bus(args.broker, args.port)

    # Command handler
    def handle_cmd(text: str):
        ts = str(now_ns())
        parts = text.strip().split("/")
        if not parts or not parts[0]:
            return
        cmd = parts[0].upper()

        # MOVE command
        if cmd == "MOVE" and len(parts) == 3:
            axis, val = parts[1].upper(), parts[2]
            if axis in AXES:
                try:
                    v = float(val)
                    state.set_target(axis, v)

                    # Check if target is within stage limits
                    actual_target = {"X": state.Tx, "Y": state.Ty,
                                     "Z": state.Tz, "R": state.Tr}[axis]

                    bus.pub("microscope/stage/result",
                            f"{ts}/COMMAND/MOVE/{axis}/SUCCESS/Target={actual_target:.1f}", qos=1)
                except Exception as e:
                    bus.pub("microscope/stage/result",
                            f"{ts}/COMMAND/MOVE/{axis}/FAILED/{type(e).__name__}", qos=1)
            else:
                bus.pub("microscope/stage/result",
                        f"{ts}/COMMAND/MOVE/{parts[1]}/FAILED/Invalid axis", qos=1)
            return

        # SET_COR command (accepts 3 parameters for X, Y, Z)
        if cmd == "SET_COR" and len(parts) == 4:
            try:
                cor_x = float(parts[1])
                cor_y = float(parts[2])
                cor_z = float(parts[3])
                state.set_cor(cor_x, cor_y, cor_z)
                bus.pub("microscope/stage/result",
                        f"{ts}/COMMAND/SET_COR/SUCCESS/COR=({cor_x:.1f},{cor_y:.1f},{cor_z:.1f})", qos=1)
            except Exception as e:
                bus.pub("microscope/stage/result",
                        f"{ts}/COMMAND/SET_COR/FAILED/{type(e).__name__}", qos=1)
            return

        # SET_RATE command
        if cmd == "SET_RATE" and len(parts) == 2:
            try:
                hz = int(parts[1])
                if hz >= 1:
                    state.set_rate_both(hz)
                    bus.pub("microscope/stage/result",
                            f"{ts}/COMMAND/SET_RATE/ALL/SUCCESS/Rate={hz}Hz", qos=1)
                else:
                    bus.pub("microscope/stage/result",
                            f"{ts}/COMMAND/SET_RATE/ALL/FAILED/Hz must be >=1", qos=1)
            except Exception as e:
                bus.pub("microscope/stage/result",
                        f"{ts}/COMMAND/SET_RATE/ALL/FAILED/{type(e).__name__}", qos=1)
            return

        # STATUS command
        if cmd == "STATUS":
            status_text = make_status(state, sampler)
            bus.pub("microscope/stage/result",
                    f"{ts}/STATUS/SYSTEM_INFO/ALL/SUCCESS/{status_text}", qos=1)
            return

        # Unknown
        bus.pub("microscope/stage/result",
                f"{ts}/COMMAND/{cmd}/FAILED/Unknown command", qos=1)

    bus.cmd_handler = handle_cmd

    # Thread control
    running = True

    def on_sigint(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_sigint)

    # Position broadcast thread
    def pos_loop():
        next_t = time.perf_counter()
        while running:
            dt = state.get_pos_dt()
            state.step_motion(dt)  # Smooth movement for all axes

            X, Y, Z, R = state.get_axes()
            line = f"{now_ns()}/{int(X)}/{int(Y)}/{int(Z)}/{int(R)}"
            bus.pub("microscope/stage/position", line, qos=0)

            next_t += dt
            rem = next_t - time.perf_counter()
            if rem > 0:
                time.sleep(rem)
            else:
                next_t = time.perf_counter()

    # Signal broadcast thread
    def sig_loop():
        next_t = time.perf_counter()
        while running:
            X, Y, Z, R = state.get_axes()
            cor_x, cor_y, cor_z = state.get_cor()

            # Full sampling with rotation, bounds checking, and X-Z compensation
            intensity = sampler.sample_full(X, Y, Z, R, cor_x, cor_y, cor_z)
            current = args.offset_pa + args.gain_pa * intensity

            line = f"{now_ns()}/{current:.3f}"
            bus.pub("picoammeter/current", line, qos=0)

            next_t += state.get_sig_dt()
            rem = next_t - time.perf_counter()
            if rem > 0:
                time.sleep(rem)
            else:
                next_t = time.perf_counter()

    t1 = threading.Thread(target=pos_loop, daemon=True)
    t2 = threading.Thread(target=sig_loop, daemon=True)
    t1.start()
    t2.start()

    print("\n" + "=" * 50)
    print("ECC Z-Stack ROTATION Simulator")
    print("with X-Z Compensation and 3D COR")
    print("=" * 50)
    print(f"\n🎯 IMAGE POSITIONING (independent of COR):")
    print(f"  Sample center at Z=0: ({args.sample_center_x}, {args.sample_center_y}) nm")
    print(f"  X-Z compensation ratio: {args.x_per_z_nm} nm/nm")
    center_x_at_z500, _ = sampler.get_sample_center_at_z(500)
    center_x_at_z1000, _ = sampler.get_sample_center_at_z(1000)
    print(f"  Sample center at Z=500: ({center_x_at_z500:.1f}, {args.sample_center_y}) nm")
    print(f"  Sample center at Z=1000: ({center_x_at_z1000:.1f}, {args.sample_center_y}) nm")
    left0, right0, bottom0, top0 = sampler.get_sample_bounds_at_z(0)
    print(f"  Sample bounds at Z=0: X=[{left0:.1f}, {right0:.1f}], Y=[{bottom0:.1f}, {top0:.1f}] nm")
    print(f"\n🔄 3D CENTER OF ROTATION:")
    print(f"  COR (at cor_z={args.cor_z}): ({args.cor_x}, {args.cor_y}, {args.cor_z}) nm")
    cor_offset_500 = sampler.get_cor_x_offset(500, args.cor_z)
    cor_x_at_500 = args.cor_x + cor_offset_500
    cor_offset_1000 = sampler.get_cor_x_offset(1000, args.cor_z)
    cor_x_at_1000 = args.cor_x + cor_offset_1000
    print(f"  Effective COR at Z=500: ({cor_x_at_500:.1f}, {args.cor_y}) nm")
    print(f"  Effective COR at Z=1000: ({cor_x_at_1000:.1f}, {args.cor_y}) nm")
    print(f"  Rotation speed: {args.speed_r} micro-deg/s ({args.speed_r / 1000000:.1f} deg/s)")
    print(f"\n📊 Z-STACK:")
    print(f"  Images: {len(args.images)} files")
    print(f"  Z-planes: {z_positions} nm")
    print(f"\n⚡ SPEEDS:")
    print(f"  X/Y: {args.speed_xy} nm/s")
    print(f"  Z: {args.speed_z} nm/s")
    print(f"  R: {args.speed_r / 1000000:.1f} deg/s")
    print(f"\n📡 MQTT Commands:")
    print(f"  MOVE/<axis>/<value>    - Move axis (X/Y in nm, Z in nm, R in micro-deg)")
    print(f"  SET_COR/<x>/<y>/<z>    - Set 3D center of rotation (nm)")
    print(f"  SET_RATE/<Hz>          - Set broadcast rate")
    print(f"  STATUS                 - Get system status")
    print(f"\n🛑 Press Ctrl+C to stop\n")

    try:
        while running:
            time.sleep(0.5)
    finally:
        bus.stop()
        print("\nSimulator stopped.")


if __name__ == "__main__":
    main()