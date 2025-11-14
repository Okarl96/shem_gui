"""
Utility Functions for Microscope Control

Shared utilities used across multiple modules.
Includes coordinate transformations and compensation calculations.
"""

from typing import Tuple
import math


def apply_z_compensation(
    x_base: float,
    y_base: float,
    z: float,
    z_base: float = 0.0,
    x_ratio: float = 0.0,
    y_ratio: float = 0.0
) -> Tuple[float, float, float]:
    """
    Apply X/Y compensation when moving Z axis.

    Due to 45° beam incidence angle and stage drift, X and Y positions
    need to be adjusted when Z changes to keep the beam on the same
    sample location.

    Compensation formula:
        x_compensated = x_base + (z - z_base) * x_ratio
        y_compensated = y_base + (z - z_base) * y_ratio

    Args:
        x_base: Base X position (nm)
        y_base: Base Y position (nm)
        z: Current Z position (nm)
        z_base: Reference Z position where (x_base, y_base) is valid (nm)
        x_ratio: X movement per Z movement (dimensionless)
                 Typical: 1.0 for 45° beam in X direction
                         0.0 if no X compensation needed
        y_ratio: Y movement per Z movement (dimensionless)
                 Typical: 0.0 for pure X-Z tilt
                         May be non-zero for stage drift

    Returns:
        Tuple of (x_compensated, y_compensated, z)

    Example:
        # 45° beam in X direction, no Y drift
        x, y, z = apply_z_compensation(
            x_base=5000, y_base=5000, z=1000,
            z_base=0, x_ratio=1.0, y_ratio=0.0
        )
        # x = 5000 + 1000*1.0 = 6000 nm
        # y = 5000 + 1000*0.0 = 5000 nm
        # z = 1000 nm
    """
    # Calculate Z offset from base
    z_offset = z - z_base

    # Apply compensation
    x_compensated = x_base + z_offset * x_ratio
    y_compensated = y_base + z_offset * y_ratio

    return (x_compensated, y_compensated, z)


def calculate_z_compensation_ratio(
    beam_angle_deg: float,
    axis: str = 'X'
) -> Tuple[float, float]:
    """
    Calculate X/Y compensation ratios from beam angle.

    For a tilted beam, when Z changes, the X/Y position must be adjusted
    to keep the beam on the same sample spot.

    Args:
        beam_angle_deg: Beam angle from vertical in degrees
                       Typical: 45° for SEM
        axis: Which axis the beam is tilted in ('X' or 'Y')

    Returns:
        Tuple of (x_ratio, y_ratio)

    Example:
        # 45° beam tilted in X direction
        x_ratio, y_ratio = calculate_z_compensation_ratio(45, axis='X')
        # x_ratio = 1.0 (move X same amount as Z)
        # y_ratio = 0.0 (no Y movement)

        # 30° beam tilted in Y direction
        x_ratio, y_ratio = calculate_z_compensation_ratio(30, axis='Y')
        # x_ratio = 0.0
        # y_ratio = 0.577 (tan(30°))
    """
    if axis not in ['X', 'Y']:
        raise ValueError(f"Invalid axis '{axis}'. Must be 'X' or 'Y'")

    # Convert to radians
    angle_rad = math.radians(beam_angle_deg)

    # Compensation ratio = tan(angle)
    # When Z moves by ΔZ, horizontal position must move by ΔZ * tan(angle)
    ratio = math.tan(angle_rad)

    if axis == 'X':
        return (ratio, 0.0)
    else:
        return (0.0, ratio)


def apply_rotation_compensation(
    x: float,
    y: float,
    r_angle_udeg: float,
    cor_x: float,
    cor_y: float
) -> Tuple[float, float]:
    """
    Apply rotation transformation around center of rotation (COR).

    Simple rotation without Z-compensation. Use apply_rotation_with_z_compensation
    for R-series scans at different Z positions.

    Args:
        x, y: Original position (nm)
        r_angle_udeg: Rotation angle (µdeg, counter-clockwise positive)
        cor_x, cor_y: Center of rotation (nm)

    Returns:
        Tuple of (x_rotated, y_rotated)

    Example:
        # Rotate point around center
        x_rot, y_rot = apply_rotation_compensation(
            x=10000, y=0,
            r_angle_udeg=90000,  # 90°
            cor_x=0, cor_y=0
        )
        # Result: (0, 10000) - rotated 90° around origin
    """
    # Convert µdeg to radians (divide by 1000 to get millidegrees, then convert)
    angle_rad = math.radians(r_angle_udeg / 1000.0)

    # Translate to origin
    x_rel = x - cor_x
    y_rel = y - cor_y

    # Rotate (counter-clockwise)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    x_rot = x_rel * cos_a - y_rel * sin_a
    y_rot = x_rel * sin_a + y_rel * cos_a

    # Translate back
    x_final = x_rot + cor_x
    y_final = y_rot + cor_y

    return (x_final, y_final)


def apply_rotation_with_z_compensation(
    x: float,
    y: float,
    r_angle_udeg: float,
    r_base_udeg: float,
    cor_x: float,
    cor_y: float,
    current_z: float,
    cor_base_z: float,
    cor_x_ratio: float = 0.0,
    cor_y_ratio: float = 0.0
) -> Tuple[float, float]:
    """
    Apply rotation transformation around COR with Z-compensation.

    This is the complete COR transformation for R-series scans at different Z heights.
    The COR position itself shifts with Z due to tilted beam geometry.

    Adopted from old scanner logic:
    1. Calculate Z-compensated COR position
    2. Apply rotation relative to base R position around compensated COR

    Args:
        x, y: Original position (nm)
        r_angle_udeg: Current rotation angle (µdeg)
        r_base_udeg: Base rotation angle (µdeg) - rotation is relative to this
        cor_x, cor_y: Center of rotation at base Z (nm)
        current_z: Current Z position (nm)
        cor_base_z: Z position where COR coordinates are valid (nm)
        cor_x_ratio: COR X movement per Z movement (typically 1.0 for 45° beam)
        cor_y_ratio: COR Y movement per Z movement (typically 0.0)

    Returns:
        Tuple of (x_rotated, y_rotated)

    Example:
        # Rotate at Z=1000nm with COR compensation
        x_rot, y_rot = apply_rotation_with_z_compensation(
            x=5000, y=5000,
            r_angle_udeg=90000,  # 90°
            r_base_udeg=0,
            cor_x=0, cor_y=0,
            current_z=1000, cor_base_z=0,
            cor_x_ratio=1.0, cor_y_ratio=0.0
        )
        # COR shifts to (1000, 0) due to Z compensation
        # Then rotation applied around this shifted COR
    """
    # Step 1: Calculate Z-compensated COR position
    z_offset = current_z - cor_base_z

    # Apply Z compensation to COR (enabled if ratio != 0)
    if abs(cor_x_ratio) > 1e-9:
        cor_x_actual = cor_x + z_offset * cor_x_ratio
    else:
        cor_x_actual = cor_x

    if abs(cor_y_ratio) > 1e-9:
        cor_y_actual = cor_y + z_offset * cor_y_ratio
    else:
        cor_y_actual = cor_y

    # Step 2: Calculate rotation RELATIVE to base position (like Z-series does)
    r_offset_udeg = r_angle_udeg - r_base_udeg

    # Convert µdeg to radians (divide by 1e6 to get degrees, then convert)
    theta = math.radians(r_offset_udeg / 1e6)

    # Step 3: Translate to Z-compensated COR origin
    x_rel = x - cor_x_actual
    y_rel = y - cor_y_actual

    # Step 4: Apply rotation matrix (clockwise, matching old scanner)
    x_new = x_rel * math.cos(theta) + y_rel * math.sin(theta)
    y_new = -x_rel * math.sin(theta) + y_rel * math.cos(theta)

    # Step 5: Translate back
    return (x_new + cor_x_actual, y_new + cor_y_actual)


def transform_scan_area_aabb(
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    r_angle_udeg: float,
    r_base_udeg: float,
    cor_x: float,
    cor_y: float,
    current_z: float = None,
    cor_base_z: float = 0.0,
    cor_x_ratio: float = 0.0,
    cor_y_ratio: float = 0.0
) -> Tuple[float, float, float, float]:
    """
    Transform scan area using AABB (Axis-Aligned Bounding Box) method.

    This is the "legacy" mode from old scanner:
    - Transform all four corners
    - Find bounding box of transformed corners
    - Results in expanding scan area as rotation increases

    Args:
        x_start, x_end: X bounds of original scan area (nm)
        y_start, y_end: Y bounds of original scan area (nm)
        r_angle_udeg: Current rotation angle (µdeg)
        r_base_udeg: Base rotation angle (µdeg)
        cor_x, cor_y: Center of rotation at base Z (nm)
        current_z: Current Z position (nm)
        cor_base_z: Z where COR is valid (nm)
        cor_x_ratio, cor_y_ratio: COR Z-compensation ratios

    Returns:
        Tuple of (x_start_new, x_end_new, y_start_new, y_end_new)
    """
    # Transform all four corners
    corners = [
        (x_start, y_start),
        (x_start, y_end),
        (x_end, y_start),
        (x_end, y_end)
    ]

    transformed = []
    for x, y in corners:
        x_t, y_t = apply_rotation_with_z_compensation(
            x, y, r_angle_udeg, r_base_udeg,
            cor_x, cor_y, current_z or cor_base_z, cor_base_z,
            cor_x_ratio, cor_y_ratio
        )
        transformed.append((x_t, y_t))

    # Find new bounding box
    x_coords = [x for x, y in transformed]
    y_coords = [y for x, y in transformed]

    return (min(x_coords), max(x_coords), min(y_coords), max(y_coords))


def transform_scan_area_center_rotate(
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    r_angle_udeg: float,
    r_base_udeg: float,
    cor_x: float,
    cor_y: float,
    current_z: float = None,
    cor_base_z: float = 0.0,
    cor_x_ratio: float = 0.0,
    cor_y_ratio: float = 0.0
) -> Tuple[float, float, float, float]:
    """
    Transform scan area using center rotation method.

    This is the "new" preferred mode:
    - Rotate the CENTER of the scan area
    - Keep the scan SIZE fixed
    - More intuitive behavior - scan area doesn't expand

    Args:
        x_start, x_end: X bounds of original scan area (nm)
        y_start, y_end: Y bounds of original scan area (nm)
        r_angle_udeg: Current rotation angle (µdeg)
        r_base_udeg: Base rotation angle (µdeg)
        cor_x, cor_y: Center of rotation at base Z (nm)
        current_z: Current Z position (nm)
        cor_base_z: Z where COR is valid (nm)
        cor_x_ratio, cor_y_ratio: COR Z-compensation ratios

    Returns:
        Tuple of (x_start_new, x_end_new, y_start_new, y_end_new)
    """
    # Calculate center of scan area
    x_center = (x_start + x_end) / 2
    y_center = (y_start + y_end) / 2

    # Calculate size
    width = abs(x_end - x_start)
    height = abs(y_end - y_start)

    # Rotate center point
    x_center_rot, y_center_rot = apply_rotation_with_z_compensation(
        x_center, y_center, r_angle_udeg, r_base_udeg,
        cor_x, cor_y, current_z or cor_base_z, cor_base_z,
        cor_x_ratio, cor_y_ratio
    )

    # Calculate new bounds maintaining size
    x_start_new = x_center_rot - width / 2
    x_end_new = x_center_rot + width / 2
    y_start_new = y_center_rot - height / 2
    y_end_new = y_center_rot + height / 2

    return (x_start_new, x_end_new, y_start_new, y_end_new)