"""
scan_patterns.py

Pure geometric / ordering logic for microscope scan planning.

This module DOES:
  - Define WHERE to measure (grid / line / z-trace positions)
  - Define the ORDER to visit those positions (raster, snake, etc.)

This module DOES NOT:
  - Talk to hardware
  - Do settling/dwelling/acquisition
  - Handle series scans (Z stacks, R stacks, etc.) â€“ that's higher level

OUTPUT FORMAT CONTRACT (IMPORTANT):
Every function in this file returns a List[Dict].

For 2D scan areas:
    {'X': <nm>, 'Y': <nm>, 'idx': <int>}

For 1D line scans:
    {'X': <nm>, 'idx': <int>}
 or {'Y': <nm>, 'idx': <int>}
(depending on which axis you scanned)

For Z scans:
    {'X': <nm>, 'Y': <nm>, 'Z': <nm>, 'idx': <int>}
where X and Y are ALREADY compensated for Z motion (tilt/drift correction).

Movement pattern functions will reorder points and rewrite 'idx'
so that 'idx' always reflects traversal order.

All distances are in nm. Rotation is not handled here.
"""

from typing import List, Tuple, Dict, Callable
import math
from typing import List, Dict, Optional
from .utils import apply_z_compensation


# =====================================================================
# GROUP 1: AREA / PATH GENERATORS
# =====================================================================

def generate_2d_rectangular_grid(
    x_min_nm: float,
    x_max_nm: float,
    x_pixels: int,
    y_min_nm: float,
    y_max_nm: float,
    y_pixels: int,
) -> List[Dict[str, float]]:
    """
    Generate a regular 2D grid over a rectangle.

    Args:
        x_min_nm, x_max_nm: bounds of X (nm)
        x_pixels: number of samples in X (>=1)
        y_min_nm, y_max_nm: bounds of Y (nm)
        y_pixels: number of samples in Y (>=1)

    Returns:
        points: List of dicts
            {'X': x_nm, 'Y': y_nm, 'idx': local_index}
        The order here is row-major: Y outer loop, X inner loop.
        This is NOT necessarily the final scan order. Use a movement pattern
        (apply_raster_pattern_2d / apply_snake_pattern_2d) to get traversal order.
    """
    if x_pixels < 1 or y_pixels < 1:
        raise ValueError("x_pixels and y_pixels must be >= 1")

    if x_pixels == 1:
        x_list = [0.5 * (x_min_nm + x_max_nm)]
    else:
        step_x = (x_max_nm - x_min_nm) / (x_pixels - 1)
        x_list = [x_min_nm + i * step_x for i in range(x_pixels)]

    if y_pixels == 1:
        y_list = [0.5 * (y_min_nm + y_max_nm)]
    else:
        step_y = (y_max_nm - y_min_nm) / (y_pixels - 1)
        y_list = [y_min_nm + j * step_y for j in range(y_pixels)]

    points: List[Dict[str, float]] = []
    idx_counter = 0
    for y_nm in y_list:
        for x_nm in x_list:
            points.append({
                'X': x_nm,
                'Y': y_nm,
                'idx': idx_counter,
            })
            idx_counter += 1

    return points


def generate_2d_custom_grid(
    vertices_nm: List[Tuple[float, float]],
    x_pixels: int,
    y_pixels: int,
) -> List[Dict[str, float]]:
    """
    Generate a grid of points INSIDE a polygon ROI.

    Steps:
      1. Compute polygon bounding box.
      2. Lay down a regular X/Y grid with the requested pixel counts.
      3. Keep only the points whose (X,Y) are inside the polygon.

    Args:
        vertices_nm:
            List of (x_nm, y_nm) polygon vertices (>=3 points).
        x_pixels:
            number of samples along X in the bounding box
        y_pixels:
            number of samples along Y in the bounding box

    Returns:
        points: List of dicts
            {'X': x_nm, 'Y': y_nm, 'idx': local_index}
        Order is row-major over the bounding box, filtered by inclusion.
        Use apply_raster_pattern_2d / apply_snake_pattern_2d to define traversal.
    """
    if len(vertices_nm) < 3:
        raise ValueError("Need at least 3 vertices for polygon")

    xs = [v[0] for v in vertices_nm]
    ys = [v[1] for v in vertices_nm]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    if x_pixels < 1 or y_pixels < 1:
        raise ValueError("x_pixels and y_pixels must be >= 1")

    if x_pixels == 1:
        x_list = [0.5 * (x_min + x_max)]
    else:
        step_x = (x_max - x_min) / (x_pixels - 1)
        x_list = [x_min + i * step_x for i in range(x_pixels)]

    if y_pixels == 1:
        y_list = [0.5 * (y_min + y_max)]
    else:
        step_y = (y_max - y_min) / (y_pixels - 1)
        y_list = [y_min + j * step_y for j in range(y_pixels)]

    pts: List[Dict[str, float]] = []
    idx_counter = 0
    for row_idx, y_nm in enumerate(y_list):
        for col_idx, x_nm in enumerate(x_list):
            if _point_in_polygon(x_nm, y_nm, vertices_nm):
                pts.append({
                    'X': x_nm,
                    'Y': y_nm,
                    'idx': idx_counter,
                    'row': row_idx,  # Row in bounding box grid
                    'col': col_idx,  # Col in bounding box grid
                })
                idx_counter += 1

    return pts


def _point_in_polygon(
    x_nm: float,
    y_nm: float,
    vertices_nm: List[Tuple[float, float]],
) -> bool:
    """
    Even-odd / ray casting polygon inclusion test.
    """
    inside = False
    n = len(vertices_nm)
    for i in range(n):
        x0, y0 = vertices_nm[i]
        x1, y1 = vertices_nm[(i + 1) % n]

        # Check if ray from (x_nm,y_nm) horizontally right crosses the edge
        cond_y = ((y0 > y_nm) != (y1 > y_nm))
        if cond_y:
            x_int = x0 + (y_nm - y0) * (x1 - x0) / (y1 - y0 + 1e-20)
            if x_int >= x_nm:
                inside = not inside
    return inside


def generate_1d_line(
    fixed_axis: str,
    fixed_value_nm: float,
    scan_axis: str,
    start_nm: float,
    end_nm: float,
    num_points: int,
) -> List[Dict[str, float]]:
    """
    Generate a 1D line scan in XY.

    Rules:
    - One axis (X or Y) is held constant.
    - The other axis (X or Y) is scanned from start_nm -> end_nm.
    - Z and R are NOT included here. Those will be added later by the runner.

    Args:
        fixed_axis:
            'X' or 'Y' (the axis that stays fixed)
        fixed_value_nm:
            value (nm) of that fixed axis
            (not included in output dicts; the runner will add it)
        scan_axis:
            'X' or 'Y' (the axis that moves)
            must be the other one
        start_nm:
            starting coordinate (nm) for the scan axis
        end_nm:
            ending coordinate (nm) for the scan axis
        num_points:
            number of samples along the line (>=1)

    Returns:
        line_points: List of dicts
            [{'X': 1234.0, 'idx': 0}, {'X': 1244.0, 'idx': 1}, ...]
        or
            [{'Y': 777.0, 'idx': 0}, {'Y': 780.0, 'idx': 1}, ...]

        Only the SCANNED axis is present (plus idx).
        The caller knows "oh, X is scanned and Y=const" because it called this
        with fixed_axis and scan_axis.
    """
    if fixed_axis not in ("X", "Y"):
        raise ValueError("fixed_axis must be 'X' or 'Y'")
    if scan_axis not in ("X", "Y"):
        raise ValueError("scan_axis must be 'X' or 'Y'")
    if scan_axis == fixed_axis:
        raise ValueError("scan_axis must be different from fixed_axis")

    if num_points < 1:
        raise ValueError("num_points must be >= 1")

    if num_points == 1:
        scan_positions = [0.5 * (start_nm + end_nm)]
    else:
        step = (end_nm - start_nm) / (num_points - 1)
        scan_positions = [start_nm + i * step for i in range(num_points)]

    results: List[Dict[str, float]] = []
    for idx, val in enumerate(scan_positions):
        results.append({
            scan_axis: val,
            'idx': idx,
        })

    return results




def generate_z_scan(
    x_nominal_nm: float,
    y_nominal_nm: float,
    z_start_nm: float,
    z_end_nm: float,
    num_points: int,
    *,
    z_ref_nm: float,
    x_ratio: float,
    y_ratio: float,
) -> List[Dict[str, float]]:
    """
    Generate a compensated Z scan path.

    We assume:
    - You conceptually "hold" X = x_nominal_nm, Y = y_nominal_nm.
    - But because of 45Â° incidence, tilt, drift, etc., the true stage
      position you must command changes in X and Y as you change Z.

    We enforce compensation here, always. No uncompensated Z scan exists
    at this layer.

    The compensation model is provided indirectly via x_ratio and y_ratio
    and implemented in utils.apply_z_compensation().

    Args:
        x_nominal_nm:
            Nominal X at the reference plane (nm).
        y_nominal_nm:
            Nominal Y at the reference plane (nm).
        z_start_nm:
            First Z in the scan (nm).
        z_end_nm:
            Last Z in the scan (nm).
        num_points:
            Number of Z samples (>=1). We sample uniformly from startâ†’end.
        z_ref_nm:
            The reference Z plane (nm). At this Z, (x_nominal_nm, y_nominal_nm)
            are considered correct / calibrated.
        x_ratio:
            nm of X lateral shift per 1 nm of (Z - z_ref_nm).
        y_ratio:
            nm of Y lateral shift per 1 nm of (Z - z_ref_nm).

    Returns:
        A list of dicts, each of which is directly usable by the runner:
            {
                'X': <compensated X_nm>,
                'Y': <compensated Y_nm>,
                'Z': <target Z_nm>,
                'idx': <visit index>
            }
    """
    if num_points < 1:
        raise ValueError("num_points must be >= 1")

    # Build the Z path (uniform spacing)
    if num_points == 1:
        z_values = [0.5 * (z_start_nm + z_end_nm)]
    else:
        step_z = (z_end_nm - z_start_nm) / (num_points - 1)
        z_values = [z_start_nm + i * step_z for i in range(num_points)]

    result: List[Dict[str, float]] = []
    for idx, z_nm in enumerate(z_values):
        x_corr_nm, y_corr_nm, z_final = apply_z_compensation(
            x_base=x_nominal_nm,  # FIXED: correct parameter name
            y_base=y_nominal_nm,  # FIXED: correct parameter name
            z=z_nm,  # FIXED: correct parameter name
            z_base=z_ref_nm,  # FIXED: correct parameter name
            x_ratio=x_ratio,
            y_ratio=y_ratio,
        )

        result.append({
            'X': x_corr_nm,
            'Y': y_corr_nm,
            'Z': z_final,  # FIXED: use returned z value
            'idx': idx,
        })

    return result


def generate_multi_z_scan(
        xy_points: List[Tuple[float, float]],
        z_positions: List[float],
        z_compensation: Optional[Dict[str, float]] = None
) -> List[Dict[str, float]]:
    """
    Generate Multi-Z scan pattern: Visit multiple (X,Y) points at each Z height.

    Order: Z-outer loop, XY-inner loop
    - Visit all XY points at Z[0]
    - Move to Z[1]
    - Visit all XY points at Z[1]
    - ...

    Args:
        xy_points: List of (x, y) coordinates in nm [(x1,y1), (x2,y2), ...]
        z_positions: List of Z heights in nm [z0, z1, z2, ...]
        z_compensation: Optional dict with keys:
                       - 'z_ref': Reference Z position (nm)
                       - 'x_ratio': X compensation ratio
                       - 'y_ratio': Y compensation ratio

    Returns:
        List of pattern dictionaries
    """
    pattern = []
    idx_counter = 0

    # Check if compensation should be applied
    use_compensation = (z_compensation is not None and
                        (z_compensation.get('x_ratio', 0) != 0 or
                         z_compensation.get('y_ratio', 0) != 0))

    # Z-outer loop: visit all XY at each Z
    for z_idx, z_pos in enumerate(z_positions):
        for xy_idx, (x_base, y_base) in enumerate(xy_points):

            # Apply Z-compensation if enabled
            if use_compensation:
                x_comp, y_comp, z_comp = apply_z_compensation(
                    x_base=x_base,
                    y_base=y_base,
                    z=z_pos,
                    z_base=z_compensation.get('z_ref', 0),
                    x_ratio=z_compensation.get('x_ratio', 0),
                    y_ratio=z_compensation.get('y_ratio', 0)
                )
            else:
                x_comp, y_comp, z_comp = x_base, y_base, z_pos

            pattern.append({
                'X': x_comp,
                'Y': y_comp,
                'Z': z_comp,
                'z_idx': z_idx,
                'xy_idx': xy_idx,
                'idx': idx_counter
            })
            idx_counter += 1

    return pattern


# =====================================================================
# GROUP 2: MOVEMENT PATTERNS / VISIT ORDER
# =====================================================================

def apply_raster_pattern_2d(
    points_2d: List[Dict[str, float]],
    fast_axis: str = 'X'
) -> List[Dict[str, float]]:
    """
    Raster traversal for 2D scans.

    The fast_axis scans continuously at each position of the slow axis.

    Example with fast_axis='X':
        Row Y=0: X: 0â†’1â†’2â†’3...
        Row Y=1: X: 0â†’1â†’2â†’3...

    Example with fast_axis='Y':
        Column X=0: Y: 0â†’1â†’2â†’3...
        Column X=1: Y: 0â†’1â†’2â†’3...

    Args:
        points_2d:
            Output of generate_2d_rectangular_grid() or generate_2d_custom_grid(),
            each element like {'X':..., 'Y':..., 'idx':...}
        fast_axis:
            'X' or 'Y' - which axis to scan continuously (inner loop)

    Returns:
        ordered list of dicts:
            {'X':..., 'Y':..., 'idx': visit_index}
    """
    if fast_axis not in ('X', 'Y'):
        raise ValueError("fast_axis must be 'X' or 'Y'")

    slow_axis = 'Y' if fast_axis == 'X' else 'X'

    # Sort: slow axis outer, fast axis inner
    sorted_points = sorted(points_2d, key=lambda p: (p[slow_axis], p[fast_axis]))

    # Rewrite idx to reflect traversal order
    ordered: List[Dict[str, float]] = []
    for new_idx, pt in enumerate(sorted_points):
        new_pt = dict(pt)  # Preserve extra fields like row/col
        new_pt['idx'] = new_idx  # Overwrite idx with visit order
        ordered.append(new_pt)

    return ordered


def apply_snake_pattern_2d(
    points_2d: List[Dict[str, float]],
    fast_axis: str = 'X'
) -> List[Dict[str, float]]:
    """
    Snake / serpentine traversal for 2D scans.

    The fast_axis alternates direction at each position of the slow axis.

    Example with fast_axis='X':
        Row Y=0: X: 0â†’1â†’2â†’3...  â†’
        Row Y=1: X: 3â†’2â†’1â†’0...  â†
        Row Y=2: X: 0â†’1â†’2â†’3...  â†’

    Example with fast_axis='Y':
        Column X=0: Y: 0â†’1â†’2â†’3...  â†’
        Column X=1: Y: 3â†’2â†’1â†’0...  â†
        Column X=2: Y: 0â†’1â†’2â†’3...  â†’

    Args:
        points_2d:
            [{'X':..., 'Y':..., 'idx':...}, ...]
        fast_axis:
            'X' or 'Y' - which axis alternates direction (inner loop)

    Returns:
        ordered path:
            [{'X':..., 'Y':..., 'idx': visit_index}, ...]
    """
    if fast_axis not in ('X', 'Y'):
        raise ValueError("fast_axis must be 'X' or 'Y'")

    slow_axis = 'Y' if fast_axis == 'X' else 'X'

    # 1. Bucket points by the slow axis value
    rows_or_cols: Dict[float, List[Dict[str, float]]] = {}
    for pt in points_2d:
        slow_val = pt[slow_axis]
        rows_or_cols.setdefault(slow_val, []).append(pt)

    # 2. Process in ascending slow axis order
    snake_order: List[Dict[str, float]] = []
    for line_idx, slow_val in enumerate(sorted(rows_or_cols.keys())):
        line_pts = rows_or_cols[slow_val]

        # Sort by fast axis ascending
        line_pts_sorted = sorted(line_pts, key=lambda p: p[fast_axis])

        # Flip every other line
        if line_idx % 2 == 1:
            line_pts_sorted = list(reversed(line_pts_sorted))

        snake_order.extend(line_pts_sorted)

    # 3. Rewrite idx to reflect traversal order
    ordered: List[Dict[str, float]] = []
    for new_idx, pt in enumerate(snake_order):
        new_pt = dict(pt)  # Preserve extra fields like row/col
        new_pt['idx'] = new_idx  # Overwrite idx with visit order
        ordered.append(new_pt)

    return ordered


def apply_single_direction_1d(
    line_pts: List[Dict[str, float]]
) -> List[Dict[str, float]]:
    """
    Single-direction traversal of a 1D path (forward only).

    Works with:
        - generate_1d_line() output:
            [{'X':..., 'idx':...}] or [{'Y':..., 'idx':...}]
        - generate_z_scan() output:
            [{'X':..., 'Y':..., 'Z':..., 'idx':...}]

    We:
        1. sort by original idx ascending
        2. rewrite idx = visit order

    Returns:
        new list of dicts with the SAME coordinate keys,
        but idx replaced by the new visit order (0..N-1).
    """
    # sort by original idx
    sorted_line = sorted(line_pts, key=lambda p: p['idx'])

    ordered: List[Dict[str, float]] = []
    for new_idx, pt in enumerate(sorted_line):
        # copy all coord keys except idx, then reassign idx
        new_pt = {k: v for k, v in pt.items() if k != 'idx'}
        new_pt['idx'] = new_idx
        ordered.append(new_pt)

    return ordered


def apply_bidirectional_1d(
    line_pts: List[Dict[str, float]],
    offset_fraction: float = 0.5,
) -> List[Dict[str, float]]:
    """
    Bidirectional traversal for 1D XY line scans.

    Forward pass:
        - Visit the provided points in ascending original idx.
        - Dwell on those exact coordinates.

    Backward pass:
        - Walk back toward the start,
        - But dwell at NEW offset positions between the original points,
          not the same positions as forward.
        - These offset positions are computed as midpoints (or any fraction)
          along the scanned axis.

    Notes:
        - This is meant for XY line scans, not Z scans.
        - We assume only ONE axis changes across the line ('X' or 'Y').
        - Output is a flat list of dicts with that axis plus 'idx'.
          (No direction flag; the output order already encodes forward+back.)

    Args:
        line_pts:
            Output of generate_1d_line().
            Each element looks like {'X': some_value, 'idx': ...}
            or {'Y': some_value, 'idx': ...}
        offset_fraction:
            Fraction along the segment to use for backtrack dwell.
            0.5 â†’ exact midpoint.

    Returns:
        path: [{'X':..., 'idx':...}, ...] or [{'Y':..., 'idx':...}, ...]
        where 'idx' is the visit order in this full forward+back sequence.
    """
    if not line_pts:
        return []

    # 1. sort by original idx to get forward order
    forward_sorted = sorted(line_pts, key=lambda p: p['idx'])

    if len(forward_sorted) == 1:
        # Only one point: nothing to interpolate
        return [{'idx': 0, **{k: v for k, v in forward_sorted[0].items() if k != 'idx'}}]

    # 2. detect which axis is scanning (the key other than 'idx')
    first_pt = forward_sorted[0]
    scan_axes = [k for k in first_pt.keys() if k != 'idx']
    if len(scan_axes) != 1:
        raise ValueError("Bidirectional 1D expects exactly one varying axis (X or Y).")
    scan_axis = scan_axes[0]

    # 3. Forward pass points
    path_forward = []
    for pt in forward_sorted:
        coord_val = pt[scan_axis]
        path_forward.append({scan_axis: coord_val})

    # 4. Backward pass offset points:
    #    for each adjacent pair (hi, lo) going backward,
    #    generate an in-between value.
    path_backward = []
    for i in range(len(forward_sorted) - 1, 0, -1):
        hi = forward_sorted[i][scan_axis]
        lo = forward_sorted[i - 1][scan_axis]
        mid_val = lo + offset_fraction * (hi - lo)
        path_backward.append({scan_axis: mid_val})

    # 5. Combine forward + backward, and assign new idx sequence
    combined = path_forward + path_backward
    final: List[Dict[str, float]] = []
    for new_idx, pt in enumerate(combined):
        final.append({
            scan_axis: pt[scan_axis],
            'idx': new_idx,
        })

    return final


# =====================================================================
# UTILITY HELPERS
# =====================================================================

def get_pattern_info_2d(
    pattern_2d: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Summarize a 2D scan plan.

    Args:
        pattern_2d:
            [{'X':..., 'Y':..., 'idx':...}, ...]

    Returns:
        {
            'num_points': N,
            'x_range': (xmin, xmax),
            'y_range': (ymin, ymax)
        }
    """
    if not pattern_2d:
        return {
            'num_points': 0,
            'x_range': (0.0, 0.0),
            'y_range': (0.0, 0.0),
        }

    xs = [p['X'] for p in pattern_2d]
    ys = [p['Y'] for p in pattern_2d]

    return {
        'num_points': len(pattern_2d),
        'x_range': (min(xs), max(xs)),
        'y_range': (min(ys), max(ys)),
    }