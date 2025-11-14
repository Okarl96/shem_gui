"""
Acquisition Primitives for Microscope Control

Low-level primitives that operate purely on reader state.
Higher-level code handles: movement commands, scan patterns, data aggregation.

This module provides:
1. check_settled()          - Wait until stage position is stable at target
2. collect_dwell_samples()  - Collect raw detector samples during dwell window
3. calculate_statistics()   - Average raw samples into a single data point
4. acquire_point()          - Convenience wrapper that runs all three

IMPORTANT TIMING RULE:
- We NEVER compare device timestamps (from hardware) to system wall time.
- We ONLY compare device timestamps to other device timestamps.
- We ONLY compare wall clock time to wall clock time.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import time


# ======================== Data Structures ======================== #

@dataclass
class SettleResult:
    """
    Result from settling check.
    
    settled:
        True if position stabilized within tolerance for required_consecutive reads.
    position:
        Latest stage position dict:
        {
            'X_nm': ...,
            'Y_nm': ...,
            'Z_nm': ...,
            'R_udeg': ...,
            't': device_timestamp_seconds
        }
        Note: values may be None if hardware didn't report them.
    settle_time:
        How long settling took in system time (seconds). For operator info only.
    consecutive_count:
        How many consecutive in-tolerance readings we achieved on exit.
    timeout:
        True if we gave up due to system-time timeout.
    """
    settled: bool
    position: Dict[str, Optional[float]]
    settle_time: float
    consecutive_count: int
    timeout: bool


@dataclass
class DwellSamples:
    """
    Raw samples collected during dwell period.
    
    samples:
        List of (t_device, value_na) pairs for each valid detector sample.
        t_device is the device timestamp in seconds from the hardware.
    t_start:
        Device timestamp of first valid sample.
    t_end:
        Device timestamp of last valid sample.
    n_samples:
        Number of valid samples.
    dwell_time:
        t_end - t_start in device time (None if we got <1 sample).
    """
    samples: List[Tuple[float, float]]
    t_start: Optional[float]
    t_end: Optional[float]
    n_samples: int
    dwell_time: Optional[float]


@dataclass
class DataPoint:
    """
    Averaged measurement at a single location.
    
    position:
        Stage position dict at settle time
        (same shape as in SettleResult.position).
    avg_signal:
        Mean signal value (nA) of the dwell samples.
    std_signal:
        Sample standard deviation (nA).
    min_signal:
        Minimum signal value (nA).
    max_signal:
        Maximum signal value (nA).
    n_samples:
        Number of samples contributing.
    t_start:
        First device timestamp in dwell.
    t_end:
        Last device timestamp in dwell.
    """
    position: Dict[str, Optional[float]]
    avg_signal: Optional[float]
    std_signal: Optional[float]
    min_signal: Optional[float]
    max_signal: Optional[float]
    n_samples: int
    t_start: Optional[float]
    t_end: Optional[float]


# ====================== Primitive Functions ====================== #

def check_settled(
    reader,
    target: Dict[str, float],
    tolerance: Dict[str, float],
    required_consecutive: int = 3,
    timeout: float = 5.0,
    check_interval: float = 0.02,
) -> SettleResult:
    """
    Wait until stage position is stable at the target.

    This DOES NOT send any movement commands. Higher-level code must
    have already commanded a move via CommandSender before calling this.

    The logic:
    - Repeatedly read reader.get_xyzr()
    - For each axis in `target`, check if |current - target| <= tolerance[axis]
    - Track how many good reads in a row
    - Stop when we hit `required_consecutive`
    - Stop with timeout if system wall time exceeds `timeout`

    NOTE ON TIMING:
    - We use system wall time ONLY for the timeout and for reporting settle_time.
    - We do NOT try to infer "freshness" by comparing device timestamp to wall time.
      That would violate the rule: device time must be compared only to device time.

    Args:
        reader:
            PositionSignalReader
        target:
            {'X': 5000, 'Y': 5000, 'Z': 1000, 'R': 0}
            Units:
                X/Y/Z in nm
                R in µdeg
        tolerance:
            {'X': 5, 'Y': 5, 'Z': 5, 'R': 1000}
            Same units as target
        required_consecutive:
            Number of consecutive "in tolerance" reads required to declare settled
        timeout:
            Safety timeout in SECONDS (system wall clock)
        check_interval:
            Sleep between reads (system sleep) to avoid busy-wait

    Returns:
        SettleResult
    """
    start_wall = time.time()
    consecutive_count = 0
    last_position = {}

    while True:
        elapsed_wall = time.time() - start_wall

        # Timeout check (system time vs system time)
        if elapsed_wall >= timeout:
            return SettleResult(
                settled=False,
                position=last_position,
                settle_time=elapsed_wall,
                consecutive_count=consecutive_count,
                timeout=True,
            )

        pos = reader.get_xyzr()

        # Build a position dict we will return / report
        last_position = {
            'X_nm': pos['X_nm'],
            'Y_nm': pos['Y_nm'],
            'Z_nm': pos['Z_nm'],
            'R_udeg': pos['R_udeg'],
            't': pos['t'],   # device timestamp [s]
        }

        # Check all requested axes are within tolerance
        all_ok = True
        for axis, target_val in target.items():
            if axis == 'X':
                current = pos['X_nm']
            elif axis == 'Y':
                current = pos['Y_nm']
            elif axis == 'Z':
                current = pos['Z_nm']
            elif axis == 'R':
                current = pos['R_udeg']
            else:
                # Unknown axis label -> ignore it entirely
                continue

            if current is None:
                all_ok = False
                break

            tol = tolerance.get(axis, float('inf'))
            err = abs(current - target_val)
            if err > tol:
                all_ok = False
                break

        # Update consecutive good count
        if all_ok:
            consecutive_count += 1
        else:
            consecutive_count = 0

        # If we've been good long enough, we're done
        if consecutive_count >= required_consecutive:
            return SettleResult(
                settled=True,
                position=last_position,
                settle_time=time.time() - start_wall,  # system time duration
                consecutive_count=consecutive_count,
                timeout=False,
            )

        time.sleep(check_interval)


def collect_dwell_samples(
    reader,
    detector_lag: float = 0.0,
    dwell_time: float = 1.0,
    sample_interval: float = 0.025,
) -> DwellSamples:
    """
    Collect raw detector samples over a dwell window.

    This assumes the stage is already stable. This function:
    1. Sleeps detector_lag (system sleep) to let detector settle mechanically/electronically.
    2. Repeatedly reads reader.get_signal().
    3. Keeps pairs (t_device, value_na) where both are valid.
    4. Stops when (t_last_device - t_first_device) >= dwell_time, using DEVICE TIMESTAMPS ONLY.

    NOTE ON TIMING:
    - We do NOT try to compare device timestamps against system wall time.
    - dwell_time is enforced in device time.
    - We still use time.sleep(sample_interval) to avoid spinning the CPU,
      but that does not enter into the logic for "have we dwelled long enough?"

    Args:
        reader:
            PositionSignalReader
        detector_lag:
            Seconds (system sleep) to wait before starting measurement.
            This is purely a physical-settling delay.
        dwell_time:
            Desired integration window length in SECONDS (device timestamp space).
        sample_interval:
            System sleep between samples.

    Returns:
        DwellSamples
    """

    # Let detector settle
    if detector_lag > 0:
        time.sleep(detector_lag)

    samples: List[Tuple[float, float]] = []
    t_first: Optional[float] = None
    t_last: Optional[float] = None

    while True:
        sig = reader.get_signal()
        value = sig['value_na']
        t_dev = sig['t']  # device timestamp in seconds

        # Accept only fully valid samples
        if value is not None and t_dev is not None:
            samples.append((t_dev, value))

            # Initialize t_first on first good sample
            if t_first is None:
                t_first = t_dev
            t_last = t_dev

        # Check if we've covered the requested dwell window in device time
        if t_first is not None and t_last is not None:
            if (t_last - t_first) >= dwell_time:
                break

        time.sleep(sample_interval)

    # Compute actual dwell duration from device times
    actual_dwell = None
    if t_first is not None and t_last is not None:
        actual_dwell = t_last - t_first

    return DwellSamples(
        samples=samples,
        t_start=t_first,
        t_end=t_last,
        n_samples=len(samples),
        dwell_time=actual_dwell,
    )


def calculate_statistics(
    position: Dict[str, Optional[float]],
    dwell_samples: DwellSamples
) -> DataPoint:
    """
    Reduce raw dwell samples into a single averaged measurement.

    Args:
        position:
            Stage position dict (the 'position' from SettleResult).
        dwell_samples:
            Result of collect_dwell_samples().

    Returns:
        DataPoint
    """
    if not dwell_samples.samples:
        return DataPoint(
            position=position,
            avg_signal=None,
            std_signal=None,
            min_signal=None,
            max_signal=None,
            n_samples=0,
            t_start=dwell_samples.t_start,
            t_end=dwell_samples.t_end,
        )

    # Extract just the values (nA)
    values = [val for (_, val) in dwell_samples.samples]
    n = len(values)

    avg = sum(values) / n

    if n > 1:
        var = sum((x - avg) ** 2 for x in values) / (n - 1)
        std = var ** 0.5
    else:
        std = 0.0

    return DataPoint(
        position=position,
        avg_signal=avg,
        std_signal=std,
        min_signal=min(values),
        max_signal=max(values),
        n_samples=n,
        t_start=dwell_samples.t_start,
        t_end=dwell_samples.t_end,
    )


def acquire_point(
    reader,
    target: Dict[str, float],
    tolerance: Dict[str, float],
    settle_params: Optional[Dict] = None,
    dwell_params: Optional[Dict] = None,
) -> DataPoint:
    """
    High-level convenience:
        1. check_settled(...)
        2. collect_dwell_samples(...)
        3. calculate_statistics(...)

    NOTE:
    - This still assumes higher-level code *already moved the stage*
      to near `target`. We do NOT send motion commands here.

    Args:
        reader:
            PositionSignalReader
        target:
            {'X': ..., 'Y': ..., 'Z': ..., 'R': ...}
        tolerance:
            {'X': ..., 'Y': ..., 'Z': ..., 'R': ...}
        settle_params:
            Optional dict to override defaults for check_settled()
            e.g. {
                'required_consecutive': 5,
                'timeout': 10.0,
                'check_interval': 0.01,
            }
        dwell_params:
            Optional dict to override defaults for collect_dwell_samples()
            e.g. {
                'detector_lag': 0.2,
                'dwell_time': 0.05,
                'sample_interval': 0.005,
            }

    Returns:
        DataPoint

    Raises:
        RuntimeError: If settling fails (timeout)
    """

    # Default parameters for settling
    settle_defaults = {
        'required_consecutive': 3,
        'timeout': 5.0,
        'check_interval': 0.02,
    }
    if settle_params:
        settle_defaults.update(settle_params)

    # Default parameters for dwelling
    dwell_defaults = {
        'detector_lag': 0.0,
        'dwell_time': 1.0,
        'sample_interval': 0.025,
    }
    if dwell_params:
        dwell_defaults.update(dwell_params)

    # Step 1: verify we're stably at target
    settle_result = check_settled(
        reader=reader,
        target=target,
        tolerance=tolerance,
        **settle_defaults,
    )

    if not settle_result.settled:
        raise RuntimeError(
            f"Failed to settle at target after {settle_result.settle_time:.2f}s "
            f"(consecutive={settle_result.consecutive_count})"
        )

    # Step 2: collect raw detector samples at that settled position
    dwell_result = collect_dwell_samples(
        reader=reader,
        **dwell_defaults,
    )

    # Step 3: summarize into one DataPoint
    return calculate_statistics(
        position=settle_result.position,
        dwell_samples=dwell_result,
    )
