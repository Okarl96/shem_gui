#!/usr/bin/env python3
"""Simple Scanner Control Script"""

import argparse
import sys
from pathlib import Path

# Import core components
from core.mqtt_client import MQTTClient
from core.command_sender import CommandSender
from core.position_signal_reader import PositionSignalReader
from scan_controller import ScanController
from data_storager import DataStorager


def run_2d_scan(args):
    """Execute 2D scan"""
    # Setup
    mqtt = MQTTClient(host=args.mqtt_host, port=args.mqtt_port)
    reader = PositionSignalReader(history_size=100)

    # Wire the reader to incoming MQTT streams
    mqtt.set_callbacks(
        on_position=reader.update_position,
        on_current=reader.update_signal
    )

    # Now connect with the default (float) timeout
    mqtt.connect()

    commander = CommandSender(mqtt)
    controller = ScanController(commander, reader)
    storager = DataStorager(args.output)

    # Parse vertices if provided
    vertices = None
    if args.vertices:
        vertices = []
        for v in args.vertices:
            # Strip parentheses if present to support format: (-500,0) or -500,0
            v = v.strip('()')
            x, y = map(float, v.split(','))
            vertices.append((x, y))

    # Configure
    settle_tol = {'X': args.settle_tol, 'Y': args.settle_tol, 'Z': args.settle_tol}

    if vertices:
        controller.configure_2d_vertices_scan(
            vertices_nm=vertices,
            num_points=tuple(args.num_points),
            pattern=args.pattern,
            fast_axis=args.fast_axis,
            z_fixed_nm=args.z_fixed,
            r_fixed_udeg=args.r_fixed,
            settle_tolerance=settle_tol,
            dwell_time=args.dwell_time,
            detector_lag=args.detector_lag,
            notes=args.notes
        )
    else:
        controller.configure_2d_scan(
            x_range=tuple(args.x_range),
            y_range=tuple(args.y_range),
            num_points=tuple(args.num_points),
            pattern=args.pattern,
            fast_axis=args.fast_axis,
            z_fixed_nm=args.z_fixed,
            r_fixed_udeg=args.r_fixed,
            settle_tolerance=settle_tol,
            dwell_time=args.dwell_time,
            detector_lag=args.detector_lag,
            notes=args.notes
        )

    # Setup live plotter if requested
    plotter = None
    if args.live_plot:
        from live_plotter import LivePlotter

        # Get actual position ranges
        if vertices:
            # For custom vertices, calculate bounding box
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
        else:
            x_min, x_max = args.x_range
            y_min, y_max = args.y_range

        # Create plotter with real position extent
        plotter = LivePlotter(
            plot_type='2D',
            shape=(args.num_points[1], args.num_points[0]),  # (rows, cols)
            colormap='viridis',
            title=f'2D Scan - Live View',
            interactive=True,
            extent=(x_min, x_max, y_max, y_min)  # (left, right, bottom, top) for imshow with origin='upper'
        )

        print("Live plotter enabled")

    # Execute with live plotting callback
    print("Starting scan...")
    if plotter:
        def on_point_callback(index, position, signal):
            if isinstance(index, tuple):
                # 2D index (row, col)
                linear_idx = index[0] * args.num_points[0] + index[1]
            else:
                linear_idx = index
            plotter.update_point(linear_idx, signal)

        result = controller.execute_scan(on_point_complete=on_point_callback)
        plotter.close()  # Close live plot when done
    else:
        result = controller.execute_scan()

    # Save
    print("Saving...")
    storager.save_scan(result, formats=args.formats)

    print("Done!")
    mqtt.disconnect()


def run_1d_scan(args):
    """Execute 1D scan"""
    # Setup
    mqtt = MQTTClient(host=args.mqtt_host, port=args.mqtt_port)
    reader = PositionSignalReader(history_size=100)

    # Wire the reader to incoming MQTT streams
    mqtt.set_callbacks(
        on_position=reader.update_position,
        on_current=reader.update_signal
    )

    # Now connect with the default (float) timeout
    mqtt.connect()

    commander = CommandSender(mqtt)
    controller = ScanController(commander, reader)
    storager = DataStorager(args.output)

    # Configure
    # Include BOTH scan axis and fixed axis in settle tolerance
    settle_tol = {
        args.scan_axis: args.settle_tol,
        args.fixed_axis: args.settle_tol
    }

    controller.configure_1d_line_scan(
        fixed_axis=args.fixed_axis,
        fixed_value=args.fixed_value,
        scan_axis=args.scan_axis,
        start=args.start,
        end=args.end,
        num_points=args.num_points,
        bidirectional=args.bidirectional,
        settle_tolerance=settle_tol,
        dwell_time=args.dwell_time,
        detector_lag=args.detector_lag,
        notes=args.notes
    )

    # Setup live plotter if requested
    plotter = None
    if args.live_plot:
        from live_plotter import LivePlotter

        plotter = LivePlotter(
            plot_type='1D',
            shape=(args.num_points,),
            colormap='viridis',
            title=f'1D Line Scan - Live View',
            interactive=True
        )

        # Set axis labels
        plotter.fig.axes[0].set_xlabel(f'{args.scan_axis} position (nm)')
        plotter.fig.axes[0].set_ylabel('Signal (nA)')

        print("Live plotter enabled")

    # Execute
    print("Starting scan...")
    if plotter:
        def on_point_callback(index, position, signal):
            plotter.update_point(index, signal)

        result = controller.execute_scan(on_point_complete=on_point_callback)
        plotter.close()
    else:
        result = controller.execute_scan()

    # Save
    print("Saving...")
    storager.save_scan(result, formats=args.formats)

    print("Done!")
    mqtt.disconnect()


def run_z_scan(args):
    """Execute Z scan"""
    # Setup
    mqtt = MQTTClient(host=args.mqtt_host, port=args.mqtt_port)
    reader = PositionSignalReader(history_size=100)

    # Wire the reader to incoming MQTT streams
    mqtt.set_callbacks(
        on_position=reader.update_position,
        on_current=reader.update_signal
    )

    # Now connect with the default (float) timeout
    mqtt.connect()

    commander = CommandSender(mqtt)
    controller = ScanController(commander, reader)
    storager = DataStorager(args.output)

    # Configure
    settle_tol = {'X': args.settle_tol, 'Y': args.settle_tol, 'Z': args.settle_tol}
    z_comp = {
        'z_ref': args.z_comp_ref,
        'x_ratio': args.z_comp_x_ratio,
        'y_ratio': args.z_comp_y_ratio
    }

    controller.configure_z_scan(
        x_nominal=args.x_pos,
        y_nominal=args.y_pos,
        z_start=args.z_range[0],
        z_end=args.z_range[1],
        num_points=args.num_points,
        z_compensation=z_comp,
        settle_tolerance=settle_tol,
        dwell_time=args.dwell_time,
        detector_lag=args.detector_lag,
        notes=args.notes
    )

    # Setup live plotter if requested
    plotter = None
    if args.live_plot:
        from live_plotter import LivePlotter

        plotter = LivePlotter(
            plot_type='1D',
            shape=(args.num_points,),
            colormap='viridis',
            title=f'Z Scan - Live View',
            interactive=True
        )

        # Set axis labels
        plotter.fig.axes[0].set_xlabel('Z position (nm)')
        plotter.fig.axes[0].set_ylabel('Signal (nA)')

        print("Live plotter enabled")

    # Execute
    print("Starting scan...")
    if plotter:
        def on_point_callback(index, position, signal):
            plotter.update_point(index, signal)

        result = controller.execute_scan(on_point_complete=on_point_callback)
        plotter.close()
    else:
        result = controller.execute_scan()

    # Save
    print("Saving...")
    storager.save_scan(result, formats=args.formats)

    print("Done!")
    mqtt.disconnect()


def run_multi_z_scan(args):
    """Execute Multi-Z scan: Visit multiple XY points at each Z height"""
    # Setup
    mqtt = MQTTClient(host=args.mqtt_host, port=args.mqtt_port)
    reader = PositionSignalReader(history_size=100)

    mqtt.set_callbacks(
        on_position=reader.update_position,
        on_current=reader.update_signal
    )
    mqtt.connect()

    commander = CommandSender(mqtt)
    controller = ScanController(commander, reader)
    storager = DataStorager(args.output)

    # Parse XY points
    xy_points = []
    for pt_str in args.xy_points:
        pt_str = pt_str.strip('()')
        x, y = map(float, pt_str.split(','))
        xy_points.append((x, y))

    print(f"Multi-Z Scan Configuration:")
    print(f"  XY points: {len(xy_points)} locations")
    for i, (x, y) in enumerate(xy_points):
        print(f"    Point {i + 1}: ({x:.0f}, {y:.0f}) nm")
    print(f"  Z range: {args.z_range[0]:.0f} to {args.z_range[1]:.0f} nm")
    print(f"  Z steps: {args.z_steps}")
    print(f"  Total points: {len(xy_points) * args.z_steps}")

    # Configure
    settle_tol = {'X': args.settle_tol, 'Y': args.settle_tol, 'Z': args.settle_tol_z}

    controller.configure_multi_z_scan(
        xy_points=xy_points,
        z_range=tuple(args.z_range),
        num_z_steps=args.z_steps,
        z_compensation={
            'z_ref': args.z_comp_ref,
            'x_ratio': args.z_comp_x_ratio,
            'y_ratio': args.z_comp_y_ratio
        },
        settle_tolerance=settle_tol,
        dwell_time=args.dwell_time,
        detector_lag=args.detector_lag,
        notes=args.notes
    )

    # Execute
    print("\nStarting Multi-Z scan...")
    result = controller.execute_scan()

    # Save
    scan_id = storager.save_scan(result, formats=args.formats)
    print(f"\nâœ“ Multi-Z scan complete!")
    print(f"  Scan ID: {scan_id}")
    print(f"  Points acquired: {len(result.positions)}")

    mqtt.disconnect()


def run_2d_z_series(args):
    """Execute Z-series of 2D scans"""
    from series_scan_controller import ZSeriesScanController

    # Setup
    mqtt = MQTTClient(host=args.mqtt_host, port=args.mqtt_port)
    reader = PositionSignalReader(history_size=100)

    mqtt.set_callbacks(
        on_position=reader.update_position,
        on_current=reader.update_signal
    )
    mqtt.connect()

    commander = CommandSender(mqtt)
    z_controller = ZSeriesScanController(commander, reader)
    storager = DataStorager(args.output)

    # Parse vertices if provided
    vertices = None
    if hasattr(args, 'vertices') and args.vertices:
        vertices = []
        for v in args.vertices:
            v = v.strip('()')
            x, y = map(float, v.split(','))
            vertices.append((x, y))

    print(f"Z-Series 2D Scan Configuration:")
    if vertices:
        print(f"  Scan area: Custom polygon ({len(vertices)} vertices)")
    else:
        print(f"  X range: {args.x_range[0]:.0f} to {args.x_range[1]:.0f} nm")
        print(f"  Y range: {args.y_range[0]:.0f} to {args.y_range[1]:.0f} nm")
    print(f"  Pixels: {args.num_points[0]} Ã— {args.num_points[1]}")
    print(f"  Z range: {args.z_start:.0f} to {args.z_end:.0f} nm")
    print(f"  Z slices: {args.z_numbers}")
    print(f"  Total scans: {args.z_numbers} (one per Z)")

    # Configure
    settle_tol = {'X': args.settle_tol, 'Y': args.settle_tol, 'Z': args.settle_tol_z}

    if vertices:
        z_controller.configure_z_series_from_vertices(
            vertices_nm=vertices,
            num_points=tuple(args.num_points),
            z_range=(args.z_start, args.z_end),
            z_numbers=args.z_numbers,
            z_compensation={
                'z_ref': args.z_comp_ref,
                'x_ratio': args.z_comp_x_ratio,
                'y_ratio': args.z_comp_y_ratio
            },
            pattern=args.pattern,
            fast_axis=args.fast_axis,
            r_fixed_udeg=args.r_fixed,
            settle_tolerance=settle_tol,
            dwell_time=args.dwell_time,
            detector_lag=args.detector_lag,
            notes=args.notes
        )
    else:
        z_controller.configure_z_series_from_2d(
            x_range=tuple(args.x_range),
            y_range=tuple(args.y_range),
            num_points=tuple(args.num_points),
            z_range=(args.z_start, args.z_end),
            z_numbers=args.z_numbers,
            z_compensation={
                'z_ref': args.z_comp_ref,
                'x_ratio': args.z_comp_x_ratio,
                'y_ratio': args.z_comp_y_ratio
            },
            pattern=args.pattern,
            fast_axis=args.fast_axis,
            r_fixed_udeg=args.r_fixed,
            settle_tolerance=settle_tol,
            dwell_time=args.dwell_time,
            detector_lag=args.detector_lag,
            notes=args.notes
        )

    # Execute with progress callback
    print("\nStarting Z-series scan...")

    def on_z_slice(z_idx, slice_result):
        z_pos = z_controller.z_positions[z_idx]
        print(f"  Z slice {z_idx + 1}/{len(z_controller.z_positions)} at Z={z_pos:.0f}nm complete "
              f"({len(slice_result.positions)} points)")
        # Save each slice
        scan_id = storager.save_scan(slice_result, formats=args.formats)
        print(f"    Saved as {scan_id}")

    result = z_controller.execute_z_series(on_slice_complete=on_z_slice)

    print(f"\nâœ“ Z-series complete!")
    print(f"  Total slices: {len(result.slices)}")
    print(f"  Total points: {sum(len(s.positions) for s in result.slices)}")

    mqtt.disconnect()


def run_2d_r_series(args):
    """Execute R-series of 2D scans"""
    from series_scan_controller import RSeriesScanController

    # Setup
    mqtt = MQTTClient(host=args.mqtt_host, port=args.mqtt_port)
    reader = PositionSignalReader(history_size=100)

    mqtt.set_callbacks(
        on_position=reader.update_position,
        on_current=reader.update_signal
    )
    mqtt.connect()

    commander = CommandSender(mqtt)
    r_controller = RSeriesScanController(commander, reader)
    storager = DataStorager(args.output)

    # Parse vertices if provided
    vertices = None
    if hasattr(args, 'vertices') and args.vertices:
        vertices = []
        for v in args.vertices:
            v = v.strip('()')
            x, y = map(float, v.split(','))
            vertices.append((x, y))

    print(f"R-Series 2D Scan Configuration:")
    if vertices:
        print(f"  Scan area: Custom polygon ({len(vertices)} vertices)")
    else:
        print(f"  X range: {args.x_range[0]:.0f} to {args.x_range[1]:.0f} nm")
        print(f"  Y range: {args.y_range[0]:.0f} to {args.y_range[1]:.0f} nm")
    print(f"  Pixels: {args.num_points[0]} Ã— {args.num_points[1]}")
    print(f"  R range: {args.r_start / 1000:.3f}Â° to {args.r_end / 1000:.3f}Â°")
    print(f"  R slices: {args.r_numbers}")
    print(f"  Mode: {args.mode.upper()}")

    # Configure rotation mode
    if args.mode == 'cor':
        print(f"  COR: ({args.cor_x:.0f}, {args.cor_y:.0f}) nm at Z={args.cor_base_z:.0f} nm")
        print(f"  COR compensation: X={args.cor_x_ratio:.6f}, Y={args.cor_y_ratio:.6f}")
        r_controller.set_rotation_mode(
            mode='cor',
            cor_x=args.cor_x,
            cor_y=args.cor_y,
            cor_base_z=args.cor_base_z,
            cor_x_ratio=args.cor_x_ratio,
            cor_y_ratio=args.cor_y_ratio
        )
    else:
        print(f"  Simple rotation: Stage only")
        r_controller.set_rotation_mode(mode='simple')

    # Configure scan
    settle_tol = {'X': args.settle_tol, 'Y': args.settle_tol, 'Z': args.settle_tol_z}

    if vertices:
        r_controller.configure_r_series_from_vertices(
            vertices_nm=vertices,
            num_points=tuple(args.num_points),
            r_start_udeg=args.r_start,
            r_end_udeg=args.r_end,
            r_numbers=args.r_numbers,
            r_base_udeg=args.r_base,
            pattern=args.pattern,
            fast_axis=args.fast_axis,
            z_fixed_nm=args.z_fixed,
            settle_tolerance=settle_tol,
            dwell_time=args.dwell_time,
            detector_lag=args.detector_lag,
            notes=args.notes
        )
    else:
        r_controller.configure_r_series_from_2d(
            x_range=tuple(args.x_range),
            y_range=tuple(args.y_range),
            num_points=tuple(args.num_points),
            r_start_udeg=args.r_start,
            r_end_udeg=args.r_end,
            r_numbers=args.r_numbers,
            r_base_udeg=args.r_base,
            pattern=args.pattern,
            fast_axis=args.fast_axis,
            z_fixed_nm=args.z_fixed,
            settle_tolerance=settle_tol,
            dwell_time=args.dwell_time,
            detector_lag=args.detector_lag,
            notes=args.notes
        )

    # Execute with progress callback
    print("\nStarting R-series scan...")

    def on_r_slice(r_idx, slice_result):
        r_angle = r_controller.r_angles_udeg[r_idx]
        print(f"  R angle {r_idx + 1}/{len(r_controller.r_angles_udeg)} at R={r_angle / 1000:.3f}Â° complete "
              f"({len(slice_result.positions)} points)")
        # Save each slice
        scan_id = storager.save_scan(slice_result, formats=args.formats)
        print(f"    Saved as {scan_id}")

    result = r_controller.execute_r_series(on_slice_complete=on_r_slice)

    print(f"\nâœ“ R-series complete!")
    print(f"  Total slices: {len(result.slices)}")
    print(f"  Total points: {sum(len(s.positions) for s in result.slices)}")

    mqtt.disconnect()


def run_z_r_series(args):
    """Execute R-series of Z-scans (old ZR scan)"""
    from series_scan_controller import RSeriesScanController

    # Setup
    mqtt = MQTTClient(host=args.mqtt_host, port=args.mqtt_port)
    reader = PositionSignalReader(history_size=100)

    mqtt.set_callbacks(
        on_position=reader.update_position,
        on_current=reader.update_signal
    )
    mqtt.connect()

    commander = CommandSender(mqtt)
    r_controller = RSeriesScanController(commander, reader)
    storager = DataStorager(args.output)

    print(f"R-Series Z-Scan Configuration (ZR Scan):")
    print(f"  XY position: ({args.x_pos:.0f}, {args.y_pos:.0f}) nm")
    print(f"  Z range: {args.z_range[0]:.0f} to {args.z_range[1]:.0f} nm")
    print(f"  Z points: {args.num_z_points}")
    print(f"  R range: {args.r_start / 1000:.3f}Â° to {args.r_end / 1000:.3f}Â°")
    print(f"  R slices: {args.r_numbers}")
    print(f"  COR: ({args.cor_x:.0f}, {args.cor_y:.0f}) nm at Z={args.cor_base_z:.0f} nm")
    print(f"  Total points: {args.num_z_points * args.r_numbers}")

    # Configure COR mode (required for Z-scan in R-series)
    r_controller.set_rotation_mode(
        mode='cor',
        cor_x=args.cor_x,
        cor_y=args.cor_y,
        cor_base_z=args.cor_base_z,
        cor_x_ratio=args.cor_x_ratio,
        cor_y_ratio=args.cor_y_ratio
    )

    # Configure scan
    settle_tol = {'X': args.settle_tol, 'Y': args.settle_tol, 'Z': args.settle_tol_z}

    r_controller.configure_r_series_from_z_scan(
        x_nominal=args.x_pos,
        y_nominal=args.y_pos,
        z_range=tuple(args.z_range),
        num_z_points=args.num_z_points,
        r_start_udeg=args.r_start,
        r_end_udeg=args.r_end,
        r_numbers=args.r_numbers,
        r_base_udeg=args.r_base,
        z_compensation={
            'z_ref': args.z_comp_ref,
            'x_ratio': args.z_comp_x_ratio,
            'y_ratio': args.z_comp_y_ratio
        },
        settle_tolerance=settle_tol,
        dwell_time=args.dwell_time,
        detector_lag=args.detector_lag,
        notes=args.notes
    )

    # Execute with progress callback
    print("\nStarting ZR scan (R-series of Z-scans)...")

    def on_r_slice(r_idx, slice_result):
        r_angle = r_controller.r_angles_udeg[r_idx]
        print(f"  R angle {r_idx + 1}/{len(r_controller.r_angles_udeg)} at R={r_angle / 1000:.3f}Â° complete "
              f"({len(slice_result.positions)} points)")
        # Save each Z-scan
        scan_id = storager.save_scan(slice_result, formats=args.formats)
        print(f"    Saved as {scan_id}")

    result = r_controller.execute_r_series(on_slice_complete=on_r_slice)

    print(f"\nâœ“ ZR scan complete!")
    print(f"  Total R-slices: {len(result.slices)}")
    print(f"  Total points: {sum(len(s.positions) for s in result.slices)}")

    mqtt.disconnect()


def run_multi_z_r_series(args):
    """Execute R-series of Multi-Z scans (3D tomography!)"""
    from series_scan_controller import RSeriesScanController

    # Setup
    mqtt = MQTTClient(host=args.mqtt_host, port=args.mqtt_port)
    reader = PositionSignalReader(history_size=100)

    mqtt.set_callbacks(
        on_position=reader.update_position,
        on_current=reader.update_signal
    )
    mqtt.connect()

    commander = CommandSender(mqtt)
    r_controller = RSeriesScanController(commander, reader)
    storager = DataStorager(args.output)

    # Parse XY points
    xy_points = []
    for pt_str in args.xy_points:
        pt_str = pt_str.strip('()')
        x, y = map(float, pt_str.split(','))
        xy_points.append((x, y))

    print(f"3D Tomography Scan Configuration (R-series of Multi-Z):")
    print(f"  XY points: {len(xy_points)} locations")
    for i, (x, y) in enumerate(xy_points):
        print(f"    Point {i + 1}: ({x:.0f}, {y:.0f}) nm")
    print(f"  Z range: {args.z_range[0]:.0f} to {args.z_range[1]:.0f} nm")
    print(f"  Z steps: {args.z_steps}")
    print(f"  R range: {args.r_start / 1000:.3f}Â° to {args.r_end / 1000:.3f}Â°")
    print(f"  R slices: {args.r_numbers}")
    print(f"  COR: ({args.cor_x:.0f}, {args.cor_y:.0f}) nm at Z={args.cor_base_z:.0f} nm")
    print(f"  Total points: {len(xy_points)} Ã— {args.z_steps} Ã— {args.r_numbers} = "
          f"{len(xy_points) * args.z_steps * args.r_numbers}")

    # Configure COR mode
    r_controller.set_rotation_mode(
        mode='cor',
        cor_x=args.cor_x,
        cor_y=args.cor_y,
        cor_base_z=args.cor_base_z,
        cor_x_ratio=args.cor_x_ratio,
        cor_y_ratio=args.cor_y_ratio
    )

    # Configure scan
    settle_tol = {'X': args.settle_tol, 'Y': args.settle_tol, 'Z': args.settle_tol_z}

    r_controller.configure_r_series_from_multi_z(
        xy_points=xy_points,
        z_range=tuple(args.z_range),
        num_z_steps=args.z_steps,
        r_start_udeg=args.r_start,
        r_end_udeg=args.r_end,
        r_numbers=args.r_numbers,
        r_base_udeg=args.r_base,
        z_compensation={
            'z_ref': args.z_comp_ref,
            'x_ratio': args.z_comp_x_ratio,
            'y_ratio': args.z_comp_y_ratio
        },
        settle_tolerance=settle_tol,
        dwell_time=args.dwell_time,
        detector_lag=args.detector_lag,
        notes=args.notes
    )

    # Execute with progress callback
    print("\nðŸ”¬ Starting 3D tomography scan...")
    print("This will take a while - grab a coffee! â˜•")

    def on_r_slice(r_idx, slice_result):
        r_angle = r_controller.r_angles_udeg[r_idx]
        print(f"\n  âœ“ R angle {r_idx + 1}/{len(r_controller.r_angles_udeg)} at R={r_angle / 1000:.3f}Â° complete")
        print(f"    Points acquired: {len(slice_result.positions)}")
        # Save the multi-Z result (will be saved as separate Z-scans per XY point)
        scan_id = storager.save_scan(slice_result, formats=args.formats)
        print(f"    Saved as {scan_id}")

    result = r_controller.execute_r_series(on_slice_complete=on_r_slice)

    print(f"\nðŸŽ‰ 3D Tomography complete!")
    print(f"  Total R-slices: {len(result.slices)}")
    print(f"  Total points: {sum(len(s.positions) for s in result.slices)}")
    print(f"  Ready for 3D reconstruction!")

    mqtt.disconnect()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Scanner Control')
    subparsers = parser.add_subparsers(dest='scan_type', required=True)

    # ========================================================================
    # 2D SCAN
    # ========================================================================
    p2d = subparsers.add_parser('2d', help='2D area scan')
    p2d.add_argument('--output', default='./scan_data', help='Output directory')
    p2d.add_argument('--x-range', nargs=2, type=float, default=[0, 10000])
    p2d.add_argument('--y-range', nargs=2, type=float, default=[0, 10000])
    p2d.add_argument('--num-points', nargs=2, type=int, default=[100, 100])
    p2d.add_argument('--pattern', choices=['raster', 'snake'], default='raster')
    p2d.add_argument('--fast-axis', choices=['X', 'Y'], default='X')
    p2d.add_argument('--z-fixed', type=float, default=None)
    p2d.add_argument('--r-fixed', type=float, default=None)
    p2d.add_argument('--vertices', nargs='+', type=str, default=None,
                     help='Polygon vertices as x,y pairs. For negative coordinates, '
                          'wrap in parentheses: --vertices "(-500,0)" "(500,0)" "(0,500)"')
    p2d.add_argument('--dwell-time', type=float, default=0.05)
    p2d.add_argument('--detector-lag', type=float, default=0.01)
    p2d.add_argument('--settle-tol', type=float, default=5)
    p2d.add_argument('--mqtt-host', default='localhost')
    p2d.add_argument('--mqtt-port', type=int, default=1883)
    p2d.add_argument('--formats', nargs='+', default=['hdf5', 'csv', 'png'],
                     choices=['hdf5', 'csv', 'png'])
    p2d.add_argument('--notes', default='')
    p2d.add_argument('--live-plot', action='store_true',
                     help='Enable live plotting during scan')
    p2d.set_defaults(func=run_2d_scan)

    # ========================================================================
    # 1D SCAN
    # ========================================================================
    p1d = subparsers.add_parser('1d', help='1D line scan')
    p1d.add_argument('--output', default='./scan_data')
    p1d.add_argument('--scan-axis', choices=['X', 'Y'], required=True)
    p1d.add_argument('--start', type=float, required=True)
    p1d.add_argument('--end', type=float, required=True)
    p1d.add_argument('--num-points', type=int, default=100)
    p1d.add_argument('--fixed-axis', choices=['X', 'Y'], required=True)
    p1d.add_argument('--fixed-value', type=float, required=True)
    p1d.add_argument('--bidirectional', action='store_true')
    p1d.add_argument('--dwell-time', type=float, default=0.05)
    p1d.add_argument('--detector-lag', type=float, default=0.01)
    p1d.add_argument('--settle-tol', type=float, default=5)
    p1d.add_argument('--mqtt-host', default='localhost')
    p1d.add_argument('--mqtt-port', type=int, default=1883)
    p1d.add_argument('--formats', nargs='+', default=['hdf5', 'csv', 'png'],
                     choices=['hdf5', 'csv', 'png'])
    p1d.add_argument('--notes', default='')
    p1d.add_argument('--live-plot', action='store_true',
                     help='Enable live plotting during scan')
    p1d.set_defaults(func=run_1d_scan)

    # ========================================================================
    # Z SCAN
    # ========================================================================
    pz = subparsers.add_parser('z', help='Z-axis scan')
    pz.add_argument('--output', default='./scan_data')
    pz.add_argument('--x-pos', type=float, required=True)
    pz.add_argument('--y-pos', type=float, required=True)
    pz.add_argument('--z-range', nargs=2, type=float, required=True)
    pz.add_argument('--num-points', type=int, default=100)
    pz.add_argument('--z-comp-ref', type=float, default=0)
    pz.add_argument('--z-comp-x-ratio', type=float, default=0)
    pz.add_argument('--z-comp-y-ratio', type=float, default=0)
    pz.add_argument('--dwell-time', type=float, default=0.05)
    pz.add_argument('--detector-lag', type=float, default=0.01)
    pz.add_argument('--settle-tol', type=float, default=10)
    pz.add_argument('--mqtt-host', default='localhost')
    pz.add_argument('--mqtt-port', type=int, default=1883)
    pz.add_argument('--formats', nargs='+', default=['hdf5', 'csv', 'png'],
                    choices=['hdf5', 'csv', 'png'])
    pz.add_argument('--notes', default='')
    pz.add_argument('--live-plot', action='store_true',
                    help='Enable live plotting during scan')
    pz.set_defaults(func=run_z_scan)

    # ========================================================================
    # MULTI-Z SCAN
    # ========================================================================
    p_multi_z = subparsers.add_parser('multi-z',
                                      help='Multi-Z scan: Visit multiple XY points at each Z')
    p_multi_z.add_argument('--output', default='./scan_data')
    p_multi_z.add_argument('--xy-points', nargs='+', required=True,
                           help='XY points as "x,y" or "(x,y)" (e.g., "0,0" "5000,5000")')
    p_multi_z.add_argument('--z-range', nargs=2, type=float, required=True,
                           help='Z range: z_start z_end (nm)')
    p_multi_z.add_argument('--z-steps', type=int, required=True,
                           help='Number of Z steps')
    p_multi_z.add_argument('--z-comp-ref', type=float, default=0,
                           help='Z compensation reference (nm)')
    p_multi_z.add_argument('--z-comp-x-ratio', type=float, default=1.0,
                           help='X compensation per Z (1.0 = 45Â° beam)')
    p_multi_z.add_argument('--z-comp-y-ratio', type=float, default=0.0,
                           help='Y compensation per Z')
    p_multi_z.add_argument('--dwell-time', type=float, default=0.1)
    p_multi_z.add_argument('--detector-lag', type=float, default=0.01)
    p_multi_z.add_argument('--settle-tol', type=float, default=5)
    p_multi_z.add_argument('--settle-tol-z', type=float, default=10)
    p_multi_z.add_argument('--mqtt-host', default='localhost')
    p_multi_z.add_argument('--mqtt-port', type=int, default=1883)
    p_multi_z.add_argument('--formats', nargs='+', default=['hdf5', 'csv'],
                           choices=['hdf5', 'csv', 'png'])
    p_multi_z.add_argument('--notes', default='')
    p_multi_z.set_defaults(func=run_multi_z_scan)

    # ========================================================================
    # Z-SERIES OF 2D SCANS
    # ========================================================================
    p_2d_z = subparsers.add_parser('2d-z-series',
                                   help='Z-series: 2D scans at different Z heights')
    p_2d_z.add_argument('--output', default='./scan_data')

    # Scan area (rectangular OR vertices)
    p_2d_z.add_argument('--x-range', nargs=2, type=float,
                        help='X range: x_start x_end (nm)')
    p_2d_z.add_argument('--y-range', nargs=2, type=float,
                        help='Y range: y_start y_end (nm)')
    p_2d_z.add_argument('--vertices', nargs='+',
                        help='Polygon vertices as "x,y" or "(x,y)"')

    p_2d_z.add_argument('--num-points', nargs=2, type=int, required=True,
                        help='Number of points: nx ny')
    p_2d_z.add_argument('--pattern', choices=['raster', 'snake'], default='snake')
    p_2d_z.add_argument('--fast-axis', choices=['X', 'Y'], default='X')

    # Z-series parameters
    p_2d_z.add_argument('--z-start', type=float, required=True)
    p_2d_z.add_argument('--z-end', type=float, required=True)
    p_2d_z.add_argument('--z-numbers', type=int, required=True,
                        help='Number of Z slices')
    p_2d_z.add_argument('--z-comp-ref', type=float, default=0)
    p_2d_z.add_argument('--z-comp-x-ratio', type=float, default=1.0)
    p_2d_z.add_argument('--z-comp-y-ratio', type=float, default=0.0)

    # Other parameters
    p_2d_z.add_argument('--r-fixed', type=float, default=None,
                        help='Fixed R angle (Âµdeg)')
    p_2d_z.add_argument('--dwell-time', type=float, default=0.05)
    p_2d_z.add_argument('--detector-lag', type=float, default=0.01)
    p_2d_z.add_argument('--settle-tol', type=float, default=5)
    p_2d_z.add_argument('--settle-tol-z', type=float, default=10)
    p_2d_z.add_argument('--mqtt-host', default='localhost')
    p_2d_z.add_argument('--mqtt-port', type=int, default=1883)
    p_2d_z.add_argument('--formats', nargs='+', default=['hdf5', 'csv', 'png'],
                        choices=['hdf5', 'csv', 'png'])
    p_2d_z.add_argument('--notes', default='')
    p_2d_z.set_defaults(func=run_2d_z_series)

    # ========================================================================
    # R-SERIES OF 2D SCANS
    # ========================================================================
    p_2d_r = subparsers.add_parser('2d-r-series',
                                   help='R-series: 2D scans at different rotation angles')
    p_2d_r.add_argument('--output', default='./scan_data')

    # Scan area (rectangular OR vertices)
    p_2d_r.add_argument('--x-range', nargs=2, type=float,
                        help='X range: x_start x_end (nm)')
    p_2d_r.add_argument('--y-range', nargs=2, type=float,
                        help='Y range: y_start y_end (nm)')
    p_2d_r.add_argument('--vertices', nargs='+',
                        help='Polygon vertices as "x,y" or "(x,y)"')

    p_2d_r.add_argument('--num-points', nargs=2, type=int, required=True,
                        help='Number of points: nx ny')
    p_2d_r.add_argument('--pattern', choices=['raster', 'snake'], default='snake')
    p_2d_r.add_argument('--fast-axis', choices=['X', 'Y'], default='X')

    # R-series parameters
    p_2d_r.add_argument('--r-start', type=float, required=True,
                        help='Starting R angle (Âµdeg)')
    p_2d_r.add_argument('--r-end', type=float, required=True,
                        help='Ending R angle (Âµdeg)')
    p_2d_r.add_argument('--r-numbers', type=int, required=True,
                        help='Number of R slices')
    p_2d_r.add_argument('--r-base', type=float, default=0,
                        help='Base R angle offset (Âµdeg)')

    # Rotation mode
    p_2d_r.add_argument('--mode', choices=['simple', 'cor'], default='simple',
                        help='Rotation mode: simple (stage only) or cor (transform coordinates)')

    # COR parameters (required if mode=cor)
    p_2d_r.add_argument('--cor-x', type=float, default=0,
                        help='COR X coordinate (nm)')
    p_2d_r.add_argument('--cor-y', type=float, default=0,
                        help='COR Y coordinate (nm)')
    p_2d_r.add_argument('--cor-base-z', type=float, default=0,
                        help='Z where COR is valid (nm)')
    p_2d_r.add_argument('--cor-x-ratio', type=float, default=1.0,
                        help='COR X shift per Z (1.0 = 45Â° beam)')
    p_2d_r.add_argument('--cor-y-ratio', type=float, default=0.0,
                        help='COR Y shift per Z')

    # Other parameters
    p_2d_r.add_argument('--z-fixed', type=float, default=0,
                        help='Fixed Z position (nm)')
    p_2d_r.add_argument('--dwell-time', type=float, default=0.05)
    p_2d_r.add_argument('--detector-lag', type=float, default=0.01)
    p_2d_r.add_argument('--settle-tol', type=float, default=5)
    p_2d_r.add_argument('--settle-tol-z', type=float, default=10)
    p_2d_r.add_argument('--mqtt-host', default='localhost')
    p_2d_r.add_argument('--mqtt-port', type=int, default=1883)
    p_2d_r.add_argument('--formats', nargs='+', default=['hdf5', 'csv', 'png'],
                        choices=['hdf5', 'csv', 'png'])
    p_2d_r.add_argument('--notes', default='')
    p_2d_r.set_defaults(func=run_2d_r_series)

    # ========================================================================
    # R-SERIES OF Z-SCANS (ZR SCAN)
    # ========================================================================
    p_z_r = subparsers.add_parser('z-r-series',
                                  help='ZR scan: R-series of Z-scans')
    p_z_r.add_argument('--output', default='./scan_data')

    # XY position
    p_z_r.add_argument('--x-pos', type=float, required=True,
                       help='X position (nm)')
    p_z_r.add_argument('--y-pos', type=float, required=True,
                       help='Y position (nm)')

    # Z parameters
    p_z_r.add_argument('--z-range', nargs=2, type=float, required=True,
                       help='Z range: z_start z_end (nm)')
    p_z_r.add_argument('--num-z-points', type=int, required=True,
                       help='Number of Z points')
    p_z_r.add_argument('--z-comp-ref', type=float, default=0)
    p_z_r.add_argument('--z-comp-x-ratio', type=float, default=1.0)
    p_z_r.add_argument('--z-comp-y-ratio', type=float, default=0.0)

    # R parameters
    p_z_r.add_argument('--r-start', type=float, required=True,
                       help='Starting R angle (Âµdeg)')
    p_z_r.add_argument('--r-end', type=float, required=True,
                       help='Ending R angle (Âµdeg)')
    p_z_r.add_argument('--r-numbers', type=int, required=True,
                       help='Number of R slices')
    p_z_r.add_argument('--r-base', type=float, default=0)

    # COR parameters (required for Z in R-series)
    p_z_r.add_argument('--cor-x', type=float, required=True,
                       help='COR X coordinate (nm)')
    p_z_r.add_argument('--cor-y', type=float, required=True,
                       help='COR Y coordinate (nm)')
    p_z_r.add_argument('--cor-base-z', type=float, default=0)
    p_z_r.add_argument('--cor-x-ratio', type=float, default=1.0)
    p_z_r.add_argument('--cor-y-ratio', type=float, default=0.0)

    # Other
    p_z_r.add_argument('--dwell-time', type=float, default=0.1)
    p_z_r.add_argument('--detector-lag', type=float, default=0.01)
    p_z_r.add_argument('--settle-tol', type=float, default=5)
    p_z_r.add_argument('--settle-tol-z', type=float, default=10)
    p_z_r.add_argument('--mqtt-host', default='localhost')
    p_z_r.add_argument('--mqtt-port', type=int, default=1883)
    p_z_r.add_argument('--formats', nargs='+', default=['hdf5', 'csv'],
                       choices=['hdf5', 'csv', 'png'])
    p_z_r.add_argument('--notes', default='')
    p_z_r.set_defaults(func=run_z_r_series)

    # ========================================================================
    # R-SERIES OF MULTI-Z (3D TOMOGRAPHY)
    # ========================================================================
    p_multi_z_r = subparsers.add_parser('tomo',
                                        help='3D Tomography: R-series of Multi-Z scans')
    p_multi_z_r.add_argument('--output', default='./scan_data')

    # Multi-Z parameters
    p_multi_z_r.add_argument('--xy-points', nargs='+', required=True,
                             help='XY points as "x,y" or "(x,y)"')
    p_multi_z_r.add_argument('--z-range', nargs=2, type=float, required=True,
                             help='Z range: z_start z_end (nm)')
    p_multi_z_r.add_argument('--z-steps', type=int, required=True,
                             help='Number of Z steps')
    p_multi_z_r.add_argument('--z-comp-ref', type=float, default=0)
    p_multi_z_r.add_argument('--z-comp-x-ratio', type=float, default=1.0)
    p_multi_z_r.add_argument('--z-comp-y-ratio', type=float, default=0.0)

    # R parameters
    p_multi_z_r.add_argument('--r-start', type=float, required=True,
                             help='Starting R angle (Âµdeg)')
    p_multi_z_r.add_argument('--r-end', type=float, required=True,
                             help='Ending R angle (Âµdeg)')
    p_multi_z_r.add_argument('--r-numbers', type=int, required=True,
                             help='Number of R slices')
    p_multi_z_r.add_argument('--r-base', type=float, default=0)

    # COR parameters (required)
    p_multi_z_r.add_argument('--cor-x', type=float, required=True,
                             help='COR X coordinate (nm)')
    p_multi_z_r.add_argument('--cor-y', type=float, required=True,
                             help='COR Y coordinate (nm)')
    p_multi_z_r.add_argument('--cor-base-z', type=float, default=0)
    p_multi_z_r.add_argument('--cor-x-ratio', type=float, default=1.0)
    p_multi_z_r.add_argument('--cor-y-ratio', type=float, default=0.0)

    # Other
    p_multi_z_r.add_argument('--dwell-time', type=float, default=0.1)
    p_multi_z_r.add_argument('--detector-lag', type=float, default=0.01)
    p_multi_z_r.add_argument('--settle-tol', type=float, default=5)
    p_multi_z_r.add_argument('--settle-tol-z', type=float, default=10)
    p_multi_z_r.add_argument('--mqtt-host', default='localhost')
    p_multi_z_r.add_argument('--mqtt-port', type=int, default=1883)
    p_multi_z_r.add_argument('--formats', nargs='+', default=['hdf5', 'csv'],
                             choices=['hdf5', 'csv', 'png'])
    p_multi_z_r.add_argument('--notes', default='')
    p_multi_z_r.set_defaults(func=run_multi_z_r_series)

    # Parse and execute
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()