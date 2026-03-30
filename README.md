# Scanning Helium Microscope (SHeM) Controlling Script

This repository contains a sandbox simulator for mimicing the behaviour of the nanopositioner and picoammeter in real SHeM (https://doi.org/10.1016/j.nimb.2014.06.028) and a controlling script which sends commands and acquire data with user specified parameters. There is no AI training involved in this work. The code is co-developed with commericially available AI models. 

# 1. Simulator: 
By its name, it is a simulation code that outputs position and signal stream in the exact format as real SHeM. It helps you to develop the scanning code without accidentally breaking the real instrument. It interpolates JPG images as the "sample" and mimics almost all behaviors we have met in real experiments, including moving the sample in XYZ, rotating the sample around a stated center of rotation, and applying drifts in linear axes,etc. You can find a full description below. 

# 2. Control Script: 
The controlling code that is co-developed with AI models and currently been used in the real microscope to conduct experiments. It communicates with hardware through MQTT client, sends moving commands

----------------------------------

# Simulator

**Coordinate System**

- Stage coordinates: Global reference frame for X, Y, Z positions
- Sample positioning: Sample images shift in X based on Z position
- Rotatio: Sample rotates around the center of rotation (COR)
- X-Z compensation: Both sample position and COR shift in X as Z changes

The Reason for a shift in X as Z moves is due to the realistic geometry used in the real SHeM. The diffraction measurement requires a constant incidence on the same spot in different Z positions. More details refer to (https://doi.org/10.1103/PhysRevLett.131.236202).

The simulator applies independent X shifts to both the sample image and the COR:
- Image shift: `image_x = sample_center_x_base + (Z * x_per_z_ratio)`
- COR shift: `cor_x_effective = cor_x + ((Z - cor_z) * x_per_z_ratio)`

**Signal Generation**

The picoammeter signal is generated based on:
- Current stage position (X, Y, Z, R); X, Y, Z in nanometers, R in micro-degrees.
- Sample image intensity at that position
- Rotation around the COR
- Out-of-bounds detection (returns 0 signal outside sample)

Formula: `current_pA = offset_pa + gain_pa × normalized_intensity`

Where `normalized_intensity` is 0.0-1.0 from the image pixel value. You can always add a noise function to the formula to mimic real situation.

## Installation 

```bash
pip install paho-mqtt pillow numpy
```

```bash
python ecc_pico_simulator.py --images img_z0.png img_z250.png img_z500.png img_z750.png img_z1000.png --z-positions 0 250 500 750 1000 --broker localhost --port 1883 --pos-rate 100 --sig-rate 100 --fov-x 1280 --fov-y 960 --speed-xy 1000 --speed-z 1000 --sample-center-x 0 --sample-center-y 0
```

There are other command line arguments available below to simulate other situations. If you run the example usage above, you will see the outcome below in your bash: 

```bash
Loading image 1/5: img_z0.png at Z=0.0 nm
Loading image 2/5: img_z250.png at Z=250.0 nm
Loading image 3/5: img_z500.png at Z=500.0 nm
Loading image 4/5: img_z750.png at Z=750.0 nm
Loading image 5/5: img_z1000.png at Z=1000.0 nm
Z-stack initialized: 5 images
Z range: 0.0 to 1000.0 nm
Sample center at Z=0: (0.0, 0.0) nm
X-Z compensation: 1.0 nm X shift per nm Z change
[microscope/stage/position] 1774877730842331400/0/0/0/0
[picoammeter/current] 1774877730842561600/958.824

==================================================
ECC Z-Stack ROTATION Simulator
with X-Z Compensation and 3D COR
==================================================

🎯 IMAGE POSITIONING (independent of COR):
  Sample center at Z=0: (0.0, 0.0) nm
  X-Z compensation ratio: 1.0 nm/nm
  Sample center at Z=500: (500.0, 0.0) nm
  Sample center at Z=1000: (1000.0, 0.0) nm
  Sample bounds at Z=0: X=[-640.0, 640.0], Y=[-480.0, 480.0] nm

🔄 3D CENTER OF ROTATION:
  COR (at cor_z=0.0): (0.0, 0.0, 0.0) nm
  Effective COR at Z=500: (500.0, 0.0) nm
  Effective COR at Z=1000: (1000.0, 0.0) nm
  Rotation speed: 45000000.0 micro-deg/s (45.0 deg/s)

📊 Z-STACK:
  Images: 5 files
  Z-planes: [0.0, 250.0, 500.0, 750.0, 1000.0] nm

⚡ SPEEDS:
  X/Y: 1000.0 nm/s
  Z: 1000.0 nm/s
  R: 45.0 deg/s

📡 MQTT Commands:
  MOVE/<axis>/<value>    - Move axis (X/Y in nm, Z in nm, R in micro-deg)
  SET_COR/<x>/<y>/<z>    - Set 3D center of rotation (nm)
  SET_RATE/<Hz>          - Set broadcast rate
  STATUS                 - Get system status

🛑 Press Ctrl+C to stop

[MQTT] Connected rc=0
```

## Command Line Arguments

***Image Configuration***
- `--images IMAGE [IMAGE ...]` - Image file paths (PNG, JPEG, etc.)
- `--z-positions Z [Z ...]` - Z position for each image in nm (default: auto-spaced by 250nm)
- `--fov-x FOV_X` - Field of view width in nm (default: image width in pixels)
- `--fov-y FOV_Y` - Field of view height in nm (default: image height in pixels)
- `--sample-center-x X` - Sample X position at Z=0 in nm (default: 0)
- `--sample-center-y Y` - Sample Y position in nm (default: 0)
- `--x-per-z-nm RATIO` - X shift per Z change ratio (default: 1.0)

***Stage Configuration***
- `--speed-xy SPEED` - X/Y stage speed in nm/s (default: 2000)
- `--speed-z SPEED` - Z stage speed in nm/s (default: 1000)
- `--speed-r SPEED` - Rotation speed in micro-deg/s (default: 45000000)
- `--cor-x X` - Center of rotation X in nm (default: 0)
- `--cor-y Y` - Center of rotation Y in nm (default: 0)
- `--cor-z Z` - Center of rotation Z reference in nm (default: 0)
- `--limit-x-min`, `--limit-x-max` - X axis limits in nm (default: -1e12 to 1e12)
- `--limit-y-min`, `--limit-y-max` - Y axis limits in nm (default: -1e12 to 1e12)
- `--limit-z-min`, `--limit-z-max` - Z axis limits in nm (default: -1e12 to 1e12)
- `--limit-r-min`, `--limit-r-max` - R axis limits in micro-deg (default: -360e6 to 360e6)

***Signal Configuration***
- `--gain-pa GAIN` - Signal gain in pA (default: 1000)
- `--offset-pa OFFSET` - Signal offset in pA (default: 100)

***Communication***
- `--broker HOST` - MQTT broker address
- `--port PORT` - MQTT broker port (default: 1883)
- `--pos-rate HZ` - Position broadcast rate in Hz (default: 100)
- `--sig-rate HZ` - Signal broadcast rate in Hz (default: 100)

## MQTT Topics

**Published Topics:**

Position telemetry:
```
microscope/stage/position
Format: timestamp_ns/X/Y/Z/R
Example: 1699876543210000000/1000/2000/500/45000000
```

Signal telemetry:
```
picoammeter/current
Format: timestamp_ns/current_pA
Example: 1699876543210000000/5.237
```

Command results:
```
microscope/stage/result
Format: timestamp_ns/STATUS/CATEGORY/SUBCATEGORY/RESULT/details
```

**Command Topic:**

Subscribe to: `microscope/stage/command`

Using `mosquitto_pub`:

```bash
# Move to origin
mosquitto_pub -h localhost -t microscope/stage/command -m "MOVE/X/0"
mosquitto_pub -h localhost -t microscope/stage/command -m "MOVE/Y/0"

# Change Z position
mosquitto_pub -h localhost -t microscope/stage/command -m "MOVE/Z/1000"

# Rotate sample
mosquitto_pub -h localhost -t microscope/stage/command -m "MOVE/R/45000000"

# Update center of rotation
mosquitto_pub -h localhost -t microscope/stage/command -m "SET_COR/500/500/250"

# Change update rate
mosquitto_pub -h localhost -t microscope/stage/command -m "SET_RATE/2000"

# Request status
mosquitto_pub -h localhost -t microscope/stage/command -m "STATUS"
```

**Stopping the Simulator**

Press `Ctrl+C` to gracefully stop the simulator.

**Tips:**

- Z-stack spacing: Use `--z-positions` to specify exact Z planes for each image
- FOV calibration: Set `--fov-x` and `--fov-y` to match real image dimensions
- Smooth rotation: Adjust `--speed-r` for realistic rotation speeds
- Signal calibration: Tune `--gain-pa` and `--offset-pa` to match expected signal levels
- Performance: Reduce `--pos-rate` and `--sig-rate` if CPU usage is too high

**Troubleshooting:**

- No MQTT connection: Verify broker is running and address/port are correct
- Zero signal everywhere: Check sample center positioning and FOV settings
- Jerky motion: Increase `--pos-rate` for smoother movement updates
- Images not loading: Verify image paths and formats (PNG, JPEG supported)

----------------------------------

# Control Script

A Python-based control system for automated scanning microscopy with real-time data acquisition, visualization, and analysis. The system supports 2D raster scanning, 1D line scans, Z-series scanning, and multi-dimensional scan sequences.

## Overview

This project provides a comprehensive scanning microscope control system that:
- Controls stage movement via MQTT protocol
- Acquires position and signal data in real-time
- Supports multiple scan patterns (raster, snake, bidirectional)
- Provides both GUI interfaces
- Stores data in SQLite, HDF5, and CSV formats
- Includes live visualization and post-processing tools

## Installation

```bash
pip install numpy matplotlib PyQt5 pyqtgraph paho-mqtt h5py scipy scikit-image
```

```bash
python scanner.py
```

## Key Features

### Scan Types
- **2D Scans**: Rectangular regions with configurable step sizes
- **1D Line Scans**: Single-line profiles between two points
- **Z-Series Scans**: Multi-layer scanning with optional XY compensation
- **Custom Patterns**: Raster, snake, bidirectional, and user-defined trajectories

### Data Acquisition
- Real-time position monitoring via MQTT
- Signal measurement with configurable averaging
- Settle tolerance checking for accurate positioning
- Continuous and step-stop scan modes

### Data Storage
- **HDF5**: Raw high-resolution scan data
- **CSV**: Exported scan results for external analysis
- **PNG**: Scan images with embedded metadata

### Visualization
- Live 2D image display during scanning
- Real-time line profile plots
- Post-scan analysis tools
- Interactive region selection
- Image registration and drift correction

### Key Components

#### 1. MQTT Client (`mqtt_client.py`)
- Manages connection to MQTT broker
- Subscribes to position and signal topics
- Publishes movement commands
- Provides callback mechanism for data updates

#### 2. Position Signal Reader (`position_signal_reader.py`)
- Processes incoming MQTT position messages
- Buffers signal data for averaging
- Maintains position history
- Thread-safe data access

#### 3. Command Sender (`command_sender.py`)
- High-level interface for stage control
- Abstracts MQTT message formatting
- Supports absolute and relative movements
- Handles multi-axis commands

#### 4. Acquisition Primitives (`acquisition_primitives.py`)
- `move_and_settle()`: Move to position and verify arrival
- `read_position_once()`: Single position measurement
- `read_signal_averaged()`: Averaged signal acquisition
- Implements tolerance checking and timeout handling

#### 5. Scan Controller (`scan_controller.py`)
- Orchestrates entire scan workflow
- Generates scan patterns
- Executes movement and acquisition sequences
- Provides progress callbacks
- Manages metadata and results

#### 6. Data Storager (`data_storager.py`)
- Creates and manages SQLite databases
- Stores scan configurations and results
- Exports to CSV and HDF5 formats
- Generates unique scan IDs

#### 7. Scan Patterns (`scan_patterns.py`)
- `generate_raster_2d()`: Standard raster pattern
- `generate_snake_2d()`: Snake/serpentine pattern
- `generate_bidirectional_1d()`: Back-and-forth line scan
- `generate_vertices_based_pattern()`: Polygon filling
- `generate_z_series_pattern()`: Multi-layer 3D scans

### Data Flow

1. **Configuration**: User sets scan parameters via GUI or CLI
2. **Pattern Generation**: Scan trajectory is calculated
3. **MQTT Connection**: System connects to broker and subscribes to topics
4. **Scan Execution**: 
   - Move to each point in pattern
   - Wait for settling
   - Acquire signal
   - Store data
5. **Data Storage**: Results saved to SQLite/HDF5/CSV
6. **Visualization**: Real-time or post-scan plotting

## Data Formats

***HDF5 Structure***

```
scan_<id>.h5
├── metadata (attributes)
│   ├── scan_type
│   ├── timestamp
│   ├── x_range_nm
│   ├── y_range_nm
│   └── ...
├── positions (dataset, shape: [N, 3])
│   └── columns: [X, Y, Z]
└── signals (dataset, shape: [N])
```

***CSV Export Format***

```csv
scan_id,point_index,x_nm,y_nm,z_nm,signal,timestamp
scan_001,0,0.0,0.0,1000.0,123.45,2025-01-15T10:30:00
scan_001,1,100.0,0.0,1000.0,124.56,2025-01-15T10:30:01
...
```

