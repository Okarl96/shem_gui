# A-SHeM Controlling Development

This project aims to develop pure Python-based code to run the scanning on A-SHeM.

You can see three folders in the project:

# 1. Simulator: 
By its name, it is a simulation code that outputs position and signal stream in the exact format as real A-SHeM. It helps you to develop the scanning code without accidentally breaking the real instrument. It interpolates JPG images as the "sample" and mimics almost all behaviors we have met in real experiments, including moving the sample in XYZ, rotating the sample around a stated center of rotation, and applying drifts in linear axes,etc. You can find a full description below. 

# 2. Old gui: 
The single-file code that is currently used on A-SHeM. It has all the functions, from connecting to MQTT to perform scans, which now becomes too overwhelming to debug and add new functions, as the GUI is acting like a prison. 

# 3. New API:
Here, I have already extracted core functions, which you can see in detail below. The idea is to separate the base functions like connecting to the MQTT, sending commands, generating scanning patterns, acquiring data points, and saving to some files from the frontend user interface. If completed, all functions will act as API functions that can be called by Linux/Windows/MacOS terminals or a new lightweight GUI. It will also make customized scanning easier as all functions are modularized.

----------------------------------

# 1 .ECC Pico Simulator

A Python-based simulator for an Electron Channeling Contrast (ECC) microscope with picoammeter signal generation. Simulates stage positioning, rotation, Z-stack imaging, and signal output with X-Z compensation and 3D center of rotation support.

## Features

- **Z-stack imaging**: Load multiple images at different Z positions with interpolation
- **4-axis stage control**: X, Y, Z (nanometers), R (micro-degrees)
- **3D Center of Rotation (COR)**: Dynamic COR positioning with Z-dependent shifts
- **X-Z compensation**: Simulates image shift in X as Z changes
- **Real-time signal generation**: Picoammeter current output based on sample position
- **MQTT control**: Command and telemetry via MQTT broker
- **Smooth motion**: Realistic stage movement with configurable speeds

## Installation

```bash
pip install paho-mqtt pillow numpy
```

## Frequent Usage

```bash
python ecc_pico_simulator.py --images img_z0.png img_z250.png img_z500.png img_z750.png img_z1000.png --z-positions 0 250 500 750 1000 --broker localhost --port 1883 --pos-rate 100 --sig-rate 100 --fov-x 1280 --fov-y 960 --speed-xy 1000 --speed-z 1000 --sample-center-x 0 --sample-center-y 0
```

## Command Line Arguments

### Image Configuration
- `--images IMAGE [IMAGE ...]` - Image file paths (PNG, JPEG, etc.)
- `--z-positions Z [Z ...]` - Z position for each image in nm (default: auto-spaced by 250nm)
- `--fov-x FOV_X` - Field of view width in nm (default: image width in pixels)
- `--fov-y FOV_Y` - Field of view height in nm (default: image height in pixels)
- `--sample-center-x X` - Sample X position at Z=0 in nm (default: 0)
- `--sample-center-y Y` - Sample Y position in nm (default: 0)
- `--x-per-z-nm RATIO` - X shift per Z change ratio (default: 1.0)

### Stage Configuration
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

### Signal Configuration
- `--gain-pa GAIN` - Signal gain in pA (default: 1000)
- `--offset-pa OFFSET` - Signal offset in pA (default: 100)

### Communication
- `--broker HOST` - MQTT broker address
- `--port PORT` - MQTT broker port (default: 1883)
- `--pos-rate HZ` - Position broadcast rate in Hz (default: 100)
- `--sig-rate HZ` - Signal broadcast rate in Hz (default: 100)

## MQTT Topics

### Published Topics

**Position telemetry** (QoS 0):
```
microscope/stage/position
Format: timestamp_ns/X/Y/Z/R
Example: 1699876543210000000/1000/2000/500/45000000
```

**Signal telemetry** (QoS 0):
```
picoammeter/current
Format: timestamp_ns/current_pA
Example: 1699876543210000000/5.237
```

**Command results** (QoS 1):
```
microscope/stage/result
Format: timestamp_ns/STATUS/CATEGORY/SUBCATEGORY/RESULT/details
```

### Command Topic

Subscribe to: `microscope/stage/command`

## Available Commands

### MOVE Command
Move a specific axis to a target position.

**Format**: `MOVE/<axis>/<value>`

**Examples**:
```bash
MOVE/X/5000        # Move X to 5000 nm
MOVE/Y/-3000       # Move Y to -3000 nm
MOVE/Z/750         # Move Z to 750 nm
MOVE/R/90000000    # Move R to 90 degrees (90,000,000 micro-degrees)
```

**Axes**:
- `X` - X position in nanometers
- `Y` - Y position in nanometers
- `Z` - Z position in nanometers
- `R` - Rotation in micro-degrees (1 degree = 1,000,000 micro-degrees)

## Example MQTT Commands

Using `mosquitto_pub`:

```bash
# Move to origin
mosquitto_pub -h localhost -t microscope/stage/command -m "MOVE/X/0"
mosquitto_pub -h localhost -t microscope/stage/command -m "MOVE/Y/0"

# Rotate sample
mosquitto_pub -h localhost -t microscope/stage/command -m "MOVE/R/45000000"

# Change Z position
mosquitto_pub -h localhost -t microscope/stage/command -m "MOVE/Z/1000"

# Update center of rotation
mosquitto_pub -h localhost -t microscope/stage/command -m "SET_COR/500/500/250"

# Request status
mosquitto_pub -h localhost -t microscope/stage/command -m "STATUS"

# Change update rate
mosquitto_pub -h localhost -t microscope/stage/command -m "SET_RATE/2000"
```

## Coordinate System

- **Stage coordinates**: Global reference frame for X, Y, Z positions
- **Sample positioning**: Sample images shift in X based on Z position
- **Rotation**: Sample rotates around the center of rotation (COR)
- **X-Z compensation**: Both sample position and COR shift in X as Z changes

### X-Z Compensation
The simulator applies independent X shifts to both the sample image and the COR:
- **Image shift**: `image_x = sample_center_x_base + (Z * x_per_z_ratio)`
- **COR shift**: `cor_x_effective = cor_x + ((Z - cor_z) * x_per_z_ratio)`

## Signal Generation

The picoammeter signal is generated based on:
1. Current stage position (X, Y, Z, R)
2. Sample image intensity at that position
3. Rotation around the COR
4. Out-of-bounds detection (returns 0 signal outside sample)

**Formula**: `current_pA = offset_pa + gain_pa × normalized_intensity`

Where `normalized_intensity` is 0.0-1.0 from the image pixel value.

## Stopping the Simulator

Press `Ctrl+C` to gracefully stop the simulator.

## Tips

1. **Z-stack spacing**: Use `--z-positions` to specify exact Z planes for each image
2. **FOV calibration**: Set `--fov-x` and `--fov-y` to match real image dimensions
3. **Smooth rotation**: Adjust `--speed-r` for realistic rotation speeds
4. **Signal calibration**: Tune `--gain-pa` and `--offset-pa` to match expected signal levels
5. **Performance**: Reduce `--pos-rate` and `--sig-rate` if CPU usage is too high

## Troubleshooting

- **No MQTT connection**: Verify broker is running and address/port are correct
- **Zero signal everywhere**: Check sample center positioning and FOV settings
- **Jerky motion**: Increase `--pos-rate` for smoother movement updates
- **Images not loading**: Verify image paths and formats (PNG, JPEG supported)

# 2&3. Scanning Microscope Control System

A Python-based control system for automated scanning microscopy with real-time data acquisition, visualization, and analysis. The system supports 2D raster scanning, 1D line scans, Z-series scanning, and multi-dimensional scan sequences.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Debugging Guide](#debugging-guide)
- [Data Formats](#data-formats)

## Overview

This project provides a comprehensive scanning microscope control system that:
- Controls stage movement via MQTT protocol
- Acquires position and signal data in real-time
- Supports multiple scan patterns (raster, snake, bidirectional)
- Provides both GUI and command-line interfaces
- Stores data in SQLite, HDF5, and CSV formats
- Includes live visualization and post-processing tools

## Project Structure

```
Old gui
├── scanner.py                    # The Old GUI application (11,429 lines)
│                                 # - Full-featured Qt5 interface
│                                 # - Real-time plotting and visualization
│                                 # - Scan configuration and execution
│                                 # - Data export and analysis tools

New API
├── run_scan.py                   # Command-line scan execution (a test field for using terminal command to run scans)
│                                 # - Headless scan execution
│                                 # - 2D, 1D, and Z-series support
│                                 # - Argument-based configuration
│
├── scan_controller.py            # Scan workflow orchestration (790 lines)
│                                 # - Pattern generation integration
│                                 # - Stage movement coordination
│                                 # - Data acquisition callbacks
│                                 # - Metadata management
│
├── series_scan_controller.py    # Multi-scan sequencing (32 KB)
│                                 # - 2D image Z-series/R-series scanning
|                                 # - multiZ scanning
│                                 # - Multi-dimensional sequences
│                                 # - Z-compensation support
│
├── scan_patterns.py              # Trajectory generation (21 KB)
│                                 # - Raster patterns
│                                 # - Snake patterns
│                                 # - Bidirectional scanning
│                                 # - Vertices-based polygons
│
├── data_storager.py              # Data persistence (20 KB)
│                                 # - SQLite database management
│                                 # - HDF5 raw data storage
│                                 # - CSV export
│                                 # - Metadata tracking
│
├── live_plotter.py               # Real-time visualization (11 KB)
│                                 # - Live 2D image updates
│                                 # - Line profile plotting
│                                 # - Thread-safe data buffering
│
├── acquisition_primitives.py    # Low-level acquisition (13 KB)
│                                 # - Stage positioning
│                                 # - Signal measurement
│                                 # - Movement verification
│                                 # - Tolerance checking
│
├── mqtt_client.py                # MQTT communication (7 KB)
│                                 # - Broker connection management
│                                 # - Message publishing/subscribing
│                                 # - Position and signal callbacks
│
├── command_sender.py             # Command interface (6.5 KB)
│                                 # - High-level movement commands
│                                 # - Stage control abstraction
│                                 # - MQTT message formatting
│
├── position_signal_reader.py    # Data stream processing (13 KB)
│                                 # - Position tracking
│                                 # - Signal buffering
│                                 # - History management
│                                 # - Averaging and filtering
│
├── utils.py                      # Utility functions (11 KB)
│                                 # - Z-compensation calculations
│                                 # - Coordinate transformations
│                                 # - Helper functions
│
└── __init__.py                   # Package initialization
```

## Key Features

### Scan Types
- **2D Scans**: Rectangular or polygon-based regions with configurable step sizes
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

## Prerequisites

### Required Software
- Python 3.7+
- MQTT broker (e.g., Mosquitto)
- Qt5 libraries (for GUI)

### Python Dependencies

```txt
numpy>=1.19.0
matplotlib>=3.3.0
PyQt5>=5.15.0
pyqtgraph>=0.12.0
paho-mqtt>=1.5.0
h5py>=2.10.0
scipy>=1.5.0
scikit-image>=0.17.0
sqlite3 (included in Python standard library)
```

### Optional Dependencies
- `scipy`: Advanced interpolation and analysis (recommended)
- `scikit-image`: Image registration features
- `h5py`: Raw data storage in HDF5 format

## Installation

### 1. Clone or Download the Project

```bash
# Download the project files to your local machine
cd /path/to/project
```

### 2. Set Up Python Environment

# Install dependencies
pip install numpy matplotlib PyQt5 pyqtgraph paho-mqtt h5py scipy scikit-image
```

## Usage

### GUI Application

Launch the full-featured integrated graphical interface:

```bash
python3 Old Gui/scanner.py
```

### Command-Line Interface

#### 2D Rectangular Scan

```bash
python3 run_scan.py 2d \
  --x-range -5000 5000 \
  --y-range -5000 5000 \
  --x-step 100 \
  --y-step 100 \
  --z-setpoint 1000 \
  --r-setpoint 45000 \
  --output ./data/scan_output.db
```

#### 2D Polygon Scan (Vertices)

```bash
python3 run_scan.py 2d \
  --vertices "(-500,0)" "(500,0)" "(0,866)" \
  --x-step 50 \
  --y-step 50 \
  --z-setpoint 1000 \
  --output ./data/triangle_scan.db
```

#### 1D Line Scan

```bash
python3 run_scan.py 1d \
  --start -1000 -1000 \
  --end 1000 1000 \
  --step 10 \
  --z-setpoint 1000 \
  --output ./data/line_scan.db
```

#### Z-Series 2D Scan

```bash
python3 run_scan.py z-series \
  --x-range -2000 2000 \
  --y-range -2000 2000 \
  --x-step 100 \
  --y-step 100 \
  --z-start 0 \
  --z-end 5000 \
  --z-steps 50 \
  --output ./data/z_series.db
```

### Common Options

```bash
--mqtt-host HOSTNAME      # MQTT broker hostname (default: localhost)
--mqtt-port PORT          # MQTT broker port (default: 1883)
--settle-tol TOLERANCE    # Position settle tolerance in nm (default: 5.0)
--settle-time SECONDS     # Time to wait for settling (default: 0.5)
--avg-count N             # Signal averaging count (default: 10)
--output PATH             # Output database file path
```

## Architecture

### Communication Flow

```
MQTT Broker
    ↕
mqtt_client.py (Connection Management)
    ↕
position_signal_reader.py (Data Stream Processing)
    ↕
scan_controller.py (Orchestration)
    ↕
acquisition_primitives.py (Low-level Control)
    ↕
command_sender.py (Command Interface)
    ↕
data_storager.py (Persistence)
```

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

**scan_data table:**
```sql
CREATE TABLE scan_data (
    scan_id TEXT,
    point_index INTEGER,
    x_nm REAL,
    y_nm REAL,
    z_nm REAL,
    signal REAL,
    FOREIGN KEY (scan_id) REFERENCES scans(scan_id)
);
```

### HDF5 Structure

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

### CSV Export Format

```csv
scan_id,point_index,x_nm,y_nm,z_nm,signal,timestamp
scan_001,0,0.0,0.0,1000.0,123.45,2025-01-15T10:30:00
scan_001,1,100.0,0.0,1000.0,124.56,2025-01-15T10:30:01
...
```

