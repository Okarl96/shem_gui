# A-SHeM Controlling Development

This project aims to develop pure Python-based code to run the scanning on A-SHeM.

You can see three folders in the project:

## Simulator: 
By its name, it is a simulation code that outputs position and signal stream in the exact format as real A-SHeM. It helps you to develop the scanning code without accidentally breaking the real instrument. It interpolates JPG images as the "sample" and mimics almost all behaviors we have met in real experiments, including moving the sample in XYZ, rotating the sample around a stated center of rotation, and applying drifts in linear axes,etc. You can find a full description below. 


# ECC Pico Simulator

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

### Required Arguments
- `--images IMAGE [IMAGE ...]` - Image file paths (PNG, JPEG, etc.)
- `--broker HOST` - MQTT broker address

### Image Configuration
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
