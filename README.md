# AI-Assisted Control System for Scanning Helium Microscopy (SHeM)

This repository contains the software framework described in:
AI-designed and AI-implemented Control Systems for Bespoke Scientific Instrumentation: Application to Scanning Microscopy
It provides a control system for a Scanning Helium Microscope (SHeM)(https://doi.org/10.1016/j.nimb.2014.06.028), along with a sandbox simulation environment used for validation prior to deployment on physical hardware.

## Scope and Clarification
This project **does not implement or train any machine learning models**.

- No model training, fine-tuning, or datasets are used.
- Large Language Models (LLMs) were used **only during development** as external tools for code generation.
- All code in this repository runs deterministically and locally.

This repository therefore supports AI-assisted software engineering, not AI inference or training.

## Repository Contents
The repository includes:

**Sandbox simulation environment**
  - Emulates instrument behaviour using synthetic input data
  - Allows validation of control logic without hardware access

**Control system script**
  - Communication layer (MQTT-based messaging)
  - Scan control logic (trajectory generation, dwell control, position verification)
  - User interfaces

## Inputs and Outputs
### Inputs

**sandbox simulation** works on:

Synthetic sample configuration
   - PNG/JPEG image used as a virtual sample surface in the sandbox simulation
   - configurable position and size of the synthetic sample

Virtual stage Configuration
   - configurable MQTT client settings
   - configurable speed and global coordinates for the sample movement

Detials of how to configure the sandbox is provided below; there will a reproduction short-cut right after this section.

**control script** works on:

Scan parameters
  - User needs to specify scan type, size, resolution, and dwell time for the scan.

Detials of the GUI is provided below; there will a reproduction short-cut right after this section.

### Outputs
**sandbox simulation** outputs:
  - Time-stamped data streams via MQTT client which is subscribed by the control script GUI or by manual subscription

**control scripT** GUI outputs:
  - 2D scan data or line scan data based on the user choice
  - All scan will be stored in PNG for easy access, HDF5 for raw data processing, CSV for simple post-processing

## Minimal Reproducible Example

### 1. Install dependencies

Both scripts have uv(https://docs.astral.sh/uv/) commands embeded which state the required package, will create a virtual environment, and automatiacally install packages upon running. 

To install the uv in Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart the powershell after installing.

### 2. Run the sandbox simulator

```bash
uv run  ecc_pico_simulator.py --images img_z0.png img_z250.png img_z500.png img_z750.png img_z1000.png --z-positions 0 250 500 750 1000 --broker localhost --port 1883 --pos-rate 100 --sig-rate 500 --fov-x 1280 --fov-y 960 --speed-xy 10000 --speed-z 10000 --sample-center-x 0 --sample-center-y 0
```

You will see packages downloading and installing. Then there will be messages showing successful loading of images and parameters.

### 2. Run the control GUI

```bash
 uv run scanner.py
```
You will see packages downloading and installing. Then there will be a GUI pops out.

<img width="1919" height="1049" alt="easy step 1" src="https://github.com/user-attachments/assets/4cf27116-26a0-4eff-9a0c-16494310f67f" />

<img width="1918" height="1041" alt="easy step 2" src="https://github.com/user-attachments/assets/d5e9c3c8-5417-44fa-b4b0-1c6b6dbb1a14" />

After the scan finish or manually clicking the "stop", there will be an image viewer pops out.

<img width="1317" height="592" alt="easy step 3" src="https://github.com/user-attachments/assets/109a05bb-a48a-4807-a917-f6b271861880" />

As you can see from the figure, the control GUI has scanned over the input digital image and produced a "crude" version of it since we are using step=40. In this example, we have verified the MQTT connection, the user interface response for the basic 2D control and 2D image live display, the scan parameter settings and scan logic, and data points forming and storage. This example reproduces the **sandbox validation workflow described in the paper**. To exactly reproduce the figure in the paper, we need to reduce the step to 10 or lower for a fine scan which will take much longer than this example. Experimental SHeM data shown in the paper requires physical instrument access and is not included.

## Control Script Structure

<img width="1815" height="841" alt="structure" src="https://github.com/user-attachments/assets/834edcea-f5c8-4784-8521-7df3e2f3bce0" />

The figure above illustrates the internal architecture of the control script. While the system handles multiple complex tasks, from low-level hardware communication to high-level image reconstruction, it is designed with a strict logical separation of concerns.

The architecture is divided into four primary functional blocks:

  - Hardware Status and Communication: acquire positions and signals streaming in the MQTT client and also check the status of the hardware
  - Scan Control: The orchestration layer that takes user inputs as scanning parameters and then generates trajectories, verifies positions, and handles dwell times for various scan types. It also stores and outputs position and signal data acquired during valid dwell window.
  - User Interface (GUI): Captures user inputs for easy access, displays live results for monitoring during real experiments, provides image viewer after a complete scan for quick review.
  - The Execution Function: as needed for all Python applications

You will notice that this entire architecture is distributed as a single Python script. This is an intentional design choice aimed at maximizing ease of deployment in laboratory environments. Setting up custom package management, linking multiple physical files, and configuring virtual environments can be a significant barrier to entry for experimental physicists who are not specialized software engineers. By consolidating the control system into one file, we prioritize rapid deployment and the immediate production of scientific results.

Note on Maintainability: while this single-file approach drastically lowers the barrier to entry and successfully drives the experimental hardware, it represents a trade-off. Consolidating the codebase introduces technical debt, making long-term maintenance and automated type-checking more difficult. However, by strictly adhering to the logical modularity outlined above (heavily commented and physically grouped within the script), the codebase remains conceptually organized, allowing users and AI assistants to navigate and modify distinct functional blocks effectively.


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

Another way to check the simulator is running is to manually subscribe to the MQTT client using:

```bash
 mosquitto_sub -h localhost -p 1883 -t "microscope/stage/position"   
```

where you will see a streaming of timestamp/X/Y/Z/R, all positions will be 0 for now if you are using default settings. You can keep the subscribtion and see changes of positions if you send commands by manually post command messages or using the control script.

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

Upon successful running, you will see:
<img width="1920" height="1037" alt="image" src="https://github.com/user-attachments/assets/aeed642d-4333-4149-81df-109e4a991d91" />

## Usage ##

**Region 1:** The connection and storage setting, you can simply click `Connect` to start the connection to the MQTT client if using the default setting. The `Path` is the file saving path, and the `Detector Lag` is used in the real microscope to compensate the detetcor sensitivity which is usually 0.35 seconds. Upon successful connection, all N/A numbers in ***Region 3*** will become the current live reading of positions and signal like this:
<img width="549" height="159" alt="image" src="https://github.com/user-attachments/assets/3d4c68c7-fb72-4dd0-990c-10ca380c934c" />

and the status on the bottom right will also become:

<img width="114" height="25" alt="image" src="https://github.com/user-attachments/assets/fefb01c3-d3d8-421d-b60f-286bb281a987" />


**Region 2:** The most important parameter setting area. There are three tabs: `Manual Control`, `2D control`, and `1D control`. 

The `Manual Control` is used to run the axes manually to a designated position, we usually use it to test the connetion and bring the sample to a optically good region for the initial scan. The big read `STOP` buttom is used to force stop all movement, which is been tested. You can test it by: first simply assign a large number to any axe and see updates on the positions in ***Region 3*** or in the MQTT client if you have manually subscribed to it; then click `STOP`, which will stop all movements. 

The `2D control` contains all parameters needed for acquiring 2D images:
<img width="797" height="546" alt="image" src="https://github.com/user-attachments/assets/f6cfe7c1-5cee-4928-8c4a-438ea37cd2fb" />

- The X and Y parameters are: start position, end position, pixel number and step which is the resolution in nanometers. The pixel number and the step are linked so that as soon as you adjust one of them, the other will be calculated based on your start and end point; when the length of X or Y cannot be divided exactly by your pixel number or step, the system takes conservative choice and results in a actually smaller scanning region which will be displayed in the `Effective FOV` (Field of view) below.
- `Mode` now only has "Step-Stop" working as we are still develop the "Continuous" mode. The "Step-Stop" is means the we move to a certain postion, dwell for the `Dwell Time` stated below, and form the pixel value based on the average signals received during the dwell window. The "Continuous" mode aims to acquire sigal continuously so that we are not wasting any signal during the scan, the `Scan Speed` is the parameter for this mode, which is under construction.
- `Pattern` is the moving pattern choice which has "Snake" and "Raster". As picture speaks thousands of words, the movement pattern is shown in ***Region 4*** if the `Show Path Preview` is toggled on.
- `Z-Series Parameters` is the settings for the serie scanning, which means we will have mutiple images one by one at different Z positions. The `Base Z` the Z position of a image of choice we take as the reference image, and we want to scan this area at different Z positions to see if any changes due to research purpose. The `Z start, end, numbers, and step` all follow the same logic as X and Y. The `X and Y Compensation Ratio` is due to incident beam geometry of the real microscope which is already described above in the simulator introduction.
- `R-Series Parameters` as similar to the Z-Series, we want to examine the same area on the sample at different orientations. In `Mode`, "Simple Rotation" means we just rotate the R and keep the X and Y as the same before, which is usually used when we are doing large scale scan or crude scan; `COR Transform` needs the user to state the position of the center of rotation below. Since we are not always lucky enough to have the region of interest just on the cenetr of rotation, we need to change X and Y to scan the same area at different orientations as we are using global coordinates.
- You can only change R or Z series parameters after toggling on the `Enable` at the top left of each series scan panel. Thers is only one series scan can be activated, the other will be force deactivated if you try to toggle both of them.

The `1D control` contains all parameters needed for acquiring line scans:
<img width="800" height="472" alt="image" src="https://github.com/user-attachments/assets/11496d04-07db-4869-97ff-ecd2c7439311" />

- The `Mode` has "Line Scan", "Z-Scan", and "R+Z Series". The "R+Z Series" is still under contruction, which aims to have R and Z series scan at the same time for acquring full diffraction pattern.
- "Line Scan" needs the user to state a fixed axis and then state the `start, end, points, step, and Dwell time` similar as before.
- "Z-Scan" aims to keep incident on the same spot and acquiring a line scan so that we have signal vs Z, which can be converted to signal vs angle to get the diffraction pattern. In this case, we have a point of interest from a reference image, which will be the `Fixed X and Y`, the `Base Z` is the Z position of the reference point. The other parameters are similar to the 2D Z-series scan.

**Region 4:** The live display area where the `2D Image` tab corresponds to `2D control`, `1D Plot` corresponds to `1D Control`, `R+Z Heatmap` corresponds to "R+Z Series" in `1D control` (under construction). Each tab is independent to each other.

**Image Process:** After either 2D or 1D scan, there will be a pop up window like this:
<img width="998" height="728" alt="image" src="https://github.com/user-attachments/assets/709ab8c6-d210-46bf-a3d7-fb8256d15191" />
which is mostly the matlabplot library functions.

- Based on the image you are openning, loading a 2D image will enable `Image View` and `Metadata` tabs, a line scan will enable `Line Plot` and `Metadata`, and the `Polar View` is under consturction aimming for viewing the full diffraction pattern.
- The `Select ROI` region of interest is for selecting a region of interest after crude scan (it is a common senario in real experiment). After drawing the ROI, the user can send the parameters of the ROI to the `2D Control` for convinence.
- The `Overlay Mode` is made for viewing mutiple line scans in the same plot for comparision.

### Code Work Flow

1. MQTT Connection: System connects to broker and subscribes to topics
2. Configuration: User sets scan parameters via GUI
3. Pattern Generation: Scan trajectory is calculated
4. Scan Execution: 
   - Move to each point in pattern
   - Wait for settling
   - Acquire signal
   - Store data
5. Data Storage: Results saved to PNG/HDF5/CSV
6. Visualization: Real-time or post-scan plotting

### Data Storage
- **HDF5**: Raw scan data
- **CSV**: Exported scan results for external analysis
- **PNG**: Scan images for easy access

***HDF5 Structure***

```
scan_id.h5
- metadata (scan_type, timestamp, x_range_nm, y_range_nm, ...)
- raw_data (currents,x,y,z,r,time_stamp)
- reconstructed_image (image, x_coord, y_coord)
```
where the x_coord and y_coord are the x and y positions of each pixel in the image, while the raw_data stores all datapoints before averaging into pixels.

***CSV Export Format***

```csv
scan_type, scan_id, timestamp, scan_parameters
X positions, Y positions, Signal
...
```
where the CSV file only stores the x and y positions of each pixel; it will be a huge file if all raw positions are stored in CSV format.
