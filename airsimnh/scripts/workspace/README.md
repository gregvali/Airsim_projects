# FPV Racing Simulation - Modular Architecture

## Overview

A realistic FPV drone racing simulator that uses **only IMU and camera data** for state estimation and gate navigation. Designed to be sim-to-real compatible and easy to modify via Claude API.

## Module Structure

### 1. `kalman_filter.py` - State Estimation
**Purpose**: Estimates drone position and velocity from noisy IMU data

**Key Features**:
- 6-state Kalman filter (position + velocity)
- Online bias estimation for accelerometer
- Vision-based position correction
- Velocity clamping to realistic limits

**Main Methods**:
- `predict(accel_measurement)` - Predict step using acceleration
- `update(accel_measurement)` - Adaptive bias correction
- `correct_with_vision(position, confidence)` - Fuse vision measurements
- `get_state()` - Returns position and velocity estimates
- `get_covariance()` - Returns estimation uncertainty

**Tuning Parameters**:
- `self.Q` - Process noise (accounts for unmeasured accelerations)
- `self.R` - Measurement noise (accelerometer noise level)
- `self.bias_learning_rate` - How fast to adapt to IMU bias

---

### 2. `imu_processor.py` - IMU Data Processing
**Purpose**: Filters raw IMU data and estimates drone attitude

**Key Features**:
- Low-pass filtering via moving average
- Online gyro bias calibration
- Complementary filter for attitude estimation (gyro + accel)
- Magnetometer heading estimation
- IMU health monitoring

**Main Methods**:
- `add_measurement(imu_data)` - Add raw IMU reading
- `get_filtered_accel()` - Denoised acceleration
- `get_filtered_gyro()` - Denoised angular velocity
- `update_attitude(dt, accel, mag_heading)` - Estimate roll/pitch/yaw
- `get_attitude()` - Returns current attitude

**Tuning Parameters**:
- `self.alpha` - Complementary filter blend (0.98 = trust gyro more)
- Buffer size - Number of samples to average

---

### 3. `gate_detector.py` - Vision-Based Navigation
**Purpose**: Detects racing gates in camera feed with robustness to noise

**Key Features**:
- HSV color-based gate detection (red/green/blue)
- Morphological noise filtering
- Temporal filtering (gate tracking across frames)
- Optical flow calculation for motion analysis
- Confidence scoring

**Main Methods**:
- `detect_gate_center(image)` - Returns gate center, area, and confidence
- `get_desired_direction(gate_center, image_shape, confidence)` - Steering target
- `calculate_optical_flow(image)` - Motion magnitude and direction

**Tuning Parameters**:
- `self.min_contour_area` - Minimum gate size to detect
- `self.min_detection_confidence` - Confidence threshold for accepting detection
- HSV color thresholds in `self.gate_colors`

---

### 4. `fpv_racing_main.py` - Main Racing Controller
**Purpose**: Integrates all components and runs the race

**Key Features**:
- Combines Kalman filter + IMU processor + gate detector
- Implements racing control logic (approach, gate detection, search)
- Real-time status monitoring
- Comprehensive race summary

**Main Methods**:
- `start()` - Takeoff and initialize
- `control_loop(duration)` - Main racing loop
- `_calculate_control()` - Convert vision/IMU to velocity commands
- `print_race_summary()` - Print statistics

**Tuning Parameters**:
- `self.cruise_speed` - Target speed in m/s
- `self.max_steering` - Max lateral velocity command
- `self.search_amplitude` - Weaving pattern size when searching
- Gate detection size thresholds (40000, 50000 pixels)

---

## Data Flow

```
IMU Data → IMUProcessor (filter + attitude)
         ↓
    KalmanFilter (state estimation)
         ↓
    Corrected by Vision ←─ GateDetector ←─ Camera
         ↓
    Velocity Commands → Drone Control
```

## Usage

```bash
python fpv_racing_main.py
```

All modules must be in the same directory.

## Editing Guide for Claude API

Each module is independent and can be modified. Common edits:

**Adjust sensitivity:**
- Kalman filter: Modify `self.Q` and `self.R` matrices
- Gate detector: Change `min_detection_confidence` or HSV thresholds
- IMU: Adjust `self.alpha` (complementary filter blend)

**Change control behavior:**
- Racing controller: Edit `_calculate_control()` method
- Gate detector: Modify `get_desired_direction()` calculations

**Add new sensors/features:**
- Create new method in IMUDataProcessor
- Update KalmanFilter to use new data
- Integrate in FPVRacingController.control_loop()

---

## Realistic Aspects

✓ IMU integration drift (bounded by vision corrections)
✓ Sensor bias estimation and removal
✓ Attitude estimation from multiple sensor sources
✓ Temporal filtering to reduce noise
✓ Velocity and attitude constraints
✓ No ground-truth position data

## Known Limitations

- Vision detection relies on colored gates (not robust to occlusion)
- Position estimation is relative, not absolute GPS
- Assumes gates are perpendicular to flight path
- No wind/disturbance simulation (would require additional filters)
