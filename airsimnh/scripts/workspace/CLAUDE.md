# AirSim FPV Drone Racing - Claude Context

## Project Goal
Simulate a drone flying and estimating its own state using **only camera + IMU** (no GPS, no ground truth).
Long-term goal: autonomous racetrack navigation using RL.

## Stack
- **Simulator**: AirSim (Unreal Engine), connected via Python API (`airsim` package)
- **Language**: Python 3
- **Key deps**: `numpy`, `opencv-python`, `scipy`, `airsim`
- **Run**: `python fpv_racing_main.py` from this directory

## Architecture

```
IMU (accel + gyro)
    → IMUDataProcessor   - noise filtering, bias calibration, complementary filter attitude
    → KalmanFilterIMU    - 6-state (pos + vel) prediction via kinematics

Camera (RGB)
    → RobustGateDetector - HSV color detection, temporal median filter, confidence score
    → KalmanFilterIMU.correct_with_vision() - position correction measurement update

KalmanFilter state → FPVRacingController._calculate_control() → velocity commands → AirSim
```

## Files
| File | Purpose |
|------|---------|
| `fpv_racing_main.py` | Top-level controller, AirSim interface, main loop |
| `kalman_filter.py` | 6-state Kalman filter (IMU predict + vision correct) |
| `imu_processor.py` | IMU filtering, bias calibration, complementary filter |
| `gate_detector.py` | HSV gate detection with temporal filtering + optical flow |

## Key Parameters to Tune

### KalmanFilterIMU (`kalman_filter.py`)
- `self.Q` — process noise; increase if state drifts, decrease if too noisy
- `self.R_vision` — vision measurement noise; decrease to trust vision more
- `self.bias_learning_rate` — how fast to adapt IMU bias (default 0.01)

### IMUDataProcessor (`imu_processor.py`)
- `self.alpha` — complementary filter blend: 0.98 = trust gyro more, 0.8 = trust accel more
- `buffer_size` — moving average window (default 10 samples)
- `self.calibration_samples` — stationary samples used for gyro bias init (default 100)

### RobustGateDetector (`gate_detector.py`)
- `self.gate_colors` — HSV thresholds for red/green/blue gates; tune to match AirSim gate colors
- `self.min_contour_area` — minimum pixel area to consider a gate detection (default 200)
- `self.min_detection_confidence` — threshold below which gate is ignored (default 0.6)

### FPVRacingController (`fpv_racing_main.py`)
- `self.cruise_speed` — forward speed in m/s (default 5.0)
- `self.max_steering` — max lateral velocity command (default 1.5)
- `self.search_amplitude` — weave pattern size when no gate visible (default 0.5)
- Gate area thresholds: 50000 = close (slow down), 40000 = passed gate

## Current Limitations / Known Issues
- Gate detection relies on colored gates — not robust to occlusion or lighting changes
- Position estimate is relative, not absolute (drifts over long flights)
- `calculate_optical_flow()` in gate_detector is implemented but not wired into the control loop
- Vision position estimation (`_estimate_position_from_vision`) assumes `focal_length=500` px — verify against AirSim camera settings
- Bias estimation in `update_bias()` assumes near-hover (gravity in -Z); may need adjustment for aggressive maneuvers

## Next Steps / Future Work
- [ ] Wire optical flow into state estimation for velocity correction
- [ ] Tune gate color HSV thresholds against actual AirSim environment
- [ ] Verify camera focal length / FOV settings match AirSim config
- [ ] Add gate sequence tracking (ordered waypoints)
- [ ] RL policy for racing (replace `_calculate_control` with learned policy)

## AirSim Notes
- Default connection: `127.0.0.1:41451`
- Drone starts at altitude 0, takeoff targets `-2.0m` (NED frame, negative = up)
- IMU data via `client.getImuData()`, camera via `client.simGetImages()`
- Magnetometer may not be available in all AirSim configs — code falls back to `mag_heading=0`
