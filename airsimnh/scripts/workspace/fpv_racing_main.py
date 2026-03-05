import time
import airsim
import numpy as np
import cv2
import sys

# Import modular components
from kalman_filter import KalmanFilterIMU
from gate_detector import RobustGateDetector
from imu_processor import IMUDataProcessor

class FPVRacingController:
    """
    FPV racing drone controller using only IMU and camera data.
    
    Components:
    - Kalman filter for state estimation
    - Robust gate detector with temporal filtering
    - IMU processor with attitude estimation
    """
    
    def __init__(self, ip="127.0.0.1", port=41451):
        self.client = airsim.MultirotorClient(ip=ip, port=port)
        self.client.confirmConnection()
        
        # State estimation
        self.kf = KalmanFilterIMU(dt=0.01)
        self.imu_proc = IMUDataProcessor(buffer_size=10)
        
        # Vision-based navigation
        self.gate_detector = RobustGateDetector(buffer_size=5)
        
        # Racing statistics
        self.gates_passed = 0
        self.race_start_time = None
        self.frame_count = 0
        self.last_gate_time = 0
        self.gate_timeout = 3.0  # seconds
        
        # Control parameters
        self.cruise_speed = 5.0
        self.max_steering = 1.5
        self.search_amplitude = 0.5
        
    def start(self):
        """Initialize drone and start race"""
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        print("Taking off...")
        self.client.takeoffAsync(timeout_sec=10).join()
        self.client.moveToZAsync(-2.0, 1.5).join()
        
        self.race_start_time = time.time()
        print("Race started! Using IMU + Vision only\n")
    
    def get_camera_frame(self):
        """Capture RGB frame from front camera"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])
            
            if responses and len(responses) > 0:
                response = responses[0]
                img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img = img_1d.reshape(response.height, response.width, 3)
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[Camera] Error: {e}")
        
        return None
    
    def control_loop(self, duration=60.0):
        """Main racing loop"""
        start_time = time.time()
        last_print = start_time
        
        print(f"Racing for {duration} seconds\n")
        
        while time.time() - start_time < duration:
            self.frame_count += 1
            current_time = time.time()
            
            # Get IMU data
            imu_data = self.client.getImuData()
            self.imu_proc.add_measurement(imu_data)
            
            # Get magnetometer data separately (if available)
            try:
                mag_data = self.client.getMagnetometerData()
                mag_heading = np.arctan2(mag_data.magnetic_field.y_val, 
                                        mag_data.magnetic_field.x_val)
            except:
                # Fallback if magnetometer not available
                mag_heading = 0.0
            
            # Update Kalman filter with IMU
            accel = self.imu_proc.get_filtered_accel()
            self.kf.predict(accel)
            self.kf.update_bias(accel)
            
            # Update attitude estimates
            self.imu_proc.update_attitude(dt=0.01, accel=accel, mag_heading=mag_heading)
            
            # Get camera frame and detect gates
            frame = self.get_camera_frame()
            gate_center, gate_area, gate_conf = self.gate_detector.detect_gate_center(frame)
            
            # Correct position with vision measurements
            if gate_center is not None and frame is not None:
                # Estimate 3D position from 2D gate location
                vision_position = self._estimate_position_from_vision(
                    gate_center, gate_area, frame.shape
                )
                self.kf.correct_with_vision(vision_position, confidence=gate_conf)
            
            # Calculate desired direction to gate
            desired_direction, in_search_mode = self.gate_detector.get_desired_direction(
                gate_center, frame.shape[:2] if frame is not None else (480, 640), gate_conf
            )
            
            # Generate control commands
            velocity_cmd = self._calculate_control(
                desired_direction, gate_area, in_search_mode, self.frame_count
            )
            
            # Execute movement
            self.client.moveByVelocityAsync(
                velocity_cmd['vx'],
                velocity_cmd['vy'],
                velocity_cmd['vz'],
                0.05  # Short update interval
            ).join()
            
            # Gate detection logic
            if gate_area > 40000 and (current_time - self.last_gate_time) > self.gate_timeout:
                self.gates_passed += 1
                self.last_gate_time = current_time
            
            # Periodic status print
            if current_time - last_print > 2.0:
                state = self.kf.get_state()
                imu_health = self.imu_proc.get_imu_health()
                attitude = self.imu_proc.get_attitude()
                
                print(f"[{self.frame_count:05d}] Gates: {self.gates_passed} | "
                      f"Conf: {gate_conf:.2f} | "
                      f"V: {np.linalg.norm(state['velocity']):.1f}m/s | "
                      f"Roll: {np.degrees(attitude['roll']):.1f}° | "
                      f"Cal: {imu_health['calibrated']}")
                last_print = current_time
            
            time.sleep(0.01)
    
    def _estimate_position_from_vision(self, gate_center, gate_area, frame_shape):
        """
        Estimate 3D position of gate from 2D image location.
        
        Simplified model: assumes gate is at fixed distance based on image size.
        """
        h, w = frame_shape[:2]
        
        # Estimate distance to gate based on apparent size
        # Assume gate is 2m x 2m in real world
        gate_size_real = 2.0
        focal_length = 500  # pixels (typical for simulation camera)
        
        distance = (gate_size_real * focal_length) / np.sqrt(gate_area)
        distance = np.clip(distance, 1.0, 20.0)  # Reasonable bounds
        
        # Convert 2D image coords to direction angles
        cx, cy = gate_center
        image_cx, image_cy = w / 2, h / 2
        
        # Small angle approximation
        angle_x = (cx - image_cx) / focal_length
        angle_y = (cy - image_cy) / focal_length
        
        # Estimate 3D position relative to drone
        # Assuming drone is roughly level and facing the gate
        x = distance * np.sin(angle_x)
        y = distance * np.sin(angle_y)
        z = -distance * np.cos(angle_x) * np.cos(angle_y)
        
        return np.array([x, y, z])
    
    def _calculate_control(self, desired_direction, gate_area, in_search_mode, frame_count):
        """Calculate velocity commands from vision and IMU data"""
        vx, vy, vz = 0.0, 0.0, 0.0
        
        if in_search_mode:
            # Search pattern if no gate detected
            vx = self.cruise_speed * 0.6
            vy = self.search_amplitude * np.sin(frame_count * 0.05)
            vz = self.search_amplitude * np.cos(frame_count * 0.05) * 0.3
        else:
            # Track gate
            dx, dy = desired_direction
            
            # Determine forward speed based on gate size/distance
            if gate_area > 50000:
                # Close to gate, slow down
                forward_speed = 2.0
                approach_confidence = 0.8
            elif gate_area > 20000:
                # Medium distance
                forward_speed = self.cruise_speed * 0.8
                approach_confidence = 0.6
            else:
                # Far, full speed
                forward_speed = self.cruise_speed
                approach_confidence = 0.4
            
            # Steering commands
            vx = forward_speed
            vy = self.max_steering * dx * approach_confidence
            vz = self.max_steering * dy * approach_confidence * 0.5
        
        return {'vx': vx, 'vy': vy, 'vz': vz}
    
    def print_race_summary(self):
        """Print final race statistics"""
        elapsed = time.time() - self.race_start_time
        state = self.kf.get_state()
        covariance = self.kf.get_covariance()
        attitude = self.imu_proc.get_attitude()
        
        print("\n" + "=" * 80)
        print("RACE SUMMARY - IMU + VISION ONLY")
        print("=" * 80)
        print(f"\nTiming:")
        print(f"  Elapsed Time: {elapsed:.2f} seconds")
        print(f"  Total Frames: {self.frame_count}")
        print(f"  Frame Rate: {self.frame_count/elapsed:.1f} Hz")
        
        print(f"\nGates:")
        print(f"  Gates Passed: {self.gates_passed}")
        print(f"  Gate Rate: {self.gates_passed/elapsed:.2f} gates/sec")
        
        print(f"\nState Estimation:")
        print(f"  Final Position: X={state['position'][0]:.2f}m, Y={state['position'][1]:.2f}m, Z={state['position'][2]:.2f}m")
        print(f"  Final Velocity: {np.linalg.norm(state['velocity']):.2f} m/s")
        print(f"  Position Uncertainty: {covariance['position_std']}")
        print(f"  Velocity Uncertainty: {covariance['velocity_std']}")
        
        print(f"\nAttitude:")
        print(f"  Roll: {np.degrees(attitude['roll']):.1f}°")
        print(f"  Pitch: {np.degrees(attitude['pitch']):.1f}°")
        print(f"  Yaw: {np.degrees(attitude['yaw']):.1f}°")
        
        print("=" * 80)
    
    def land_and_disarm(self):
        """Land drone safely"""
        print("\nLanding...")
        self.client.hoverAsync().join()
        time.sleep(1)
        self.client.landAsync(timeout_sec=15).join()
        
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("Disarmed.")

class _Tee:
    """Writes to both a file and the original stream simultaneously."""
    def __init__(self, stream, filepath):
        self._stream = stream
        self._file = open(filepath, 'w', buffering=1)  # line-buffered

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def main():
    import sys
    import os

    debug_path = os.path.join(os.path.dirname(__file__), "debug.log")
    tee_out = _Tee(sys.stdout, debug_path)
    tee_err = _Tee(sys.stderr, debug_path)
    sys.stdout = tee_out
    sys.stderr = tee_err

    racer = FPVRacingController()

    try:
        racer.start()
        racer.control_loop(duration=60.0)
        racer.print_race_summary()
    except KeyboardInterrupt:
        print("\n\nRace interrupted by user")
    except Exception as e:
        print(f"\n\nError during race: {e}")
    finally:
        racer.land_and_disarm()
        sys.stdout = tee_out._stream
        sys.stderr = tee_err._stream
        tee_out.close()
        tee_err.close()

if __name__ == "__main__":
    main()
