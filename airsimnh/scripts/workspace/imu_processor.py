import numpy as np
from collections import deque

class IMUDataProcessor:
    """
    Processes IMU data with:
    - Noise filtering
    - Bias estimation
    - Gyroscope integration for attitude
    - Sensor fusion preparation
    """
    
    def __init__(self, buffer_size=10):
        # Raw measurements
        self.accel_buffer = deque(maxlen=buffer_size)
        self.gyro_buffer = deque(maxlen=buffer_size)
        self.mag_buffer = deque(maxlen=buffer_size)
        
        # Estimated biases
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        
        # Attitude (roll, pitch, yaw) from gyro integration
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        # Complementary filter gain
        self.alpha = 0.98  # Higher = trust gyro more, lower = trust accel more
        
        # Calibration state
        self.is_calibrated = False
        self.calibration_samples = 100
        self.calibration_count = 0
        
    def add_measurement(self, imu_data):
        """Add raw IMU measurement"""
        accel = np.array([
            imu_data.linear_acceleration.x_val,
            imu_data.linear_acceleration.y_val,
            imu_data.linear_acceleration.z_val,
        ])
        
        gyro = np.array([
            imu_data.angular_velocity.x_val,
            imu_data.angular_velocity.y_val,
            imu_data.angular_velocity.z_val,
        ])
        
        # Magnetometer data - use if available
        try:
            mag = np.array([
                imu_data.magnetic_field.x_val,
                imu_data.magnetic_field.y_val,
                imu_data.magnetic_field.z_val,
            ])
        except (AttributeError, TypeError):
            # Fallback if magnetic field not in IMU data
            mag = np.array([0.0, 0.0, 0.0])
        
        self.accel_buffer.append(accel)
        self.gyro_buffer.append(gyro)
        self.mag_buffer.append(mag)
        
        # Online bias estimation during initialization
        if not self.is_calibrated and self.calibration_count < self.calibration_samples:
            self._calibrate_online(accel, gyro)
            self.calibration_count += 1
        elif not self.is_calibrated:
            self.is_calibrated = True
            print(f"[IMU] Calibration complete. Gyro bias: {self.gyro_bias}")
    
    def _calibrate_online(self, accel, gyro):
        """Running average for gyro bias estimation"""
        # Assume drone is roughly stationary during initialization
        # Gyro should be near zero if calibrated, difference is bias
        n = self.calibration_count + 1
        self.gyro_bias = ((n - 1) * self.gyro_bias + gyro) / n
    
    def get_filtered_accel(self):
        """Get low-pass filtered acceleration"""
        if len(self.accel_buffer) == 0:
            return np.zeros(3)
        
        accel_array = np.array(list(self.accel_buffer))
        
        # Moving average filter
        filtered = np.mean(accel_array, axis=0)
        
        # Subtract estimated bias
        filtered -= self.accel_bias
        
        return filtered
    
    def get_filtered_gyro(self):
        """Get low-pass filtered angular velocity"""
        if len(self.gyro_buffer) == 0:
            return np.zeros(3)
        
        gyro_array = np.array(list(self.gyro_buffer))
        
        # Moving average filter
        filtered = np.mean(gyro_array, axis=0)
        
        # Subtract estimated bias
        filtered -= self.gyro_bias
        
        return filtered
    
    def get_magnetometer_heading(self):
        """
        Get yaw/heading from magnetometer.
        
        Returns heading in radians (-pi to pi)
        Falls back to 0 if no mag data available.
        """
        if len(self.mag_buffer) == 0:
            return 0.0
        
        mag_array = np.array(list(self.mag_buffer))
        
        # Check if all zeros (no mag data)
        if np.allclose(mag_array, 0.0):
            return 0.0
        
        mag_mean = np.mean(mag_array, axis=0)
        
        # X-Y plane magnetic field gives heading
        # Handle case where both X and Y are zero
        if np.allclose(mag_mean[:2], 0.0):
            return 0.0
        
        heading = np.arctan2(mag_mean[1], mag_mean[0])
        
        return heading
    
    def update_attitude(self, dt, accel=None, mag_heading=None):
        """
        Update attitude estimates using complementary filter.
        
        Combines:
        - Gyro integration (high frequency, drifts over time)
        - Accelerometer (leveling, noisy)
        - Magnetometer (yaw reference)
        """
        if accel is None:
            accel = self.get_filtered_accel()
        
        gyro = self.get_filtered_gyro()
        
        # Gyro integration (angular velocity)
        self.roll += gyro[0] * dt
        self.pitch += gyro[1] * dt
        self.yaw += gyro[2] * dt
        
        # Accelerometer-based roll/pitch (gravity reference)
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1:  # Only if significant acceleration
            accel_roll = np.arctan2(accel[1], accel[2])
            accel_pitch = np.arctan2(-accel[0], 
                                     np.sqrt(accel[1]**2 + accel[2]**2))
            
            # Complementary filter: blend gyro and accel
            self.roll = self.alpha * self.roll + (1 - self.alpha) * accel_roll
            self.pitch = self.alpha * self.pitch + (1 - self.alpha) * accel_pitch
        
        # Magnetometer yaw update
        if mag_heading is not None:
            mag_alpha = 0.95  # Higher trust in gyro for yaw
            self.yaw = mag_alpha * self.yaw + (1 - mag_alpha) * mag_heading
        
        # Clamp angles to [-pi, pi]
        self.roll = self._clamp_angle(self.roll)
        self.pitch = self._clamp_angle(self.pitch)
        self.yaw = self._clamp_angle(self.yaw)
    
    @staticmethod
    def _clamp_angle(angle):
        """Wrap angle to [-pi, pi]"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def get_attitude(self):
        """Return current attitude (roll, pitch, yaw) in radians"""
        return {
            'roll': self.roll,
            'pitch': self.pitch,
            'yaw': self.yaw,
        }
    
    def get_imu_health(self):
        """Return IMU data quality metrics"""
        accel = self.get_filtered_accel()
        gyro = self.get_filtered_gyro()
        
        return {
            'accel_magnitude': np.linalg.norm(accel),
            'gyro_magnitude': np.linalg.norm(gyro),
            'calibrated': self.is_calibrated,
            'buffer_size': len(self.accel_buffer),
        }