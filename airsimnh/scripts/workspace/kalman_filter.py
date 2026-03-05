import numpy as np


class KalmanFilterIMU:
    """
    Kalman filter for drone state estimation using IMU data.

    State vector: [x, y, z, vx, vy, vz]
    Control input: accelerometer readings (bias-corrected)
    Measurements: vision-based position estimates

    IMU drives the prediction step as a control input (standard INS approach).
    Vision corrections are applied as position measurements via full Kalman update.
    """

    def __init__(self, dt=0.01):
        self.dt = dt

        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)
        self.x[2] = -2.0  # Start at altitude -2m

        # State covariance
        self.P = np.eye(6) * 0.1

        # Process noise (Q) - accounts for unmodeled accelerations
        self.Q = np.diag([0.01, 0.01, 0.01,   # Position process noise
                          0.1,  0.1,  0.1])     # Velocity process noise

        # Vision measurement noise (R) - scales inversely with confidence
        self.R_vision = np.diag([0.5, 0.5, 0.5])

        # State transition matrix (kinematics: p += v*dt)
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Control input matrix (acceleration → position and velocity)
        self.B = np.zeros((6, 3))
        self.B[0, 0] = 0.5 * dt ** 2  # x += 0.5*ax*dt^2
        self.B[1, 1] = 0.5 * dt ** 2
        self.B[2, 2] = 0.5 * dt ** 2
        self.B[3, 0] = dt              # vx += ax*dt
        self.B[4, 1] = dt
        self.B[5, 2] = dt

        # Measurement matrix for vision (observes position only)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # IMU bias estimates
        self.accel_bias = np.zeros(3)
        self.bias_learning_rate = 0.01

    def predict(self, accel_measurement):
        """Prediction step: propagate state using IMU acceleration as control input."""
        accel_corrected = accel_measurement - self.accel_bias

        # Clamp acceleration to physical limits
        accel_corrected = np.clip(accel_corrected, -50.0, 50.0)

        # State prediction: x = F*x + B*u
        self.x = self.F @ self.x + self.B @ accel_corrected

        # Clamp velocity to realistic limits
        max_v = 20.0
        v_norm = np.linalg.norm(self.x[3:6])
        if v_norm > max_v:
            self.x[3:6] = self.x[3:6] / v_norm * max_v

        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_bias(self, accel_measurement):
        """
        Online accelerometer bias estimation.

        Assumes near-hover: expected lateral acceleration is ~0,
        vertical is ~-9.81 m/s^2 (gravity in NED frame).
        """
        gravity = np.array([0.0, 0.0, -9.81])
        self.accel_bias += self.bias_learning_rate * (accel_measurement - gravity)
        self.accel_bias = np.clip(self.accel_bias, -1.0, 1.0)

    def correct_with_vision(self, position_measurement, confidence=0.5):
        """
        Correct position estimate using vision-based measurements.

        Args:
            position_measurement: Estimated 3D position from vision [x, y, z]
            confidence: Measurement confidence (0-1), higher = lower noise assumed
        """
        if position_measurement is None:
            return

        # Scale measurement noise inversely with confidence
        R = self.R_vision / max(confidence, 0.01)

        # Innovation: difference between measurement and predicted position
        y = position_measurement - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x += K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def get_state(self):
        """Return current state estimate."""
        return {
            'position': self.x[0:3].copy(),
            'velocity': self.x[3:6].copy(),
        }

    def get_covariance(self):
        """Return estimation uncertainty (standard deviations)."""
        return {
            'position_std': np.sqrt(np.diag(self.P[0:3, 0:3])),
            'velocity_std': np.sqrt(np.diag(self.P[3:6, 3:6])),
        }
