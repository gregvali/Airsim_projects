import time
import airsim

c = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
c.confirmConnection()

c.enableApiControl(True)
c.armDisarm(True)

c.takeoffAsync(timeout_sec=10).join()
c.moveToZAsync(-2.0, 1.5).join()

# Fly square path with altitude changes
# Side length: 10 meters, altitude range: -2m to -5m
altitude_low = -2.0
altitude_high = -5.0
side_length = 10.0
velocity = 5.0
duration_per_side = side_length / velocity

# Square path: right, forward, left, back
directions = [
    (side_length, 0.0, altitude_high),   # Right and down
    (0.0, side_length, altitude_low),    # Forward and up
    (-side_length, 0.0, altitude_high),  # Left and down
    (0.0, -side_length, altitude_low),   # Back and up
]

for x, y, z in directions:
    c.moveByVelocityAsync(x / duration_per_side, y / duration_per_side, z / duration_per_side, duration_per_side).join()
    time.sleep(0.5)

c.hoverAsync().join()
time.sleep(1)
c.landAsync(timeout_sec=15).join()

c.armDisarm(False)
c.enableApiControl(False)
print("FPV square path smoke test complete")
