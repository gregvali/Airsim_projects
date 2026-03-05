import time
import airsim

c = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
c.confirmConnection()

c.enableApiControl(True)
c.armDisarm(True)

c.takeoffAsync(timeout_sec=10).join()
c.moveToZAsync(-2.0, 1.5).join()
c.moveByVelocityAsync(3.0, 0.8, 0.0, 2.0).join()

c.hoverAsync().join()
time.sleep(1)
c.landAsync(timeout_sec=15).join()

c.armDisarm(False)
c.enableApiControl(False)
print("FPV multirotor smoke test complete")