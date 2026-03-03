import os
import time

import airsimdroneracinglab as airsim


# Change this to your own ADRL executable path if you want this script to launch ADRL for you.
# Example: ADRL_EXE_PATH = r"C:\Users\greg\Documents\AirSim\ADRL.exe"
ADRL_EXE_PATH = r"C:\Users\greg\Documents\AirSim\ADRL.exe"
LEVEL_NAME = "Soccer_Field_Easy"


def wait_for_level_ready(client, timeout_seconds=30):
    # After loading a level, wait until the simulator starts responding.
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            client.getMultirotorState()
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Level did not finish loading in time.")


def launch_simulator_if_available():
    # If ADRL_EXE_PATH is set and valid, start ADRL automatically.
    if ADRL_EXE_PATH and os.path.isfile(ADRL_EXE_PATH):
        os.startfile(ADRL_EXE_PATH)
        print("Launching ADRL...")
        # Small delay so ADRL has time to open before connecting.
        time.sleep(3)
    else:
        print("ADRL path not set or file not found. Start ADRL manually, then run this script.")


def main():
    # Step 1: Launch simulator (optional if already open).
    launch_simulator_if_available()

    # Step 2: Create a client object and connect to ADRL.
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim Drone Racing Lab.")

    # Step 3: Load a simple level and start a race session.
    client.simLoadLevel(LEVEL_NAME)
    wait_for_level_ready(client)
    client.simStartRace(tier=2)

    # Step 4: Give Python control of the drone and arm motors.
    client.enableApiControl()
    client.arm()
    print("API control enabled and drone armed.")

    # Step 5: Basic movement commands.
    # Format: moveByRollPitchYawrateThrottleAsync(roll, pitch, yaw_rate, throttle, duration_seconds)
    print("Takeoff...")
    client.moveByRollPitchYawrateThrottleAsync(0, 0, 0, 0.62, 3).join()

    print("Fly forward...")
    client.moveByRollPitchYawrateThrottleAsync(0, 1, 0, 0.67, 2).join()

    print("Climb a bit...")
    client.moveByRollPitchYawrateThrottleAsync(0, 0, 0, 0.77, 4).join()

    print("Tutorial flight complete.")

    # Step 6: Safety cleanup.
    # client.arm(False)
    # client.enableApiControl(False)
    # print("Disarmed and released API control.")


if __name__ == "__main__":
    # This lets you run the script directly with: python airsim_tutorial.py
    main()
