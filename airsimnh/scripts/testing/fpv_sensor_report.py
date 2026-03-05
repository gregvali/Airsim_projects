import time
import airsim
import json

c = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
c.confirmConnection()

c.enableApiControl(True)
c.armDisarm(True)

c.takeoffAsync(timeout_sec=10).join()
c.moveToZAsync(-2.0, 1.5).join()

# Perform a simple maneuver
c.moveByVelocityAsync(3.0, 0.8, 0.0, 2.0).join()

c.hoverAsync().join()
time.sleep(1)

# Collect and print all available sensor data
print("=" * 80)
print("AIRSIM MULTIROTOR SENSOR AND STATE DATA REPORT")
print("=" * 80)

# Get vehicle state
state = c.getMultirotorState()
print("\n--- VEHICLE STATE ---")
print(f"Position: X={state.kinematics_estimated.position.x_val:.3f}, Y={state.kinematics_estimated.position.y_val:.3f}, Z={state.kinematics_estimated.position.z_val:.3f}")
print(f"Velocity: X={state.kinematics_estimated.linear_velocity.x_val:.3f}, Y={state.kinematics_estimated.linear_velocity.y_val:.3f}, Z={state.kinematics_estimated.linear_velocity.z_val:.3f}")
print(f"Angular Velocity: X={state.kinematics_estimated.angular_velocity.x_val:.3f}, Y={state.kinematics_estimated.angular_velocity.y_val:.3f}, Z={state.kinematics_estimated.angular_velocity.z_val:.3f}")
print(f"Orientation (Quaternion): W={state.kinematics_estimated.orientation.w_val:.3f}, X={state.kinematics_estimated.orientation.x_val:.3f}, Y={state.kinematics_estimated.orientation.y_val:.3f}, Z={state.kinematics_estimated.orientation.z_val:.3f}")

# Get IMU data
imu_data = c.getImuData()
print("\n--- IMU DATA ---")
print(f"Accelerometer: X={imu_data.linear_acceleration.x_val:.3f}, Y={imu_data.linear_acceleration.y_val:.3f}, Z={imu_data.linear_acceleration.z_val:.3f} m/s²")
print(f"Gyroscope: X={imu_data.angular_velocity.x_val:.3f}, Y={imu_data.angular_velocity.y_val:.3f}, Z={imu_data.angular_velocity.z_val:.3f} rad/s")
print(f"Magnetic Field: X={imu_data.magnetic_field.x_val:.3f}, Y={imu_data.magnetic_field.y_val:.3f}, Z={imu_data.magnetic_field.z_val:.3f}")
print(f"Barometer Altitude: {imu_data.barometer.altitude:.3f} m")
print(f"Barometer Pressure: {imu_data.barometer.pressure:.3f} Pa")
print(f"Time Stamp: {imu_data.time_stamp} ns")

# Get barometer data
baro = c.getBarometerData()
print("\n--- BAROMETER DATA ---")
print(f"Altitude: {baro.altitude:.3f} m")
print(f"Pressure: {baro.pressure:.3f} Pa")
print(f"Qnh: {baro.qnh:.3f}")
print(f"Time Stamp: {baro.time_stamp} ns")

# Get GPS data
gps = c.getGpsData()
print("\n--- GPS DATA ---")
print(f"Latitude: {gps.gnss.geo_point.latitude:.6f}")
print(f"Longitude: {gps.gnss.geo_point.longitude:.6f}")
print(f"Altitude: {gps.gnss.geo_point.altitude:.3f} m")
print(f"Eph: {gps.gnss.eph:.3f} m")
print(f"Epv: {gps.gnss.epv:.3f} m")
print(f"Num Sat: {gps.gnss.num_sat}")
print(f"Fix Type: {gps.gnss.fix_type}")
print(f"Velocity: X={gps.gnss.velocity.x_val:.3f}, Y={gps.gnss.velocity.y_val:.3f}, Z={gps.gnss.velocity.z_val:.3f} m/s")
print(f"Time Stamp: {gps.time_stamp} ns")

# Get magnetometer data
mag = c.getMagnetometerData()
print("\n--- MAGNETOMETER DATA ---")
print(f"Magnetic Field: X={mag.magnetic_field.x_val:.3f}, Y={mag.magnetic_field.y_val:.3f}, Z={mag.magnetic_field.z_val:.3f}")
print(f"Time Stamp: {mag.time_stamp} ns")

# Get distance sensor data
distance = c.getDistanceSensorData()
print("\n--- DISTANCE SENSOR DATA ---")
print(f"Current Distance: {distance.current_distance:.3f} m")
print(f"Min Distance: {distance.min_distance:.3f} m")
print(f"Max Distance: {distance.max_distance:.3f} m")
print(f"Range: {distance.range:.3f} m")
print(f"Relative Position: X={distance.relative_pose.position.x_val:.3f}, Y={distance.relative_pose.position.y_val:.3f}, Z={distance.relative_pose.position.z_val:.3f}")
print(f"Time Stamp: {distance.time_stamp} ns")

# Get landed state
landed = c.getMultirotorState().landed_state
print("\n--- LANDED STATE ---")
print(f"Landed State: {landed}")

# Get rotor speeds (if available)
try:
    rotor_speeds = c.getRotorSpeeds()
    print("\n--- ROTOR SPEEDS ---")
    if rotor_speeds:
        for i, speed in enumerate(rotor_speeds.speeds):
            print(f"Rotor {i}: {speed:.1f} RPM")
except:
    print("\n--- ROTOR SPEEDS ---")
    print("Rotor speed data not available")

# Get collision info
collisions = c.simGetCollisionInfo()
print("\n--- COLLISION INFO ---")
print(f"Has Collided: {collisions.has_collided}")
print(f"Collision Count: {collisions.collision_count}")
print(f"Collision Position: X={collisions.collision_position.x_val:.3f}, Y={collisions.collision_position.y_val:.3f}, Z={collisions.collision_position.z_val:.3f}")
print(f"Collision Normal: X={collisions.collision_normal.x_val:.3f}, Y={collisions.collision_normal.y_val:.3f}, Z={collisions.collision_normal.z_val:.3f}")
print(f"Penetration Depth: {collisions.penetration_depth:.3f}")
print(f"Object Name: {collisions.object_name}")

# Get camera info
print("\n--- CAMERA INFO ---")
camera_list = c.simListSceneObjects(airsim.ObjectFilter(".*Camera.*"))
print(f"Available Cameras: {camera_list}")

# Attempt to get image data from front camera
try:
    camera_data = c.simGetImage("0", airsim.ImageType.Scene)
    if camera_data:
        print(f"Front Camera (Scene): Data received, size={len(camera_data)} bytes")
except:
    print("Front camera data not available")

print("\n" + "=" * 80)
print("REPORT COMPLETE")
print("=" * 80)

c.landAsync(timeout_sec=15).join()

c.armDisarm(False)
c.enableApiControl(False)
