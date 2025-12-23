"""
test_drone.py - Basic quadcopter control test for CMSE443
This script demonstrates real-time API control of the drone simulator.
"""

import cosysairsim as airsim
import time

def main():
    # Step 1: Connect to the simulator
    # The simulator runs an RPC server on localhost:41451
    print("Connecting to AirSim simulator...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected successfully!")
    
    # Step 2: Enable API control and arm the drone
    # "Arming" means the motors are ready to spin
    client.enableApiControl(True)
    client.armDisarm(True)
    print("Drone armed and ready.")
    
    # Step 3: Take off to 3 meters altitude
    # takeoffAsync() returns a Future object; join() waits for completion
    print("Taking off...")
    client.takeoffAsync().join()
    print("Takeoff complete!")
    
    # Step 4: Move to a position (NED coordinates: North, East, Down)
    # NED means: +X = North, +Y = East, +Z = Down (negative Z = up)
    # moveToPositionAsync(x, y, z, velocity)
    print("Moving to position (10, 0, -5)...")
    client.moveToPositionAsync(10, 0, -5, 5).join()
    print("Reached target position!")
    
    # Step 5: Hover for 3 seconds
    print("Hovering for 3 seconds...")
    time.sleep(3)
    
    # Step 6: Return to start position
    print("Returning to start...")
    client.moveToPositionAsync(0, 0, -3, 5).join()
    
    # Step 7: Land
    print("Landing...")
    client.landAsync().join()
    print("Landed successfully!")
    
    # Step 8: Disarm and release control
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Test complete!")

if __name__ == "__main__":
    main()