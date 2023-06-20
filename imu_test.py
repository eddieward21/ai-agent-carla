import carla 
import math 
import random 
import time 
import numpy as np
import cv2
import sys


# Connect the client and set up bp library and spawn points
client = carla.Client('localhost', 2000) 
world = client.get_world()
bp_lib = world.get_blueprint_library()  
spawn_points = world.get_map().get_spawn_points() 

vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020') 
vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[79])

# Move the spectator behind the vehicle to view it
spectator = world.get_spectator() 
transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation) 
spectator.set_transform(transform)


# Add traffic
for i in range(50): 
    vehicle_bp = random.choice(bp_lib.filter('vehicle')) 
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    
for v in world.get_actors().filter('*vehicle*'):
    v.set_autopilot(True)
    
    
imu_bp = bp_lib.find('sensor.other.imu')
imu_transform = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(pitch=0, yaw=0, roll=0))
imu_sensor = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)

def imu_callback(imu_data):
    # Process the IMU data here
    # Example: Print the linear acceleration data
    print(imu_data.accelerometer)

imu_sensor.listen(imu_callback)

world.tick()  # Advance the simulation by one tick

for i in range (10):
    world.tick()
