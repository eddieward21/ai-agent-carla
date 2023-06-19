import carla 
import math 
import random 
import time 
import numpy as np
import cv2

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
    
# Set initial camera translation
camera_init_trans = carla.Transform(carla.Location(z=2))

# Add one of each type of camera
# cameras in this file are rgb, semantic segmentation, instance segmentation, and depth
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1280')
camera_bp.set_attribute('image_size_y', '1024')
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

sem_camera_bp = bp_lib.find('sensor.camera.semantic_segmentation') 
sem_camera_bp.set_attribute('image_size_x', '1280')
sem_camera_bp.set_attribute('image_size_y', '1024')
sem_camera = world.spawn_actor(sem_camera_bp, camera_init_trans, attach_to=vehicle)

inst_camera_bp = bp_lib.find('sensor.camera.instance_segmentation') 
inst_camera_bp.set_attribute('image_size_x', '1280')
inst_camera_bp.set_attribute('image_size_y', '1024')
inst_camera = world.spawn_actor(inst_camera_bp, camera_init_trans, attach_to=vehicle)

depth_camera_bp = bp_lib.find('sensor.camera.depth') 
depth_camera_bp.set_attribute('image_size_x', '1280')
depth_camera_bp.set_attribute('image_size_y', '1024')
depth_camera = world.spawn_actor(depth_camera_bp, camera_init_trans, attach_to=vehicle)

sensor_data = {
    'dvs_image': np.zeros((image_h, image_w, 4))
}

# rgb camera
camera.listen(lambda image: image.save_to_disk('rgb/%06d.png' % image.frame))

# semantic segmentation camera
cc = carla.ColorConverter.CityScapesPalette
sem_camera.listen(lambda image: image.save_to_disk('sem/%06d.png' % image.frame, cc))

# instance segmentation camera
inst_camera.listen(lambda image: image.save_to_disk('inst/%06d.png' % image.frame))

# depth camera
dp = carla.ColorConverter.LogarithmicDepth
depth_camera.listen(lambda image: image.save_to_disk('depth/%06d.png' % image.frame, dp))

camera.stop()
sem_camera.stop()
inst_camera.stop()
depth_camera.stop()
