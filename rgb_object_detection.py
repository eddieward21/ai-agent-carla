import numpy as np
import cv2 
from matplotlib import pyplot as plt
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

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
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1280')
camera_bp.set_attribute('image_size_y', '1024')
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

# Define respective callbacks
# Initialise parameters and data
image_w = 1280
image_h = 1024

sensor_data = {
    'rgb_image': np.zeros((image_h, image_w, 4)),
}


import requests
import os



def rgb_callback(image, data_dict):

    i = np.array(image.raw_data)
    i2 = i.reshape((image_h, image_w, 4))
    i3 = i2[:,:,:3]
    cap = cv2.VideoCapture()
    while cap.IsOpened():
        results = model(sensor_data['rgb_image'])
        cv2.imshow("All cameras", np.squeeze(results.render()))
        if cv2.waitKey(1) == ord('q'):
            break
    data_dict['rgb_image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    return i3 / 255.0

camera.listen(lambda image: rgb_callback(image, sensor_data))



camera.stop()
cv2.destroyAllWindows()