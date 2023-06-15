import carla
import math
import random
import time
import numpy as np
import cv2
client = carla.Client('localhost', 2000)
world = client.get_world()
bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
vehicle_bp = bp_lib.find('vehicle.mercedes.coupe')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
spectator = world.get_spectator()
transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)), vehicle.get_transform().rotation)
spectator.set_transform(transform)
for i in range(30):
    vehicle_bp = random.choice(bp_lib.filter('vehicle'))
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    
for v in world.get_actors().filter('*vehicle*'):
    v.set_autopilot(True)
    
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1280')
camera_bp.set_attribute('image_size_y', '1024')

camera_init_trans = carla.Transform(carla.Location(z=1.6,x=0.4))

camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

camera.listen(lambda image: image.save_to_disk('out/%06d.png' % image.frame))

#camera.stop()
