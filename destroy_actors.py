import carla
import random
import math
import time

client = carla.Client("localhost", 2000)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

actor_list = world.get_actors()

# Iterate over each actor and destroy it
for actor in actor_list:
    actor.destroy()

for v in world.get_actors():
    v.destroy()

