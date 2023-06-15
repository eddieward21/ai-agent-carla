import carla


client = carla.Client("localhost", 2000)

client.reload_world()