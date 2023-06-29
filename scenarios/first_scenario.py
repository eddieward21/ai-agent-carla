"""
test scenario:

This scenario is a pre-set route that the user must follow. The
user-controlled ego vehicle drives until they can take the first
available right turn. They must stop fully before taking this right.
There will be a vehicle stopped on the right side of the road, which 
the ego vehicle must avoid. Finally, they must stop again before the 
roundabout to have a successful run.
"""

import random

import py_trees

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      KeepVelocity,
                                                                      StopVehicle,
                                                                      WaypointFollower,
                                                                      Idle)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (CollisionTest, OutsideRouteLanesTest)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToVehicle,InTriggerDistanceToNextIntersection,DriveDistance,StandStill,WaitUntilInFront,AtRightmostLane)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_waypoint_in_distance

from srunner.tools.background_manager import Scenario2Manager


class test(BasicScenario):

    """
    This class holds everything required for a simple "Follow a leading vehicle"
    scenario involving two vehicles.  (Traffic Scenario 2)

    This is a single ego vehicle scenario
    """

    timeout = 70            # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=70):
        """
        Setup all relevant parameters and create scenario

        If randomize is True, the scenario parameters are randomized
        """

        self._map = CarlaDataProvider.get_map()
        self._first_vehicle_location = 80
        self._first_vehicle_speed = 10
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._other_actor_max_brake = 1.0
        self._other_actor_stop_in_front_intersection = 20
        self._other_actor_transform = None
        # Timeout of scenario in seconds
        self.timeout = timeout

        super(test, self).__init__("test",
                                                   ego_vehicles,
                                                   config,
                                                   world,
                                                   debug_mode,
                                                   criteria_enable=criteria_enable)

        if randomize:
            self._ego_other_distance_start = random.randint(4, 8)

            # Example code how to randomize start location
            # distance = random.randint(20, 80)
            # new_location, _ = get_location_in_distance(self.ego_vehicles[0], distance)
            # waypoint = CarlaDataProvider.get_map().get_waypoint(new_location)
            # waypoint.transform.location.z += 39
            # self.other_actors[0].set_transform(waypoint.transform)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        
        waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        transform = waypoint.transform
        first_vehicle = CarlaDataProvider.request_new_actor('vehicle.nissan.patrol', transform)
        self.other_actors.append(first_vehicle)
        

    def _create_behavior(self):
      
        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        endcondition_part1 = InTriggerDistanceToVehicle(self.other_actors[0], self.ego_vehicles[0], 
                                                    distance=60, name="FinalDistance")                                   
        endcondition_part2 = StandStill(self.ego_vehicles[0], name="FinalSpeed", duration=1)                                                
        endcondition_part3 = WaitUntilInFront(self.ego_vehicles[0], self.other_actors[0], 2)
        endcondition_part4 = StandStill(self.ego_vehicles[0], name="FinalSpeed2", duration=1)
        endcondition.add_child(endcondition_part1)
        endcondition.add_child(endcondition_part2)
        endcondition.add_child(endcondition_part3)
        endcondition.add_child(endcondition_part4)

        # Build behavior tree
        sequence = py_trees.composites.Sequence(name="Sequence Behavior", memory=False)
        sequence.add_child(endcondition)
        sequence.add_child(ActorDestroy(self.other_actors[0]))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        criteria = []

	      # if the ego vehicle has a collision, scenario fails
        collision_criterion = CollisionTest(self.ego_vehicles[0])
        
        # if the ego vehicle deviates outside of the right side of the road, fails (i wish)
        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
