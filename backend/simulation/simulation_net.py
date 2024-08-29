import os
import pickle
import sys

from input_ocel_processing.process_config import ProcessConfig
from object_model_generation.object_instance import ObjectInstance, ScheduledActivity
from object_model_generation.object_model import ObjectModel
from ocpn_discovery.net_utils import Place, NetProjections, Transition
from .sim_utils import Token, Marking
from object_model_generation.object_instance import SimulationObjectInstance
from .state_space_computer import StateSpaceComputer
#import resource

class SimulationNet:

    @classmethod
    def load(cls, session_path, object_model_name):
        simulation_net_path = os.path.join(session_path, "simulation_net_ " + object_model_name + ".pkl")
        return pickle.load(open(simulation_net_path, "rb"))

    marking: Marking
    netProjections: NetProjections
    stateSpaceComputer: StateSpaceComputer
    objects: ObjectModel
    objectModelName: str
    terminatingObjects: set
    processConfig: ProcessConfig

    def __init__(self, session_path, net_projections, marking, simulation_objects, object_model_name: str = ""):
        self.processConfig = ProcessConfig.load(session_path)
        self.objectModelName = object_model_name
        self.netProjections = net_projections
        self.marking = marking
        self.simulationObjects = simulation_objects
        self.objects = ObjectModel.load(session_path, self.processConfig.useOriginalMarking, object_model_name)
        self.stateSpaceComputer = StateSpaceComputer(self.processConfig, net_projections)
        self.sessionPath = session_path
        self.otypes = self.processConfig.otypes
        self.acts = self.processConfig.acts
        self.activityLeadingTypes = self.processConfig.activityLeadingTypes
        self.terminatingObjects = set()

    def save(self):
        simulation_net_path = os.path.join(self.sessionPath, "simulation_net_ " + self.objectModelName + ".pkl")
        with open(simulation_net_path, "wb") as write_file:
            success = False
            while not success and sys.getrecursionlimit() < 20000:
                try:
                    pickle.dump(self, write_file)
                    success = True
                except RecursionError:
                    print("Recursion error, increasing limit " + str(sys.getrecursionlimit()) + "...")
                    sys.setrecursionlimit(sys.getrecursionlimit() + 100)
            if success:
                print("Successfully saved simulation net.")
            else:
                raise RecursionError

    def fire(self, transition_name, objects):
        obj: ObjectInstance
        otypes = {obj.otype for obj in objects}
        bound_objects_by_otype = {otype: set() for otype in otypes}
        for obj in objects:
            bound_objects_by_otype[obj.otype].add(obj)
        preset_indices_by_otype = {}
        postset_indices_by_otype = {}
        currentScheduledActivity: ScheduledActivity
        if transition_name in self.acts:
            leading_type = self.processConfig.activityLeadingTypes[transition_name]
            leading_obj = list(filter(lambda obj: obj.otype == leading_type, objects))[0]
            leading_sim_obj: SimulationObjectInstance = self.simulationObjects[leading_obj.oid]
            currentScheduledActivity = leading_sim_obj.nextActivity
            execution_time = currentScheduledActivity.time
        token: Token
        transition: Transition
        for otype in otypes:
            # Check binding validity and determine pre- and postsets
            projected_net = self.netProjections.get_otype_projection(otype)
            transition = [t for t in projected_net.transitions if t.id == transition_name][0]
            t_index = projected_net.transitions.index(transition)
            places = projected_net.places
            Tin = projected_net.Tin
            Tout = projected_net.Tout
            preset_indices  = {i for i in range(len(Tin))  if Tin[i][t_index]  == -1}
            postset_indices = {i for i in range(len(Tout)) if Tout[i][t_index] == 1}
            # check if each place in the preset of the transition has each bound object of the type
            bound_objects = bound_objects_by_otype[otype]
            for preset_index in preset_indices:
                for obj in bound_objects:
                    place = places[preset_index]
                    if not any(token.oid == obj.oid for token in self.marking.tokensByOtype[otype]):
                        raise AssertionError("Invalid binding when trying to fire a transition: token" + str(obj.oid) \
                                             + " missing at place '" + place.id + "'.")
            preset_indices_by_otype[otype] = preset_indices
            postset_indices_by_otype[otype] = postset_indices
        token_removals_by_place = {}
        token_removals_by_otype = {}
        token_removals = []
        for otype in otypes:
            token_removals_by_otype[otype] = []
            # deduct tokens from preset and find maximal time of bound tokens
            projected_net = self.netProjections.get_otype_projection(otype)
            places = projected_net.places
            for obj in bound_objects_by_otype[otype]:
                bound_tokens = []
                place: Place
                for preset_index in preset_indices_by_otype[otype]:
                    place = places[preset_index]
                    old_preset = self.marking.tokensByPlaces[place]
                    obj_in_old_preset = list(filter(lambda token: token.oid == obj.oid, old_preset))
                    bound_token: Token = obj_in_old_preset[0]
                    bound_tokens.append(bound_token)
                    if place not in token_removals_by_place:
                        token_removals_by_place[place] =[]
                    token_removals.append(bound_token)
                    token_removals_by_place[place].append(bound_token)
                    token_removals_by_otype[otype].append(bound_token)
        self.marking.remove_tokens(token_removals, token_removals_by_place, token_removals_by_otype)
        aobj: None
        for otype in otypes:
            for obj in bound_objects_by_otype[otype]:
                aobj = obj
                place: Place
                simulation_object: SimulationObjectInstance = self.simulationObjects[obj.oid]
                simulation_object.active = False
                if transition_name in self.acts:
                    simulation_object.lastActivity = transition_name
                    simulation_object.nextActivity = None
        for otype in otypes:
            projected_net = self.netProjections.get_otype_projection(otype)
            places = projected_net.places
            for obj in bound_objects_by_otype[otype]:
                simulation_object: SimulationObjectInstance = self.simulationObjects[obj.oid]
                if transition_name in self.acts:
                    obj.time = execution_time
                else:
                    execution_time = obj.time
                for postset_index in postset_indices_by_otype[otype]:
                    place = places[postset_index]
                    new_token = Token(obj.oid, otype, execution_time, place)
                    self.marking.add_token(new_token)
                if otype not in self.processConfig.nonEmittingTypes:
                    self.marking.update_object_time(obj.oid, execution_time)
                    simulation_object.time = execution_time
        print(
            "Firing " + transition_name + \
            " at " + str(execution_time) + \
            ", object: " + str(aobj.otype) + " "+ str(aobj.oid)
        )
        return execution_time

    def get_all_running_emitting_tokens(self):
        all_running_tokens = []
        for otype in self.otypes:
            places = self.netProjections.get_otype_projection(otype).places
            otype_final_places = [place for place in places if place.isFinal]
            otype_tokens = self.marking.tokensByOtype[otype]
            all_running_tokens += [token for token in otype_tokens
                                   if token.place not in otype_final_places
                                   and token.otype not in self.processConfig.nonEmittingTypes]
        return all_running_tokens

    def get_all_simulation_objects(self):
        return self.simulationObjects

    def get_all_active_simulation_objects(self):
        return [sim_obj for sim_obj in self.simulationObjects.values() if sim_obj.active]

    def get_simulation_objects_by_ids(self, oids):
        return [sim_obj for sim_obj in self.simulationObjects.values() if sim_obj.oid in oids]

    # TODO: refactor
    def getAllTransitions(self):
        all_transitions = []
        for otype in self.processConfig.otypes:
            all_transitions += self.netProjections.get_otype_projection(otype).transitions
        return list(set(all_transitions))

    def find_feasible_next_activities(self, obj):
        return self.stateSpaceComputer.find_feasible_next_activities(self.marking, obj)

    def compute_path(self, model_obj, next_activity):
        return self.stateSpaceComputer.compute_path(self.marking, next_activity, model_obj)

    def initialize_simulation_objects(self):

        pass
