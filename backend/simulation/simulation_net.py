import os
import pickle

from input_ocel_processing.process_config import ProcessConfig
from object_model_generation.object_instance import ObjectInstance
from object_model_generation.object_model import ObjectModel
from ocpn_discovery.net_utils import Place, NetProjections
from .sim_utils import Token, Marking
from .simulation_object_instance import SimulationObjectInstance
from .state_space_computer import StateSpaceComputer


class SimulationNet:

    @classmethod
    def load(cls, session_path):
        simulation_net_path = os.path.join(session_path, "simulation_net.pkl")
        return pickle.load(open(simulation_net_path, "rb"))

    marking: Marking
    netProjections: NetProjections
    stateSpaceComputer: StateSpaceComputer
    objects: ObjectModel
    terminatingObjects: set
    processConfig: ProcessConfig

    def __init__(self, session_path, net_projections, marking, simulation_objects):
        self.processConfig = ProcessConfig.load(session_path)
        self.netProjections = net_projections
        self.marking = marking
        self.simulationObjects = simulation_objects
        self.objects = ObjectModel.load(session_path)
        self.stateSpaceComputer = StateSpaceComputer(self.processConfig, net_projections)
        self.sessionPath = session_path
        self.otypes = self.processConfig.otypes
        self.acts = self.processConfig.acts
        self.activityLeadingTypes = self.processConfig.activityLeadingTypes
        self.terminatingObjects = set()

    def save(self):
        simulation_net_path = os.path.join(self.sessionPath, "simulation_net.pkl")
        with open(simulation_net_path, "wb") as write_file:
            pickle.dump(self, write_file)

    def fire(self, transition_name, objects, delays: dict):
        obj: ObjectInstance
        otypes = {obj.otype for obj in objects}
        bound_objects_by_otype = {otype: set() for otype in otypes}
        for obj in objects:
            bound_objects_by_otype[obj.otype].add(obj)
        preset_indices_by_otype = {}
        postset_indices_by_otype = {}
        token: Token
        for otype in otypes:
            # Check binding validity and determine pre- and postsets
            projected_net = self.netProjections.get_otype_projection(otype)
            transition = [t for t in projected_net.transitions if t.id == transition_name][0]
            t_index = projected_net.transitions.index(transition)
            places = projected_net.places
            Tin = projected_net.Tin
            Tout = projected_net.Tout
            preset_indices = {i for i in range(len(Tin)) if Tin[i][t_index] == -1}
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
        firing_time = -1
        for otype in otypes:
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
                    firing_time = bound_token.time if bound_token.time > firing_time else firing_time
                    bound_tokens.append(bound_token)
                    self.marking.remove_token(place, bound_token)
        for otype in otypes:
            # add tokens to postsets with object-type specific delays
            projected_net = self.netProjections.get_otype_projection(otype)
            places = projected_net.places
            for obj in bound_objects_by_otype[otype]:
                place: Place
                new_tokens_time = firing_time + delays[obj.oid]
                for postset_index in postset_indices_by_otype[otype]:
                    place = places[postset_index]
                    new_token = Token(obj.oid, otype, new_tokens_time, place)
                    self.marking.add_token(new_token)
                if otype not in self.processConfig.nonEmittingTypes:
                    self.marking.update_object_time(obj.oid, new_tokens_time)
                obj.time = new_tokens_time
        return firing_time

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
        return list(filter(lambda sim_obj: sim_obj.active, self.simulationObjects))

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
