import numpy as np

from input_ocel_processing.process_config import ProcessConfig
from object_model_generation.object_instance import ObjectInstance
from ocpn_discovery.net_utils import Place, Transition, Arc, NetProjections, TransitionType
from simulation.sim_utils import Marking


class Path:
    otype: str
    marking: np.array
    firing_sequence: []
    start_places: []
    bound_tokens: []

    def __init__(self, otype, marking, firing_sequence, start_places=None):
        self.otype = otype
        self.marking = marking
        self.firing_sequence = firing_sequence
        self.start_places = start_places


class StateSpaceComputer:
    projectedNets: NetProjections
    processConfig: ProcessConfig
    activityPlacePathMap: dict

    def __init__(self, process_config, net_projections):
        self.processConfig = process_config
        self.projectedNets = net_projections
        self.__compute_place_to_activity_reachabilities()

    def __compute_place_to_activity_reachabilities(self):
        self.activityPlacePathMap = dict()
        activity_transitions = []
        t: Transition
        for otype in self.processConfig.otypes:
            ts = self.projectedNets.get_otype_projection(otype).transitions
            activity_transitions += [t for t in ts
                                     if t.transitionType == TransitionType.ACTIVITY
                                     or t.transitionType == TransitionType.FINAL]
        activity_transitions = set(activity_transitions)
        for t in activity_transitions:
            activity = t.label
            self.activityPlacePathMap[activity] = {}
            self.__make_paths_map(t)

    def find_feasible_next_activities(self, marking: Marking, obj: ObjectInstance):
        otype = obj.otype
        net = self.projectedNets.get_otype_projection(otype)
        transitions = net.transitions
        places = net.places
        nof_transitions = len(transitions)
        Tin = net.Tin
        Tout = net.Tout
        N = Tin + Tout
        m0 = self.__get_marking_object_projection(marking, obj)
        p0 = Path(otype=otype, marking=m0, firing_sequence=[])
        complete_paths = []
        buffer = [p0]
        m: np.array
        p: Path
        handled_markings = []
        activity_type = TransitionType.ACTIVITY
        final_type = TransitionType.FINAL
        while len(buffer) > 0:
            p = buffer[0]
            buffer = buffer[1:]
            m = p.marking
            fs = p.firing_sequence
            handled_markings.append(m)
            enabled_transitions_indices = self.__get_enabled_transitions_indices(m, Tin)
            for index in enabled_transitions_indices:
                transition = transitions[index]
                t_vec = np.zeros(nof_transitions)
                t_vec[index] = 1
                m_new = m + np.dot(N, t_vec)
                # assumption: sound, in particular bounded net. otherwise replace >= with ==
                fs_new = fs + [transition]
                new_p = Path(otype, m_new, fs_new)
                transition_type = transition.transitionType
                if transition_type == activity_type or transition_type == final_type:
                    complete_paths += [new_p]
                    continue
                if any(all(i >= 0 for i in m_i - m_new) for m_i in handled_markings):
                    continue
                buffer.append(new_p)
        complete_path: Path
        shortest_paths = dict()
        for complete_path in complete_paths:
            fs = complete_path.firing_sequence
            activity = fs[-1].label
            if activity not in shortest_paths:
                shortest_paths[activity] = complete_path
            if len(fs) < len(shortest_paths[activity].firing_sequence):
                shortest_paths[activity] = complete_path
            firing_vector_delta = sum([N.transpose()[transitions.index(t)] for t in complete_path.firing_sequence])
            complete_path.start_places = {
                places[i]: abs(firing_vector_delta[i])
                for i in range(len(firing_vector_delta)) if firing_vector_delta[i] < 0
            }
            start_place_bound_tokens = {
                start_place: [t for t in marking.tokensByPlaces[start_place] if t.oid == obj.oid]
                for start_place in complete_path.start_places
            }
            complete_path.bound_tokens = start_place_bound_tokens
        return list(shortest_paths.values())

    # compute reachability of transition for a certain object id and return a firing sequence if reachable
    def compute_path(self, marking: Marking, activity: str, obj: ObjectInstance) -> []:
        otype = obj.otype
        net = self.projectedNets.get_otype_projection(otype)
        transitions = net.transitions
        nof_transitions = len(transitions)
        typed_places = net.places
        Tin = net.Tin
        Tout = net.Tout
        N = Tin + Tout
        m0 = self.__get_marking_object_projection(marking, obj)
        p0 = Path(otype=otype, marking=m0, firing_sequence=[])
        complete_paths = []
        buffer = [p0]
        m: np.array
        p: Path
        handled_markings = []
        who_can_reach_target = []
        initially_marked = [place for place in typed_places if m0[typed_places.index(place)] > 0]
        for place in initially_marked:
            if place.id not in self.activityPlacePathMap[activity]:
                continue
            # syntax-based informed search (who can reach target via a path of edges)
            who_can_reach_target += self.activityPlacePathMap[activity][place.id]
        while len(buffer) > 0:
            p = buffer[0]
            buffer = buffer[1:]
            m = p.marking
            fs = p.firing_sequence
            handled_markings.append(m)
            enabled_transitions_indices = self.__get_enabled_transitions_indices(m, Tin)
            for index in enabled_transitions_indices:
                transition = transitions[index]
                t_vec = (np.array(list(map(lambda i: 1 if i == index else 0, range(nof_transitions)))))
                m_new = m + np.dot(N, t_vec)
                # assumption: sound, in particular bounded net. otherwise replace >= with ==
                fs_new = fs + [transition]
                new_p = Path(otype, m_new, fs_new)
                if transition.label == activity:
                    complete_paths += [new_p]
                    continue
                if any(all(i >= 0 for i in m_i - m_new) for m_i in handled_markings):
                    continue
                if transition.label in self.processConfig.acts or transition not in who_can_reach_target:
                    continue
                buffer.append(new_p)
        complete_paths.sort(key=lambda p: len(p.firing_sequence))
        if len(complete_paths) == 0:
            return None
        complete_path: Path = complete_paths[0]
        firing_vector_delta = sum([N.transpose()[transitions.index(t)] for t in complete_path.firing_sequence])
        complete_path.start_places = {
            typed_places[i]: abs(firing_vector_delta[i])
            for i in range(len(firing_vector_delta)) if firing_vector_delta[i] < 0
        }
        start_place_bound_tokens = {
            start_place: [t for t in marking.tokensByPlaces[start_place] if t.oid == obj.oid]
            for start_place in complete_path.start_places
        }
        if any(len(ts) > 1 for ts in start_place_bound_tokens.values()):
            pass#raise ValueError("Warning: More than 1 token in 1 place for 1 object bound for a firing sequence."
                 #            + "Are you sure this is correct, developer? Please assess & fix code.")
        complete_path.bound_tokens = start_place_bound_tokens
        return complete_path

    def __make_paths_map(self, t: Transition):
        arc: Arc
        activity = t.label
        path = {t}
        handled_nodes = {t}
        in_places = list(map(lambda arc: arc.placeEnd, t.incomingArcs))
        for in_place in in_places:
            in_place: Place
            self.__trace_back_from_place(activity, in_place, path, handled_nodes)

    def __trace_back_from_place(self, activity, place: Place, path, handled_nodes):
        if place in handled_nodes:
            return
        self.__update_path(activity, place, path)
        arc: Arc
        in_transitions = list(map(lambda arc: arc.transEnd, place.incomingArcs))
        in_transition: Transition
        for in_transition in in_transitions:
            if in_transition in handled_nodes \
                    or in_transition.transitionType == TransitionType.ACTIVITY \
                    or in_transition.transitionType == TransitionType.FINAL:
                continue
            new_path = set(list(path))
            new_path.add(in_transition)
            new_handled_nodes = set(list(handled_nodes))
            new_handled_nodes.add(place)
            new_handled_nodes.add(in_transition)
            in_places = list(map(lambda arc: arc.placeEnd, in_transition.incomingArcs))
            for in_place in in_places:
                self.__trace_back_from_place(activity, in_place, new_path, new_handled_nodes)

    def __update_path(self, activity, place: Place, path):
        path_map = self.activityPlacePathMap[activity]
        if place.id not in path_map:
            path_map[place.id] = set()
        path_map[place.id].update(path)

    def __get_marking_object_projection(self, marking: Marking, obj):
        places = self.projectedNets.get_otype_projection(obj.otype).places
        obj_tokens_in_places = list(map(lambda p:
                                        [t for t in marking.tokensByPlaces[p] if t.oid == obj.oid], places
                                        ))
        projected_marking = [len(ts) for ts in obj_tokens_in_places]
        return np.array(projected_marking)

    def __get_enabled_transitions_indices(self, m, Tin):
        TinT = Tin.transpose()
        n = len(TinT)
        enabled_transition_indices = []
        for j in range(n):
            tin = TinT[j]
            checksum = tin + m
            is_enabled = all(i >= 0 for i in checksum)
            if is_enabled:
                enabled_transition_indices.append(j)
        return enabled_transition_indices
