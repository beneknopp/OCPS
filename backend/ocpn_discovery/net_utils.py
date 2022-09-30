import json
import os
import pickle
from enum import Enum

import numpy as np


class OCPN_Node:
    id: str

    def __init__(self, id):
        self.id = id
        self.incomingArcs = []
        self.outgoingArcs = []


class Place(OCPN_Node):
    otype: str
    isInitial: bool
    isFinal: bool

    def __init__(self, id, otype, is_initial, is_final):
        self.otype = otype
        self.isInitial = is_initial
        self.isFinal = is_final
        OCPN_Node.__init__(self, id)

    def as_json(self):
        return {"id": self.id, "otype": self.otype, "isInitial": self.isInitial, "isFinal": self.isFinal}


class TransitionType(Enum):
    ACTIVITY = "ACTIVITY"
    SILENT = "SILENT"
    INITIAL = "INITIAL"
    FINAL = "FINAL"


class Transition(OCPN_Node):
    transitionType: TransitionType
    label: str

    def __init__(self, id, label, transitionType):
        self.transitionType = transitionType
        self.label = label if label is not None else ""
        OCPN_Node.__init__(self, id)

    def as_json(self):
        return {"id": self.id, "label": self.label, "transitionType": self.transitionType.value}


class Arc:
    id: str
    source: OCPN_Node
    target: OCPN_Node
    placeEnd: Place
    transEnd: Transition
    isVariableArc: bool

    def __init__(self, id, source: OCPN_Node, target: OCPN_Node, is_variable_arc):
        self.id = id
        self.source = source
        self.target = target
        if isinstance(source, Place) and isinstance(target, Transition):
            self.placeEnd = source
            self.transEnd = target
        elif isinstance(source, Transition) and isinstance(target, Place):
            self.placeEnd = target
            self.transEnd = source
        else:
            raise ValueError("Trying to create arc not between a place and a transition")
        source.outgoingArcs.append(self)
        target.incomingArcs.append(self)
        self.isVariableArc = is_variable_arc

    def as_json(self):
        return {"id": self.id, "source": self.source.id,
                "target": self.target.id, "isVariableArc": self.isVariableArc}

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


class NetProjection:
    places: list
    transitions: list
    Tin: np.array
    Tout: np.array

    def __init__(self, places, transitions, Tin, Tout):
        self.places = places
        self.transitions = transitions
        self.Tin = Tin
        self.Tout = Tout


class NetProjections:

    @classmethod
    def load(cls, session_path):
        projected_nets_path = os.path.join(session_path, "projected_nets.pkl")
        return pickle.load(open(projected_nets_path, "rb"))

    __net_projections: dict

    def __init__(self, places, otypes):
        self.__net_projections = {}
        place: Place
        transition: Transition
        arc: Arc
        for otype in otypes:
            typed_places = list(filter(lambda place: place.otype == otype, places.values()))
            typed_arcs = list(map(lambda place: place.incomingArcs, typed_places)) + \
                         list(map(lambda place: place.outgoingArcs, typed_places))
            typed_arcs = [arc for sublist in typed_arcs for arc in sublist]
            typed_transitions = list(
                set(map(lambda arc: arc.transEnd, typed_arcs)))
            n_t = len(typed_transitions)
            n_p = len(typed_places)
            T_in = np.array([[0] * n_t] * n_p)
            T_out = np.array([[0] * n_t] * n_p)
            for arc in typed_arcs:
                if arc.source in typed_places:
                    dir = "PtoT"
                    place = arc.source
                    transition = arc.target
                else:
                    dir = "TtoP"
                    place = arc.target
                    transition = arc.source
                i_p = typed_places.index(place)
                i_t = typed_transitions.index(transition)
                if dir == "PtoT":
                    T_in[i_p][i_t] = T_in[i_p][i_t] - 1
                elif dir == "TtoP":
                    T_out[i_p][i_t] = T_out[i_p][i_t] + 1
            self.__net_projections[otype] = NetProjection(typed_places, typed_transitions, T_in, T_out)

    def get_otype_projection(self, otype) -> NetProjection:
        return self.__net_projections[otype]

    def save(self, session_path):
        projected_nets_path = os.path.join(session_path, "projected_nets.pkl")
        with open(projected_nets_path, "wb") as write_file:
            pickle.dump(self, write_file)
