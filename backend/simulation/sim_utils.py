import math
import os
import pickle

import numpy as np
import pandas as pd

from input_ocel_processing.process_config import ProcessConfig
from object_model_generation.object_instance import ObjectInstance
from ocpn_discovery.net_utils import Place, Transition, TransitionType


class Token:
    oid: str
    otype: str
    time: int
    place: Place

    def __init__(self, oid, otype, time, place):
        self.oid = oid
        self.otype = otype
        self.time = time
        self.place = place


class Marking:
    places: set
    otypes: set
    tokens: list
    tokensByPlaces: dict
    tokensByOtype: dict

    @classmethod
    def load(cls, session_path):
        marking_path = os.path.join(session_path, "marking.pkl")
        return pickle.load(open(marking_path, "rb"))

    def __init__(self, places, otypes, tokens):
        self.places = places
        self.otypes = otypes
        self.tokens = tokens
        self.tokensByPlaces = {
            place: [t for t in tokens if t.place == place]
            for place in places
        }
        self.tokensByOtype = {
            otype: [t for t in tokens if t.otype == otype]
            for otype in otypes
        }

    def save(self, session_path):
        marking_path = os.path.join(session_path, "marking.pkl")
        with open(marking_path, "wb") as write_file:
            pickle.dump(self, write_file)

    def add_token(self, token: Token):
        otype = token.otype
        place = token.place
        self.tokensByPlaces[place] = self.tokensByPlaces[place] + [token]
        self.tokensByOtype[otype] = self.tokensByOtype[otype] + [token]
        self.tokens = self.tokens + [token]

    def remove_token(self, place: Place, token: Token):
        otype = token.otype
        tokens_at_place = self.tokensByPlaces[place]
        tokens_at_place = self.__remove_one_token(tokens_at_place, token)
        self.tokensByPlaces[place] = tokens_at_place
        tokens_by_otype = self.tokensByOtype[otype]
        tokens_by_otype = self.__remove_one_token(tokens_by_otype, token)
        self.tokensByOtype[otype] = tokens_by_otype
        tokens = self.tokens
        tokens = self.__remove_one_token(tokens, token)
        self.tokens = tokens

    def __remove_one_token(self, some_list, token):
        token_in_list = list(filter(lambda t: t == token, some_list))
        if len(token_in_list) > 1:
            raise ValueError("Token found more than one time. Tokens must be unique by reference.")
        some_list = list(filter(lambda t: t != token, some_list))
        return some_list

    def get_object_projection(self, obj: ObjectInstance):
        token: Token
        tokens = self.tokensByOtype[obj.otype]
        object_projection = [list(filter(lambda token: token.id == obj.oid, tokens)) for tokens in tokens]
        return np.array([len(object_projection[place_index]) for place_index in range(len(object_projection))])

    def update_object_time(self, oid, new_tokens_time):
        tokens = [t for t in self.tokens if t.oid == oid]
        for token in tokens:
            token.time = new_tokens_time


class Predictors:

    @classmethod
    def load(cls, session_path):
        predictors_path = os.path.join(session_path, "predictors.pkl")
        return pickle.load(open(predictors_path, "rb"))

    next_activity_predictors: dict
    delay_predictors: dict

    def __init__(self, otypes, object_feature_names, session_path):
        self.otypes = otypes
        self.trainingData = dict()
        for otype in self.otypes:
            training_path = os.path.join(session_path, otype + "_training_data.csv")
            self.trainingData[otype] = pd.read_csv(training_path)
        self.object_feature_names = object_feature_names
        self.session_path = session_path
        self.processConfig = ProcessConfig.load(session_path)

    def initialize_activity_prediction_function(self):
        next_activity_predictors = {}
        for otype in self.otypes:
            training_data = self.trainingData[otype]
            probabilities_by_features = self.__make_next_activity_probabilities(otype, training_data)
            next_activity_predictors[otype] = probabilities_by_features
        self.next_activity_predictors = next_activity_predictors

    def __make_next_activity_probabilities(self, otype, training_data):
        activity_counts_by_features = training_data[self.object_feature_names + ["concept:name"]] \
            .groupby(self.object_feature_names)["concept:name"] \
            .value_counts()
        stats_dict = dict(activity_counts_by_features)
        act_freqs_by_features = {}
        for key in stats_dict:
            features = key[:-1]
            act = key[-1]
            freq = stats_dict[key]
            if not features in act_freqs_by_features:
                act_freqs_by_features[features] = []
            act_freqs_by_features[features].append((act, freq))
        next_activity_predictors = dict()
        for features, act_freqs in act_freqs_by_features.items():
            total = sum(list(map(lambda act_freq: act_freq[1], act_freqs)))
            probabilities = {
                act_freq[0]: float(act_freq[1]) / float(total)
                for act_freq in act_freqs
            }
            next_activity_predictors[features] = probabilities
        return next_activity_predictors

    def __value_counts(self, series):
        return series.value_counts()

    def __binomial_distribution(self, series):
        mean = series.mean()
        var = series.var()
        stdev = math.sqrt(var)
        return lambda x: np.random.normal(mean, stdev, 1)[0]

    def __binom_info(self, series):
        mean = series.mean()
        var = series.var()
        stdev = math.sqrt(var)
        return mean, stdev

    def get_mean_act_delay(self, otype, next_act):
        return self.mean_act_delays[otype][next_act]

    def initialize_delay_prediction_function(self):
        delay_predictors = {}
        self.mean_act_delays = {}
        for otype in self.otypes:
            training_data = self.trainingData[otype]
            delays_by_features = training_data[self.object_feature_names + ["concept:name"] + ["delay"]]\
                .groupby(self.object_feature_names + ["concept:name"])["delay"] \
                .value_counts()
            stats_dict = dict(delays_by_features)
            delay_freqs_by_features = {}
            for key in stats_dict:
                features = key[:-1]
                delay = key[-1]
                freq = stats_dict[key]
                if features not in delay_freqs_by_features:
                    delay_freqs_by_features[features] = []
                delay_freqs_by_features[features].append((delay, freq))
            ot_delay_predictors = dict()
            for features, delay_freqs in delay_freqs_by_features.items():
                total = sum(list(map(lambda delay_freq: delay_freq[1], delay_freqs)))
                probabilities = {
                    delay_freq[0]: float(delay_freq[1]) / float(total)
                    for delay_freq in delay_freqs
                }
                ot_delay_predictors[features] = probabilities
            delay_predictors[otype] = ot_delay_predictors
            mean_training_frame = training_data[["concept:name"] + ["delay"]]
            stats = mean_training_frame.groupby(["concept:name"]).mean()
            stats_dict = dict(stats.to_dict()["delay"])
            stats_dict = {
                key: round(value) if otype not in self.processConfig.nonEmittingTypes else 0
                for key, value in stats_dict.items()
            }
            self.mean_act_delays[otype] = stats_dict
        self.delay_predictors = delay_predictors

    def save(self):
        predictors_path = os.path.join(self.session_path, "predictors.pkl")
        with open(predictors_path, "wb") as write_file:
            pickle.dump(self, write_file)


class SimulationStateExport:
    clock: int
    steps: int
    activeTokens: list
    bindings: list
    objectsInitialized: dict
    objectsTerminated: dict
    totalObjects: dict
    markingInfo: dict

    def __init__(self, old_clock, otypes, marking: Marking, bindings, steps, transitions):
        self.steps = steps
        self.marking = marking
        self.transitions = transitions
        self.bindings = bindings
        self.objectsInitialized = {otype: 0 for otype in otypes}
        self.objectsTerminated = {otype: 0 for otype in otypes}
        self.totalObjects = {otype: 0 for otype in otypes}
        self.markingInfo = {
            place.id: len(tokens)
            for place, tokens in marking.tokensByPlaces.items()
        }
        for otype in otypes:
            typed_tokens = [t for t in marking.tokens if t.otype == otype]
            total_objects = len(set([t.oid for t in typed_tokens]))
            self.objectsInitialized[otype] = len(set([t.oid for t in typed_tokens if not t.place.isInitial]))
            self.objectsTerminated[otype] = len(set([t.oid for t in typed_tokens if t.place.isFinal]))
            self.totalObjects[otype] = total_objects
        self.clock = max(map(lambda t: t.time, marking.tokens))
        self.activeTokens = []
        if len(bindings) == 0:
            return
        (next_obj, next_transition_id) = bindings[0]
        next_active_tokens = self.__get_active_tokens(next_obj, next_transition_id)
        self.activeTokens += next_active_tokens
        for obj_id, transition_id in bindings[1:]:
            self.activeTokens += self.__get_active_tokens(obj_id, transition_id)
        next_transition: Transition = [t for t in self.transitions if t.id == next_transition_id][0]
        if next_transition.transitionType != TransitionType.ACTIVITY:
            # the next transition to fire determines the clock
            self.clock = max(map(lambda t: t.time, next_active_tokens))
        else:
            # the activity is the last transition to be executed in a binding sequence
            # so all active tokens are bound in the next firing
            self.clock = max(map(lambda t: t.time, self.activeTokens))
        if old_clock > self.clock:
            self.clock = old_clock
        self.activeTokens = list(set(self.activeTokens))
        self.activeTokens.sort(key=lambda t: t.time)

    def __get_active_tokens(self, obj_id, transition_id):
        transition = [t for t in self.transitions if t.id == transition_id][0]
        incoming_places = [arc.placeEnd for arc in transition.incomingArcs]
        active_tokens = []
        for place in incoming_places:
            active_tokens += [t for t in self.marking.tokensByPlaces[place] if t.oid == obj_id]
        return active_tokens

    def toJSON(self):
        return {
            "clock": self.clock,
            "steps": self.steps,
            "objectsInitialized": self.objectsInitialized,
            "objectsTerminated": self.objectsTerminated,
            "totalObjects": self.totalObjects,
            "activeTokens": [{
                "oid": token.oid,
                "place_id": token.place.id,
                "time": token.time
            } for token in self.activeTokens],
            "bindings": self.bindings,
            "markingInfo": self.markingInfo
        }


class NextActivityCandidate:

    def __init__(self, transition, leading_object: ObjectInstance, paths):
        self.transition = transition
        self.leadingObject = leading_object
        self.paths = paths
