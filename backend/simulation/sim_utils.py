from datetime import datetime
import math
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ocel_processing.process_config import ProcessConfig
from object_model_generation.generator_parametrization import Modeler, ModelType, DataType
from object_model_generation.object_instance import ObjectInstance, SimulationObjectInstance
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

    def remove_tokens(self, token_removals, token_removals_by_place, token_removals_by_otype):
        self.tokensByPlaces = {
            place: [t for t in tokens if not (place in token_removals_by_place.keys() and t in token_removals_by_place[place])]
            for place, tokens in self.tokensByPlaces.items()
        }
        self.tokensByOtype = {
            otype: [t for t in tokens if not (otype in token_removals_by_otype.keys() and t in token_removals_by_otype[otype])]
            for otype, tokens in self.tokensByOtype.items()
        }
        self.tokens = [t for t in self.tokens if not t in token_removals]

    def __remove_one_token(self, some_list, token):
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
    def load(cls, session_path, object_model_name: str = ""):
        predictors_path = os.path.join(session_path, "predictors_" + object_model_name + ".pkl")
        return pickle.load(open(predictors_path, "rb"))

    objectModelName: str
    next_activity_predictors: dict
    mean_delays_act_to_act: dict
    mean_delays_act: dict
    mean_delays_independent: dict

    def __init__(self, otypes, object_feature_names, session_path, object_model_name: str = ""):
        self.otypes = otypes
        self.objectModelName = object_model_name
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
        return self.mean_delays_independent[otype][next_act]

    def __fit_delays(self, otype, group, keys, modelers, model_types):
        assert len(model_types) == len(modelers)
        modeler_name = []
        for key in keys:
            modeler_name += [list(group[key])[0]]
        modeler_name = tuple(modeler_name)
        data = list(group["delay_to"])
        for i in range(len(modelers)):
            model_type = model_types[i]
            modeler = Modeler(model_type, DataType.CONTINUOUS)
            modeler.fit_data(data[:])
            modelers[i][otype][modeler_name] = modeler

    def get_batch_size_prediction(self, act):
        return self.batch_size_predictors[act].draw()

    def make_service_densities(self):
        pass

    def __reg_fit_joint_delays(self, data, dic, act_from=None, act_to=None):
        NUMBER_OF_BINS = 50
        if act_to is None:
            raise AttributeError()
        if act_from is not None:
            d = dic[act_from]
        else:
            d = dic
        number_of_bins = min(data["sync_time"].nunique(), NUMBER_OF_BINS)
        _, bins = (pd.qcut(data["sync_time"], q=number_of_bins, retbins=True, duplicates='drop'))
        data_ = data[["sync_time", "delay_to"]].sort_values(by="sync_time")
        bin_to_modeler = {}
        j = 1
        bins[0] = -1
        bins[-1] = float("inf")
        for i in range(len(bins)-1):
            left = bins[i]
            right = bins[i+1]
            in_bin = (data_["sync_time"] > left)
            in_bin = in_bin & (data_["sync_time"] <= right)
            bin_content = data_[in_bin]
            if not len(bin_content):
                continue
            modeler = Modeler(ModelType.CUSTOM, DataType.CONTINUOUS)
            modeler.fit_data(bin_content["delay_to"])
            bin_to_modeler[j] = modeler
            j = j + 1
        bin_to_modeler[0] = bin_to_modeler[1]
        bin_to_modeler[j] = bin_to_modeler[j-1]
        d[act_to] = {}
        d[act_to]["bins"] = bins
        d[act_to]["bin_to_modeler"] = bin_to_modeler

    def initialize_joint_delay_prediction_function(self):
        training_frames = []
        for otype in self.otypes:
            training_data = self.trainingData[otype]
            training_frames.append(training_data[["ocel:eid", "case:concept:name", "concept:name", "last_act", "delay_to", "int:timestamp", "time:timestamp"]])
        joint_training_frame = pd.concat(training_frames)
        joint_delays_independent = {}
        joint_delays_a2a = {}
        acts_from = sorted(list(joint_training_frame["last_act"].unique()))
        acts_to   = sorted(list(joint_training_frame["concept:name"].unique()))
        for act_from in acts_from:
            joint_delays_a2a[act_from] = {}
        for act_to in acts_to:
            joint_delays_independent[act_to] = {}
            act_joint_tf = joint_training_frame[joint_training_frame["concept:name"] == act_to]
            act_joint_tf["last_act:int:timestamp"] = act_joint_tf["int:timestamp"] - act_joint_tf["delay_to"]
            max_times = act_joint_tf.groupby(["ocel:eid"])["last_act:int:timestamp"].max()
            min_times = act_joint_tf.groupby(["ocel:eid"])["last_act:int:timestamp"].min()
            sync_times = (max_times - min_times).reset_index(name="sync_time")
            # for each event, the last preceding event for any involved object
            last_last_events = act_joint_tf.sort_values(by=["ocel:eid", "last_act:int:timestamp"]).groupby(["ocel:eid"]).last().reset_index()
            last_last_events = last_last_events.merge(sync_times, on="ocel:eid")
            self.__reg_fit_joint_delays(
                last_last_events, joint_delays_independent, act_from=None, act_to=act_to
            )
            last_last_events.groupby("last_act").apply(
                lambda grp: self.__reg_fit_joint_delays(
                    grp, joint_delays_a2a, act_from=grp.name, act_to=act_to
                )
            )
        self.joint_delays_independent = joint_delays_independent
        self.joint_delays_a2a = joint_delays_a2a

    def initialize_delay_prediction_function(self):
        mean_delays_act_to_act = {}
        mean_delays_act = {}
        mean_delays_independent = {}
        mean_delays_act_to_act_independent = {}
        custom_delays_independent = {}
        custom_delays_act_to_act_independent = {}
        exp_delays_independent = {}
        exp_delays_act_to_act_independent = {}
        for otype in self.otypes:
            custom_delays_independent[otype] = {}
            exp_delays_independent[otype] = {}
            custom_delays_act_to_act_independent[otype] = {}
            exp_delays_act_to_act_independent[otype] = {}
            training_data = self.trainingData[otype]
            delays_a_independent = training_data[
                ["concept:name"] + ["delay_to"]
            ]
            delays_a2a_independent = training_data[
                ["concept:name"] + ["delay_to"] + ["last_act"]
            ]
            delays_by_features_a = training_data[
                self.object_feature_names + ["concept:name"] + ["delay_to"]
            ]
            delays_by_features_a2a = training_data[
                self.object_feature_names + ["concept:name"] + ["delay_to"] + ["last_act"]
            ]
            delays_a_independent.groupby("concept:name").apply(
                lambda grp: self.__fit_delays(
                    otype=otype,
                    group=grp,
                    keys=["concept:name"],
                    modelers=[exp_delays_independent, custom_delays_independent],
                    model_types=[ModelType.EXPONENTIAL, ModelType.CUSTOM]
                )
            )
            delays_a2a_independent.groupby(["last_act", "concept:name"]).apply(
                lambda grp: self.__fit_delays(
                    otype=otype,
                    group=grp,
                    keys=["last_act", "concept:name"],
                    modelers=[exp_delays_act_to_act_independent, custom_delays_act_to_act_independent],
                    model_types=[ModelType.EXPONENTIAL, ModelType.CUSTOM]
                )
            )
            stats_a2a = delays_by_features_a2a.groupby(self.object_feature_names + ["last_act", "concept:name"]).mean()
            stats_a2a_independent = delays_a2a_independent.groupby(["last_act", "concept:name"]).mean()
            stats_a = delays_by_features_a.groupby(self.object_feature_names + ["concept:name"]).mean()
            stats_indie = delays_a_independent.groupby(["concept:name"]).mean()
            stats_dict_a2a = dict(stats_a2a.to_dict()["delay_to"])
            stats_dict_a2a_independent = dict(stats_a2a_independent["delay_to"])
            stats_dict_a = dict(stats_a.to_dict()["delay_to"])
            stats_dict_indie = dict(stats_indie.to_dict()["delay_to"])
            stats_dict_a2a = {
                key: int(round(value))
                for key, value in stats_dict_a2a.items()
            }
            stats_dict_a2a_independent = {
                key: int(round(value))
                for key, value in stats_dict_a2a_independent.items()
            }
            stats_dict_a = {
                key: round(value)
                for key, value in stats_dict_a.items()
            }
            stats_dict_indie = {
                key: round(value)
                for key, value in stats_dict_indie.items()
            }
            mean_delays_act_to_act[otype] = stats_dict_a2a
            mean_delays_act[otype] = stats_dict_a
            mean_delays_independent[otype] = stats_dict_indie
            mean_delays_act_to_act_independent[otype] = stats_dict_a2a_independent
        self.mean_delays_act_to_act = mean_delays_act_to_act
        self.mean_delays_act = mean_delays_act
        self.mean_delays_independent = mean_delays_independent
        self.mean_delays_act_to_act_independent = mean_delays_act_to_act_independent
        self.custom_delays_independent = custom_delays_independent
        self.custom_delays_act_to_act_independent = custom_delays_act_to_act_independent
        self.exp_delays_act_to_act_independent = exp_delays_act_to_act_independent
        self.exp_delays_independent = exp_delays_independent

    def initialize_batch_size_predictor(self):
        BATCH_WINDOW_SECONDS = 120
        self.batch_size_predictors = {}
        for act, leading_type in self.processConfig.activityLeadingTypes.items():
            training_frame = self.trainingData[leading_type]
            tfa = training_frame[training_frame["concept:name"] == act][["time:timestamp"]]
            tfa.sort_values(by="time:timestamp", inplace=True)
            tfa["time:timestamp"] = pd.to_datetime(tfa["time:timestamp"])
            tfa["time:timestamp:previous"] = tfa["time:timestamp"].shift(1)
            tfa["time:timestamp:next"] = tfa["time:timestamp"].shift(-1)
            tfa = tfa[1:-1]
            tfa["is_batch"] = (tfa["time:timestamp:next"] - tfa["time:timestamp:previous"])\
                                  .apply(lambda x: x.seconds) < BATCH_WINDOW_SECONDS
            self.group_id = 1
            def __make_batch_group(is_batch):
                if not is_batch:
                    self.group_id = self.group_id + 1
                return self.group_id
            tfa["batch_group"] = tfa["is_batch"].apply(__make_batch_group)
            batch_sizes = list(tfa.groupby("batch_group").size())
            modeler = Modeler(ModelType.CUSTOM, DataType.CONTINUOUS)
            modeler.fit_data(batch_sizes)
            self.batch_size_predictors[act] = modeler

    def get_delay_a2a(self, otype, key):
        return self.custom_delays_act_to_act_independent[otype][key].draw()
        #return self.exp_delays_act_to_act_independent[otype][key].draw()

    def get_joint_delay(self, simulation_objects, next_act):
        times = list(map(lambda so: so.time, simulation_objects))
        min_time = min(times)
        max_time = max(times)
        sync_time = max_time - min_time
        latest_so: SimulationObjectInstance
        latest_so = list(filter(lambda so: so.time == max_time, simulation_objects))[0]
        last_act = latest_so.lastActivity
        if next_act == "Delivery: Post Goods Issue":
            print(1)
        try:
            delay_dict = self.joint_delays_a2a[last_act][next_act]
        except:
            delay_dict = self.joint_delays_independent[next_act]
        bins = delay_dict["bins"]
        bin_to_modeler = delay_dict["bin_to_modeler"]
        bin_ix = np.digitize(sync_time, bins, right=True)
        modeler: Modeler = bin_to_modeler[bin_ix]
        joint_delay = modeler.draw()
        execution_time = max_time + joint_delay
        return execution_time

    def get_delay_a(self, otype, next_act):
        return self.custom_delays_independent[otype][tuple([next_act])].draw()
        #return self.exp_delays_independent[otype][tuple([next_act])].draw()

    def save(self):
        predictors_path = os.path.join(self.session_path, "predictors_" + self.objectModelName + ".pkl")
        with open(predictors_path, "wb") as write_file:
            pickle.dump(self, write_file)


class SimulationStateExport:
    clock: int
    dateClock: str
    steps: int
    activeTokens: list
    bindings: list
    objectsInitialized: dict
    objectsTerminated: dict
    totalObjects: dict
    markingInfo: dict

    def __init__(self, old_clock, process_config: ProcessConfig, marking: Marking, bindings, steps, transitions):
        self.steps = steps
        self.marking = marking
        self.processConfig = process_config
        otypes = process_config.otypes
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
            self.__make_date_clock()
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
        self.__make_date_clock()
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
            "clock": self.dateClock,
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

    # TODO!!!
    def __make_date_clock(self):
        intClock= self.clock
        clockOffset = self.processConfig.clockOffset
        clockOffset = clockOffset / 1000000000
        timestamp = datetime.fromtimestamp(clockOffset + intClock)
        time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        self.dateClock = time_str


class NextActivityCandidate:

    def __init__(self, transition, leading_object: ObjectInstance, paths):
        self.transition = transition
        self.leadingObject = leading_object
        self.paths = paths
