import logging
import os
import pickle
from datetime import datetime

import pandas as pd
import pm4py

from input_ocel_processing.process_config import ProcessConfig
from object_model_generation.object_instance import ObjectInstance
from object_model_generation.object_model import ObjectModel
from object_model_generation.training_model_preprocessor import TrainingModelPreprocessor
from ocpn_discovery.net_utils import Place, NetProjections
from .ocel_maker import OcelMaker
from .sim_utils import Marking, Token, Predictors
from .simulation_net import SimulationNet
from object_model_generation.object_instance import SimulationObjectInstance


class SimulationInitializer:

    def __init__(self, session_path, use_original_marking, object_model_name):
        logging.basicConfig(filename=os.path.join(session_path, "ocps_session.log"),
                            encoding='utf-8', level=logging.DEBUG)
        self.sessionPath = session_path
        self.objectModelName = object_model_name
        ocel_path = os.path.join(session_path, "postprocessed_input.jsonocel")
        self.ocel = pm4py.read_ocel(ocel_path)
        self.useOriginalMarking = use_original_marking
        self.processConfig: ProcessConfig = ProcessConfig.load(session_path)
        self.otypes = self.processConfig.otypes

    def load_net_and_objects(self):
        self.__load_net()
        self.__load_object_model(self.objectModelName)

    def initialize(self):
        self.__make_initial_marking()
        self.__make_simulation_net()
        self.__make_initial_features()
        # comment out after first initialization for speed up #
        self.__load_training_object_model()
        self.__load_training_data()
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.__make_predictors()
        self.__initialize_ocel()

    def save(self):
        features_path = os.path.join(self.sessionPath, "object_features_" + self.objectModelName + ".pkl")
        with open(features_path, "wb") as write_file:
            pickle.dump(self.objectFeatures, write_file)
        feature_names_path = os.path.join(self.sessionPath, "object_feature_names_" + self.objectModelName + ".pkl")
        with open(feature_names_path, "wb") as write_file:
            pickle.dump(self.objectFeatureNames, write_file)
        self.simulationNet.save()
        self.predictors.save()
        self.ocelMaker.save()

    def __make_initial_features(self):
        obj: ObjectInstance
        self.__make_object_feature_names()
        self.objectFeatures = {
            otype: dict()
            for otype in self.otypes
        }
        for obj in self.objectModel.objectsById.values():
            self.__initialize_object_features(obj)

    def __make_object_feature_names(self):
        self.objectFeatureNames = ["act:" + act for act in self.processConfig.acts] + \
                                  ["otype:" + any_otype for any_otype in self.otypes]

    def __initialize_object_features(self, obj: ObjectInstance):
        features = {}
        for act in self.processConfig.acts:
            features["act:" + act] = 0
        for any_otype in self.otypes:
            features["otype:" + any_otype] = len(obj.total_local_model[any_otype])
        # TOOO. how to incorporate non-categorical features?
        # for key, value in obj.attributes.items():
        #    features["attr:" + key] = value
        self.objectFeatures[obj.otype][obj.oid] = features

    def __load_net(self):
        self.netProjections = NetProjections.load(self.sessionPath)

    def __load_object_model(self, name=""):
        self.objectModel = ObjectModel.load(self.sessionPath, self.processConfig.useOriginalMarking, name)

    def __make_initial_marking(self):
        self.tokens = []
        p: Place
        all_places = []
        simulation_objects = dict()
        for otype in self.otypes:
            places = self.netProjections.get_otype_projection(otype).places
            all_places += places
            initial_places = list(filter(lambda p: p.otype == otype and p.isInitial, places))
            if len(initial_places) != 1:
                raise ValueError("The number of initial places for '" + otype + "' is not exactly 1.")
            p = initial_places[0]
            # TODO: fix error when using original marking
            for obj in self.objectModel.objectsByType[otype].keys():
                token = Token(obj.oid, otype, obj.time, p)
                simulation_object = SimulationObjectInstance(obj, [token])
                simulation_objects[obj.oid] = simulation_object
                self.tokens.append(token)
        self.marking = Marking(all_places, self.otypes, self.tokens)
        for otype in self.otypes:
            for obj in self.objectModel.objectsByType[otype].keys():
                oid = obj.oid
                sim_obj = simulation_objects[oid]
                for any_otype in self.otypes:
                    for any_obj in obj.direct_object_model[any_otype]:
                        any_sim_obj = simulation_objects[any_obj.oid]
                        any_otype = any_sim_obj.otype
                        if any_otype not in sim_obj.directObjectModel:
                            sim_obj.directObjectModel[any_otype] = []
                        sim_obj.directObjectModel[any_otype].append(any_sim_obj)
                        if otype not in any_sim_obj.reverseObjectModel:
                            any_sim_obj.reverseObjectModel[otype] = []
                        any_sim_obj.reverseObjectModel[otype] = list(
                            set(any_sim_obj.reverseObjectModel[otype] + [sim_obj]))
        self.simulationObjects = simulation_objects

    def __make_simulation_net(self):
        self.simulationNet = SimulationNet(self.sessionPath, self.netProjections, self.marking, self.simulationObjects,
                                           self.objectModelName)

    def __load_training_object_model(self):
        self.trainingModelPreprocessor = TrainingModelPreprocessor.load(self.sessionPath)

    def __load_training_data(self):
        otypes = self.otypes
        flattened_logs = {
            otype: pm4py.ocel_flattening(self.ocel, otype)
            .sort_values(['case:concept:name', 'time:timestamp'], ascending=[True, True])
            for otype in otypes
        }
        self.flattened_logs = flattened_logs
        logging.info("Creating Training Frames...")
        self.training_data = {}
        log_items = list(flattened_logs.items())
        log_items.sort(key=lambda item: len(item[1]))
        for otype, log in log_items:
            print(f"{otype}...")
            self.__compute_trace_histories(otype, log)
            self.__compute_delays(otype)
            training_path = os.path.join(self.sessionPath, otype + "_training_data.csv")
            self.training_data[otype].to_csv(training_path)
        logging.info("Training Frames Created.")

    def __compute_trace_histories(self, otype, log_frame):
        activities = self.processConfig.acts
        for act in activities:
            log_frame["act:" + act] = 0
        for any_otype in self.otypes:
            log_frame["otype:" + any_otype] = log_frame["case:concept:name"].apply(
                lambda obj: len(self.trainingModelPreprocessor.totalObjectModel[otype][obj][any_otype]))
        iterator = log_frame.iterrows()
        index, line = next(iterator, None)
        lastline, lastindex = line, index
        nextline = next(iterator, None)
        max_eid = max([round(float(eid)) for eid in log_frame["ocel:eid"].unique()])
        max_index = max(log_frame.index)
        count = 1
        end_events = []
        all_attributes = list(log_frame.columns)
        inherited_attributes = [col for col in all_attributes if
                                col not in ["ocel:eid", "concept:name", "time:timestamp"]]
        while nextline is not None:
            index, line = nextline
            if line["case:concept:name"] == lastline["case:concept:name"]:
                # Update Trace history
                act = lastline["concept:name"]
                for any_act in activities:
                    log_frame.at[index, "act:" + any_act] = log_frame.at[lastindex, "act:" + any_act]
                log_frame.at[index, "act:" + act] = log_frame.at[index, "act:" + act] + 1
            else:
                # Add artificial End Event
                eid = max_eid + count
                new_index = max_index + count
                count = count + 1
                end_event = self.__create_end_event(otype, new_index, eid, log_frame.loc[lastindex],
                                                    inherited_attributes)
                end_events.append(end_event)
            lastline, lastindex = line, index
            nextline = next(iterator, None)
        eid = max_eid + count
        new_index = max_index + count
        end_event = self.__create_end_event(otype, new_index, eid, log_frame.loc[lastindex], inherited_attributes)
        end_events.append(end_event)
        log_frame = log_frame[all_attributes]
        for row in end_events:
            log_frame.loc[len(log_frame)] = row
        log_frame = log_frame.sort_values(['case:concept:name', 'time:timestamp'], ascending=[True, True])
        self.training_data[otype] = log_frame

    def __create_end_event(self, otype, index, eid, last_event, inherited_attributes):
        row = {"ocel:eid": eid, "concept:name": "END_" + otype}
        for attr in inherited_attributes:
            row[attr] = last_event[attr]
        # TODO
        # +1 : just make the end event right after the last event.
        #row["time:timestamp"] = datetime.utcfromtimestamp(last_event["time:timestamp"].timestamp() + 1)
        lasttimestamp = last_event["time:timestamp"]
        lasttime = lasttimestamp.timestamp()
        timezone = lasttimestamp.tzname()
        row["time:timestamp"] = pd.Timestamp(lasttime + 1, unit="s", tz=timezone)
        last_act = last_event["concept:name"]
        row["act:" + last_act] = row["act:" + last_act] + 1
        return pd.Series(data=row, name=index)

    def __compute_delays(self, otype):
        training_frame = self.training_data[otype]
        training_frame["delay_to"] = 0
        training_frame["delay_from"] = 0
        training_frame["last_act"] = "START_" + otype
        training_frame["next_act"] = "END_" + otype
        training_frame["int:timestamp"] = training_frame \
            .apply(lambda row: int(row["time:timestamp"].timestamp()), axis=1)
        iterator = training_frame.iterrows()
        index, line = next(iterator, None)
        lastline, lastindex = line, index
        nextline = next(iterator, None)
        while nextline is not None:
            index, line = nextline
            if line["case:concept:name"] == lastline["case:concept:name"]:
                # Update DELAY
                last_act = lastline["concept:name"]
                act = line["concept:name"]
                delay = line["int:timestamp"] - lastline["int:timestamp"]
                training_frame.at[lastindex, 'delay_from'] = delay
                training_frame.at[index, 'delay_to'] = delay
                training_frame.at[lastindex, 'next_act'] = act
                training_frame.at[index, 'last_act'] = last_act
            lastline, lastindex = line, index
            nextline = next(iterator, None)
        self.training_data[otype] = training_frame

    def __make_predictors(self):
        predictors = Predictors(self.otypes, self.objectFeatureNames, self.sessionPath, self.objectModelName)
        predictors.initialize_activity_prediction_function()
        logging.info("Initializing Delay Prediction Function...")
        predictors.initialize_delay_prediction_function()
        self.predictors = predictors

    def __initialize_ocel(self):
        objects = {}
        df = self.ocel.get_extended_table()
        timestamp_offset = min(df["ocel:timestamp"]).timestamp()
        for oid, obj in self.objectModel.objectsById.items():
            objects[oid] = {"ocel:type": obj.otype, "ocel:ovmap": {}}
        self.ocelMaker = OcelMaker(self.sessionPath, self.objectModelName, self.useOriginalMarking, objects, {},
                                   timestamp_offset)
