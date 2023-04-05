import logging
import math
import os
import pickle
import random
from datetime import datetime

import numpy as np
import pm4py

from input_ocel_processing.process_config import ProcessConfig
from object_model_generation.object_instance import ObjectInstance
from object_model_generation.object_model import ObjectModel
from ocpn_discovery.net_utils import Transition, TransitionType
from simulation.ocel_maker import OcelMaker
from simulation.sim_utils import Token, Predictors, SimulationStateExport, NextActivityCandidate
from simulation.simulation_net import SimulationNet
from simulation.simulation_object_instance import SimulationObjectInstance, ScheduledActivity
from utils.cumulative_distribution import CumulativeDistribution
from eval.evaluators import ocel_to_ocel


class Simulator:

    @classmethod
    def load(cls, session_path):
        simulator_path = os.path.join(session_path, "simulator.pkl")
        return pickle.load(open(simulator_path, "rb"))

    processConfig: ProcessConfig
    simulationNet: SimulationNet
    objectModel: ObjectModel
    initializedObjects: set
    terminatedObjects: set
    totalNumberOfObjects: int
    features: dict
    verbose = True
    enabledBindings: list

    def __init__(self, session_path, use_original_marking):
        logging.basicConfig(filename=os.path.join(session_path, "ocps_session.log"),
                            encoding='utf-8', level=logging.DEBUG)
        self.sessionPath = session_path
        self.processConfig = ProcessConfig.load(session_path)
        self.objectModel = ObjectModel.load(session_path, use_original_marking)
        self.totalNumberOfObjects = len(self.objectModel.objectsById)
        self.initializedObjects = set()
        self.terminatedObjects = set()
        self.steps = 0
        self.clock = 0

    def initialize(self):
        self.__load_simulation_net()
        self.__load_ocel_maker()
        self.__load_features()
        self.__load_predictors()
        self.__initialize_predictions()

    def run_steps(self, steps):
        step_count = 0
        while step_count != steps:
            terminated = self.__execute_step()
            step_count = step_count + 1
            if step_count % 500 == 0:
                print(step_count)
            if step_count % 1000 == 0:
                self.__write_ocel()
            if terminated:
                break
        self.__write_ocel()

    def __initialize_predictions(self):
        simulation_objects = list(self.simulationNet.get_all_simulation_objects().values())
        simulation_objects.sort(key=lambda so: so.time)
        simulation_object: SimulationObjectInstance
        simulation_objects = [simulation_object for simulation_object in simulation_objects
                              if simulation_object.oid not in self.terminatedObjects]
        for simulation_object in simulation_objects:
            self.__predict_leading_activity(simulation_object)

    def __predict_leading_activity(self, simulation_object: SimulationObjectInstance):
        predicted_transition : Transition = self.__make_feature_based_leading_prediction(simulation_object)
        if predicted_transition is None:
            simulation_object.active = False
            simulation_object.nextActivity = None
            return
            # the prediction is feature-based and does not respect the marking
            # now, if the marking allows to realize the prediction, then schedule
            # is this correct? TODO
        execution_objects = [simulation_object]
        if predicted_transition.transitionType == TransitionType.FINAL:
            # only terminate if parent objects have terminated
            if not self.__ready_for_termination(simulation_object):
                simulation_object.active = False
                simulation_object.nextActivity = None
                return
        else:
            direct_om = simulation_object.directObjectModel
            for any_otype in self.processConfig.otypes:
                execution_objects += direct_om[any_otype] if any_otype in direct_om else []
        any_sim_obj: SimulationObjectInstance
        paths = dict()
        for any_sim_obj in execution_objects:
            obj_instance: ObjectInstance = any_sim_obj.objectInstance
            path_from_obj = self.simulationNet.compute_path(obj_instance, predicted_transition.label)
            if path_from_obj is None:
                simulation_object.active = False
                simulation_object.nextActivity = None
                return
            paths[obj_instance] = path_from_obj
        execution_time = max(map(lambda obj : obj.time, execution_objects))
        scheduled_activity = ScheduledActivity(predicted_transition, paths, execution_time)
        simulation_object.active = True
        simulation_object.nextActivity = scheduled_activity

    def __get_execution_probability(self, candidate_activity, bound_objects):
        obj: ObjectInstance
        p = 0
        #p = 1
        n = 0
        for obj in bound_objects:
            otype = obj.otype
            #if otype in self.processConfig.nonEmittingTypes :
             #   continue
            n = n + 1
            features_by_object = self.objectFeatures[otype][obj.oid]
            object_features = tuple(
                list(map(lambda feature: int(features_by_object[feature]), self.objectFeatureNames)))
            next_act_predictor = self.predictors.next_activity_predictors[otype]
            if object_features in next_act_predictor:
                if candidate_activity in next_act_predictor[object_features]:
                    probability = next_act_predictor[object_features][candidate_activity]
                else:
                    probability = 0
            else:
                probability = self.__get_nearest_prediction(next_act_predictor, object_features, candidate_activity)
            p += probability
        if n == 0:
            return 0
        return float(p)/float(n)

    def __ALT_get_execution_probability(self, candidate_activity, bound_objects):
        obj: ObjectInstance
        #p = 0
        p = 1
        n = 0
        for obj in bound_objects:
            otype = obj.otype
            if otype in self.processConfig.nonEmittingTypes:
                continue
            n = n + 1
            features_by_object = self.objectFeatures[otype][obj.oid]
            object_features = tuple(
                list(map(lambda feature: int(features_by_object[feature]), self.objectFeatureNames)))
            next_act_predictor = self.predictors.next_activity_predictors[otype]
            if object_features in next_act_predictor and candidate_activity in next_act_predictor[object_features]:
                probability = next_act_predictor[object_features][candidate_activity]
            else:
                #return 0
                probability = self.__get_nearest_prediction(next_act_predictor, object_features, candidate_activity)
            p = min(probability, p)
        if n == 0:
            return 0
        return float(p)#/float(n)

    def __get_nearest_prediction(self, next_act_predictor, object_features, candidate_activity):
        domain = list(next_act_predictor.keys())
        # TODO: make more efficient
        domain_with_distances = list(map(lambda x: (x, np.linalg.norm(np.array(x)-np.array(object_features))), domain))
        min_dist = min(domain_with_distances, key=lambda x: x[1])[1]
        nearest_neighbors = [neighbor[0] for neighbor in filter(lambda x: x[1] == min_dist, domain_with_distances)]
        total_probability = 0
        for neighbor in nearest_neighbors:
            if candidate_activity not in next_act_predictor[neighbor]:
                continue
            total_probability += next_act_predictor[neighbor][candidate_activity]
        return total_probability / float(len(nearest_neighbors))

    def __make_feature_based_leading_prediction(self, simulation_object: SimulationObjectInstance):
        oid = simulation_object.oid
        otype = simulation_object.otype
        obj = simulation_object.objectInstance
        features_by_object = self.objectFeatures[otype][oid]
        object_features = tuple(list(map(lambda feature: int(features_by_object[feature]), self.objectFeatureNames)))
        next_act_predictor = self.predictors.next_activity_predictors[otype]
        if object_features in next_act_predictor:
            predictions = next_act_predictor[object_features]
            execution_probabilities = dict()
            if predictions:
                max_prob = 0
                activity_leading_types = self.processConfig.activityLeadingTypes
                for candidate_activity, p in predictions.items():
                    if candidate_activity[:4] == "END_":
                        execution_probabilities[candidate_activity] = p
                        max_prob = max(p, max_prob)
                    else:
                        leading_type = activity_leading_types[candidate_activity]
                        try:
                            leading_obj = obj if leading_type == otype else \
                                list(obj.reverse_object_model[leading_type])[0]
                        except:
                            continue
                        execution_model = [leading_obj]
                        #execution_model += [any_obj for sl in leading_obj.direct_object_model.values() for any_obj in sl]
                        execution_model = execution_model + [any_obj for sl in leading_obj.direct_object_model.values() for any_obj in sl]
                        execution_probability = self.__get_execution_probability(candidate_activity, execution_model)
                        execution_probabilities[candidate_activity] = execution_probability
                        max_prob = max(max_prob, execution_probability)
                total_prob = sum(list(execution_probabilities.values()))
                inact = 1 - total_prob
                if inact > 0:
                    execution_probabilities["INACT"] = inact
                if total_prob > 0:
                    cum_dist = CumulativeDistribution(execution_probabilities)
                    prediction: str = cum_dist.sample()
                    if prediction != "INACT":
                        transition = self.__get_transition(prediction)
                        if transition.transitionType == TransitionType.FINAL or activity_leading_types[prediction] == otype:
                            return transition

    def schedule_next_activity(self):
        active_simulation_objects = self.simulationNet.get_all_active_simulation_objects()
        if len(active_simulation_objects) == 0:
            if len(self.terminatedObjects) == self.totalNumberOfObjects:
                return False
            self.__initialize_predictions()
            active_simulation_objects = self.simulationNet.get_all_active_simulation_objects()
            if len(active_simulation_objects) == 0:
                return False
            print("klaubing remaining simulation objects: " + str(len(active_simulation_objects)))
        active_simulation_objects.sort(key=lambda so: so.nextActivity.time)
        simulation_object: SimulationObjectInstance = active_simulation_objects[0]
        next_activity = simulation_object.nextActivity
        predicted_transition = next_activity.transition
        if predicted_transition.transitionType == TransitionType.FINAL:
            self.simulationNet.terminatingObjects.add(simulation_object.oid)
        self.__update_enabled_transitions(next_activity.paths)
        return True

    def __update_predictions(self, objects):
        obj_instance: ObjectInstance
        rescheduled_objects = set()
        for obj_instance in objects:
            sim_obj = self.simulationNet.simulationObjects[obj_instance.oid]
            rescheduled_objects.add(sim_obj)
            total_om = []
            for otype in self.processConfig.otypes:
                total_om += list(obj_instance.total_local_model[otype])
            for any_obj in total_om:
                if any_obj in self.terminatedObjects:
                    continue
                sim_obj = self.simulationNet.simulationObjects[any_obj.oid]
                rescheduled_objects.add(sim_obj)
        for any_obj in rescheduled_objects:
            self.__predict_leading_activity(any_obj)

    def __get_weighted_alt_probability_from_neighborhood(self, features, next_act_predictor, next_act):
        projected_features = [e for e in features if not type(e) == str]
        neighbor_features = next_act_predictor.keys()
        projected_neighbor_features = list(
            map(lambda w: tuple([e for e in w if not type(e) == str]), neighbor_features))
        projected_neighbor_features.sort(key=lambda w: math.dist(projected_features, w))
        neighbors = projected_neighbor_features[:10]
        p = 0
        w = 0
        for neighbor in neighbors:
            dist = math.dist(neighbor, features)
            weight = 1.0 / (dist + 0.05)
            prob = 0 if next_act not in next_act_predictor[neighbor] else next_act_predictor[neighbor][next_act]
            p = p + prob * weight
            w = w + weight
        p = p / w
        return p

    def export_current_state(self):
        transitions = self.simulationNet.getAllTransitions()
        state_export = SimulationStateExport(
            self.clock,
            self.processConfig.otypes,
            self.simulationNet.marking,
            self.enabledBindings,
            self.steps,
            transitions
        )
        self.clock = state_export.clock
        return state_export.toJSON()

    def save(self):
        simulator_path = os.path.join(self.sessionPath, "simulator.pkl")
        with open(simulator_path, "wb") as write_file:
            pickle.dump(self, write_file)

    def __load_simulation_net(self):
        self.simulationNet = SimulationNet.load(self.sessionPath)

    def __load_ocel_maker(self):
        self.ocelMaker = OcelMaker.load(self.sessionPath)

    def __load_features(self):
        self.objectFeatures = pickle.load(open(os.path.join(self.sessionPath, "object_features.pkl"), "rb"))
        self.objectFeatureNames = pickle.load(open(os.path.join(self.sessionPath, "object_feature_names.pkl"), "rb"))

    def __load_predictors(self):
        self.predictors = Predictors.load(self.sessionPath)

    def __execute_step(self):
        (obj_id, transition_id) = self.enabledBindings[0]
        self.enabledBindings = self.enabledBindings[1:]
        object_ids = [obj_id]
        if transition_id in self.processConfig.acts:
            objects_with_t = list(filter(lambda ot: ot[1] == transition_id, self.enabledBindings))
            object_ids += [ot[0] for ot in objects_with_t]
            self.enabledBindings = []
        objects = [self.objectModel.objectsById[oid] for oid in object_ids]
        timestamp = self.__fire(transition_id, objects)
        has_finished = self.__init_finit_procedure(transition_id, object_ids)
        if has_finished:
            logging.info("Simulation finished. All objects have terminated.")
            return True
        if not (transition_id in self.processConfig.acts or transition_id[:4] == "END_"):
            return False
        self.__update_features(transition_id, object_ids)
        self.__update_predictions(objects)
        obj_ids_str = ", ".join([str(i) for i in object_ids])
        date_str = datetime.utcfromtimestamp(timestamp).strftime("%d/%m/%Y, %H:%M:%S")
        logging.info(f"Executed {transition_id} with {obj_ids_str} at {date_str}")
        do_next = self.schedule_next_activity()
        if not transition_id[:4] == "END_":
            self.__update_ocel(transition_id, object_ids, timestamp)
        if not do_next:
            logging.info("Activity could not be predicted. Terminating simulation...")
            return True
        # put next transition to be executed in front
        self.__sort_bindings()

    def __fire(self, transition_id, objects):
        if transition_id in self.processConfig.acts:
            delays = self.__get_delay_predictions(transition_id, objects)
        else:
            delays = {obj.oid: 0 for obj in objects}
        timestamp = self.simulationNet.fire(transition_id, objects, delays)
        return timestamp

    def __init_finit_procedure(self, transition_name, object_ids):
        if transition_name[:6] == "START_":
            self.initializedObjects.update(object_ids)
        if transition_name[:4] == "END_":
            self.terminatedObjects.update(object_ids)
            if len(self.terminatedObjects) == self.totalNumberOfObjects:
                return True
        return False

    def __get_delay_predictions(self, next_act, objects):
        delays = {}
        delay_ws = []
        for obj in objects:
            otype = obj.otype
            features_by_object =  self.objectFeatures[otype][obj.oid]
            object_features_and_act = tuple(
                list(map(lambda feature: int(features_by_object[feature]), self.objectFeatureNames)) + [next_act])
            predictors = self.predictors.delay_predictors[otype]
            if object_features_and_act in predictors:
                delays_for_obj = predictors[object_features_and_act]
                cum_dist = CumulativeDistribution(delays_for_obj)
                delay = cum_dist.sample()
            else:
                delay = self.predictors.get_mean_act_delay(otype, next_act)
            delays[obj.oid] = delay
            delay_ws.append(str(obj) + ":" + str(delay))
        return delays

    def __ready_for_termination(self, sim_obj: SimulationObjectInstance):
        obj = sim_obj.objectInstance
        reverse_object_model = [any_obj.oid for subset in obj.reverse_object_model.values() for any_obj in subset]
        return all(x in self.simulationNet.terminatingObjects for x in reverse_object_model)

    def __get_all_paths(self, obj, next_activity):
        # return all paths for all other objects beside obj needed to perform next_activity
        otype = obj.otype
        lead_type = self.processConfig.activityLeadingTypes[next_activity]
        if otype == lead_type:
            lead_obj = obj
        else:
            lead_objs = list(obj.reverse_object_model[lead_type])
            lead_objs.sort(key=lambda obj: obj.time)
            lead_obj = lead_objs[0]
        direct_obj_model = [objs for any_otype, objs in lead_obj.direct_object_model.items()]
        direct_obj_model = [lead_obj] + [model_obj for objs in direct_obj_model for model_obj in objs]
        paths_by_obj = {}
        for model_obj in direct_obj_model:
            path = self.simulationNet.compute_path(model_obj, next_activity)
            if path is None:
                return None, None
            paths_by_obj[model_obj] = path
        return lead_obj, paths_by_obj

    def __update_enabled_transitions(self, paths):
        self.enabledBindings = []
        objects = list(paths.keys())
        for obj in objects:
            enabled_transitions = paths[obj].firing_sequence
            for transition in enabled_transitions:
                self.enabledBindings.append((obj.oid, transition.id))
        self.__sort_bindings()

    def __sort_bindings(self):
        # find the binding that will be realized next
        sorted_bindings = []
        next_activity_bindings = []
        earliest_firing_time = -1
        transition: Transition
        for obj_id, transition_id in self.enabledBindings:
            transition = self.__get_transition(transition_id)
            bound_tokens = []
            ready = True
            for place in map(lambda arc: arc.placeEnd, transition.incomingArcs):
                # assumption: unweighted net, so need one token from each 
                # incoming place.
                bound = [token for token in self.simulationNet.marking.tokensByPlaces[place]
                         if token.oid == obj_id]
                if len(bound) == 0:
                    ready = False
                    break
                bound_tokens += bound
            if ready:
                firing_time = max(map(lambda token: token.time, bound_tokens))
                if len(sorted_bindings) == 0 or firing_time < earliest_firing_time \
                        and transition.transitionType != TransitionType.ACTIVITY:
                    earliest_firing_time = firing_time
                    sorted_bindings = [(obj_id, transition.id)] + sorted_bindings
                    continue
            if transition.transitionType == TransitionType.ACTIVITY:
                next_activity_bindings.append((obj_id, transition.id))
                continue
            sorted_bindings += [(obj_id, transition_id)]
        self.enabledBindings = sorted_bindings + next_activity_bindings

    # TODO: refactor
    def __get_transition(self, transition_id):
        return [t for t in self.simulationNet.getAllTransitions() if t.id == transition_id][0]

    def __update_ocel(self, activity, objects, timestamp):
        omap = [str(obj) for obj in objects]
        timestamp = str(timestamp)
        vmap = {}
        self.ocelMaker.add_event(activity, timestamp, omap, vmap)

    def __update_features(self, transition_name, objects):
        for obj in objects:
            if transition_name[:3] == "END":
                return
            otype = self.objectModel.objectsById[obj].otype
            feature = "act:" + transition_name
            self.objectFeatures[otype][obj][feature] = self.objectFeatures[otype][obj][feature] + 1

    def __write_ocel(self):
        self.ocelMaker.write_ocel()

    def __evaluate(self):
        # TODO: get ocel in the right format from class variable without loading it from fs
        original_log_file_path = os.path.join(self.sessionPath, "input.jsonocel")
        original_ocel = ocel_import_factory.apply(file_path=original_log_file_path)
        simulated_log_file_path = os.path.join(self.sessionPath, "simulated_ocel.jsonocel")
        simulated_ocel = ocel_import_factory.apply(file_path=simulated_log_file_path)
        allowed_types = [col for col in simulated_ocel.object_types if not col.startswith("LEAD_")]
        filtered_simulated_ocel = pm4py.filter_ocel_object_types(simulated_ocel, allowed_types)
        precision, fitness = ocel_to_ocel(original_ocel=original_ocel, simulated_ocel=filtered_simulated_ocel)
        eval_path = os.path.join(self.sessionPath, "ocim_eval.txt")
        with open(eval_path, "w") as wf:
            wf.write("precision=" + str(precision) + ";fitness=" + str(fitness))
