import logging
import math
import os
import pickle
import random
from datetime import datetime

from input_ocel_processing.process_config import ProcessConfig
from object_model_generation.object_instance import ObjectInstance
from object_model_generation.object_model import ObjectModel
from ocpn_discovery.net_utils import Transition, TransitionType
from simulation.ocel_maker import OcelMaker
from simulation.sim_utils import Token, Predictors, SimulationStateExport, NextActivityCandidate
from simulation.simulation_net import SimulationNet
from simulation.simulation_object_instance import SimulationObjectInstance
from utils.cumulative_distribution import CumulativeDistribution


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

    def __init__(self, session_path):
        logging.basicConfig(filename=os.path.join(session_path, "ocps_session.log"),
                            encoding='utf-8', level=logging.DEBUG)
        self.sessionPath = session_path
        self.processConfig = ProcessConfig.load(session_path)
        self.objectModel = ObjectModel.load(session_path)
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
        self.scheduledLeadSteps = {}
        self.failedLeadSteps = {}

    def run_steps(self, steps):
        step_count = 0
        while step_count != steps:
            terminated = self.__execute_step()
            step_count = step_count + 1
            self.steps = self.steps + 1
            if terminated:
                self.__write_ocel()
                break

    def __initialize_predictions(self):
        simulation_objects = self.simulationNet.get_all_active_simulation_objects():
        simulation_object: SimulationObjectInstance
        for simulation_object in simulation_objects:
            self.__predict_leading_activity(simulation_object)

    def __predict_leading_activity(self, simulation_object: SimulationObjectInstance):
        oid = simulation_object.oid
        otype = simulation_object.otype
        features_by_object = self.objectFeatures[otype][oid]
        object_features = tuple(list(map(lambda feature: int(features_by_object[feature]), self.objectFeatureNames)))
        next_act_predictor = self.predictors.next_activity_predictors[otype]
        if object_features in next_act_predictor:
            predictions = next_act_predictor[object_features]
            if predictions:
                cum_dist = CumulativeDistribution(predictions)
                prediction = cum_dist.sample()
                transition = self.__get_transition(prediction)
                if transition.transitionType == TransitionType.FINAL or \
                        self.processConfig.activityLeadingTypes[transition.label] == otype:

        self.scheduledLeadSteps[oid] = None

    def predict(self):
        active_simulation_objects = self.simulationNet.get_all_active_simulation_objects()
        active_simulation_objects.sort(key=lambda so: so.nextActivity.time)


    def predict2(self):
        running_tokens = self.simulationNet.get_all_running_emitting_tokens()
        running_tokens.sort(key=lambda t: t.time)
        token: Token
        prediction = None
        objects_by_id = self.simulationNet.objects.objectsById
        for token in running_tokens:
            oid = token.oid
            obj = objects_by_id[oid]
            if oid not in self.scheduledLeadSteps:
                self.__schedule_lead_step(obj)
            if self.scheduledLeadSteps[oid] is None:
                continue
            scheduled_transition = self.scheduledLeadSteps[oid]
            scheduled_activity = scheduled_transition.label
            if scheduled_transition.transitionType == TransitionType.FINAL:
                leading_obj = obj
                path = self.simulationNet.compute_path(obj, scheduled_activity)
                paths_by_objs = {obj: path} if path is not None else None
            else:
                leading_obj, paths_by_objs = self.__get_all_paths(obj, scheduled_activity)
            if paths_by_objs is not None:
                prediction = NextActivityCandidate(scheduled_transition, leading_obj, paths_by_objs)
                do_predict = self.__decide_prediction(prediction)
                if do_predict:
                    break
                else:
                    prediction = None
        if prediction is None:
            return False
        prediction: NextActivityCandidate
        if prediction.transition.transitionType == TransitionType.FINAL:
            self.simulationNet.terminatingObjects.add(prediction.leadingObject.oid)
        self.__update_enabled_transitions(prediction.paths)
        return True

    def __decide_prediction(self, prediction: NextActivityCandidate):
        p = 0
        n = 0
        zeros = 0
        act = prediction.transition.label
        objs = prediction.paths.keys()
        for obj in objs:
            if obj == prediction.leadingObject or obj.otype in self.processConfig.nonEmittingTypes:
                continue
            n = n + 1
            otype = obj.otype
            if otype in self.processConfig.nonEmittingTypes:
                continue
            features_by_object = self.objectFeatures[otype][obj.oid]
            object_features = tuple(
                list(map(lambda feature: int(features_by_object[feature]), self.objectFeatureNames)))
            next_act_predictor = self.predictors.next_activity_predictors[otype]
            if object_features in next_act_predictor \
                    and act in next_act_predictor[object_features]:
                p_obj = next_act_predictor[object_features][act]
            else:
                p_obj = 0
            p = p + p_obj
            if p_obj == 0:
                zeros += 1
        p = p / n if n > 0 else 1
        if zeros * 2 > n:
            return False
        rnd = random.random()
        if rnd > p:
            return False
        return True

    def __schedule_lead_step(self, obj):
        oid = obj.oid
        otype = obj.otype
        features_by_object = self.objectFeatures[otype][oid]
        object_features = tuple(list(map(lambda feature: int(features_by_object[feature]), self.objectFeatureNames)))
        next_act_predictor = self.predictors.next_activity_predictors[otype]
        if object_features in next_act_predictor:
            predictions = next_act_predictor[object_features]
            if predictions:
                cum_dist = CumulativeDistribution(predictions)
                prediction = cum_dist.sample()
                transition = self.__get_transition(prediction)
                if transition.transitionType == TransitionType.FINAL or \
                        self.processConfig.activityLeadingTypes[transition.label] == otype:
                    self.scheduledLeadSteps[oid] = transition
                    return
        self.scheduledLeadSteps[oid] = None

    def __get_execution_probability(self, candidate: NextActivityCandidate):
        candidate_activity = candidate.transition.label
        leading_type = candidate.leadingObject.otype
        bound_objects = candidate.paths.keys()
        obj: ObjectInstance
        p = 1
        for obj in bound_objects:
            otype = obj.otype
            if otype in self.processConfig.nonEmittingTypes:
                continue
            features_by_object = self.objectFeatures[otype][obj.oid]
            object_features = tuple(
                list(map(lambda feature: int(features_by_object[feature]), self.objectFeatureNames)))
            next_act_predictor = self.predictors.next_activity_predictors[otype]
            if object_features in next_act_predictor and candidate_activity in next_act_predictor[object_features]:
                probability = next_act_predictor[object_features][candidate_activity]
                p = min(probability, p)
                continue
            else:
                p = 0
        return p

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
        if transition_id in self.processConfig.acts or transition_id[:4] == "END_":
            obj_ids_str = ", ".join([str(i) for i in object_ids])
            date_str = datetime.utcfromtimestamp(timestamp).strftime("%d/%m/%Y, %H:%M:%S")
            logging.info(f"Executed {transition_id} with {obj_ids_str} at {date_str}")
            self.__update_features(transition_id, object_ids)
            self.__clear_lead_steps_scheduling(object_ids)
            do_next = self.predict()
            if not do_next:
                logging.info("Activity could not be predicted. Terminating simulation...")
                return True
            if not transition_id[:4] == "END_":
                self.__update_ocel(transition_id, object_ids, timestamp)
        # put next transition to be executed in front
        self.__sort_bindings()

    def __clear_lead_steps_scheduling(self, object_ids):
        for object_id in object_ids:
            if object_id in self.scheduledLeadSteps:
                del self.scheduledLeadSteps[object_id]

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
            feature_vector = next_act
            predictors = self.predictors.delay_predictors[otype]
            delay = predictors[feature_vector]
            delays[obj.oid] = delay
            delay_ws.append(str(obj) + ":" + str(delay))
        return delays

    def __ready_for_termination(self, obj: ObjectInstance):
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
