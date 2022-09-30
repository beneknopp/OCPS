import logging
import math
import os
import pickle
import random

import pandas as pd
import pm4py

from object_model_generation.initial_seed_maker import InitialSeedMaker
from object_model_generation.object_instance import ObjectInstance
from object_model_generation.object_link_prediction import PredictionMode, ObjectLinkPrediction
from object_model_generation.object_model import ObjectModel
from object_model_generation.object_model_parameters import ObjectModelParameters
from object_model_generation.object_type_graph import ObjectTypeGraph
from object_model_generation.training_model_preprocessor import TrainingModelPreprocessor
from utils.arrival_time_distribution import ArrivalTimeDistribution
from utils.cumulative_distribution import CumulativeDistribution
from utils.running_id import RunningId


class ObjectModelGenerator:
    trainingModelPreprocessor: TrainingModelPreprocessor
    objectTypeGraph: ObjectTypeGraph

    def __init__(self, session_path, ocel, object_model_parameters: ObjectModelParameters,
                 training_model_preprocessor: TrainingModelPreprocessor):
        self.sessionPath = session_path
        self.ocel = ocel
        self.objectModelParameters = object_model_parameters
        self.objectTypeGraph = training_model_preprocessor.objectTypeGraph
        self.otypes = training_model_preprocessor.otypes
        self.trainingModelPreprocessor = training_model_preprocessor

    def generate(self):
        self.__initialize_object_instance_class()
        self.__run_generation()
        self.__reindex_generated_objects()
        self.__make_arrival_times()

    def __make_arrival_times(self):
        self.__make_prior_arrival_times_distributions()
        self.__make_relative_arrival_times_distributions()
        self.__assign_arrival_times()
        self.__make_arrival_stats()

    def save(self, session_path):
        obj: ObjectInstance
        object_model = ObjectModel(session_path)
        for otype in self.otypes:
            generated_of_type = self.generatedObjects[otype]
            otype_model = {
                obj: [adj_obj
                      for adj_type in obj.total_local_model
                      for adj_obj in obj.total_local_model[adj_type]
                      ]
                for obj in generated_of_type
            }
            object_model.addModel(otype, otype_model)
            ################################
            for i, any_otype in enumerate(self.otypes):
                otype_relations = dict()
                for obj in generated_of_type:
                    card = len(obj.global_model[any_otype])
                    if card not in otype_relations:
                        otype_relations[card] = 0
                    otype_relations[card] = otype_relations[card] + 1
                if len(otype_relations) > 0:
                    min_card = min(list(otype_relations.keys()))
                    max_card = max(list(otype_relations.keys()))
                    x_axis = range(min_card, max_card + 1)
                    total = sum(otype_relations.values())
                    log_based_schema_dist = list(map(lambda card: float(otype_relations[card]) / total
                    if card in otype_relations else 0, x_axis))
                    stats = {"simulated": log_based_schema_dist, "x_axis": x_axis}
                else:
                    stats = {"simulated": [1], "x_axis": [0]}
                dist_path = os.path.join(session_path, otype + "_to_" + any_otype + "_schema_dist_simulated.pkl")
                with open(dist_path, "wb") as wf:
                    pickle.dump(stats, wf)
        object_model.save()

    def __initialize_object_instance_class(self):
        global_schemata = {
            otype: {
                schema: list(oid_to_schema.values()).count(schema)
                for schema in set(oid_to_schema.values())
            }
            for otype, oid_to_schema in self.trainingModelPreprocessor.flatGlobalSchemata.items()
        }
        ObjectInstance.set_(
            execution_model_paths=self.objectTypeGraph.shortest_paths,
            otypes=self.otypes,
            global_schemata=global_schemata
        )

    def __run_generation(self):
        logging.info("Running Generation of Object Model...")
        object_model_parameters = self.objectModelParameters
        object_type_graph = self.objectTypeGraph
        seed_type = object_model_parameters.seedType
        number_of_objects = object_model_parameters.numberOfObjects
        self.nonEmittingTypes = object_model_parameters.nonEmittingTypes
        oid = RunningId()
        open_objects = {otype: [] for otype in self.otypes}
        closed_objects = {otype: [] for otype in self.otypes}
        total_objects = {otype: [] for otype in self.otypes}
        buffer = []
        # InitialSeedMaker.initialize_unconnected_objs(
        #   self.trainingModelPreprocessor.leading_type_process_executions, oid, buffer, seed_type, number_of_objects,
        #  open_objects, total_objects)
        # for i in range(number_of_objects):
        InitialSeedMaker.create_obj(buffer, seed_type, oid, open_objects, total_objects)
        self.enforcements = 0
        while len(buffer) > 0:
            current_obj = buffer[0]
            buffer = buffer[1:]
            current_otype = current_obj.otype
            neighbor_types = object_type_graph.get_neighbor_otypes(current_otype)
            if current_otype not in self.nonEmittingTypes:
                prediction: ObjectLinkPrediction = self.__predict_neighbor(
                    current_obj, neighbor_types, open_objects, oid)
                if prediction.predict:
                    selected_neighbor = prediction.selected_neighbor
                    predicted_type = prediction.predicted_type
                    reverse = prediction.reverse
                    if not reverse:
                        ObjectInstance.merge(current_obj, selected_neighbor)
                    else:
                        ObjectInstance.merge(selected_neighbor, current_obj)
                    if selected_neighbor not in total_objects[predicted_type]:
                        total_objects[predicted_type].append(selected_neighbor)
                    if selected_neighbor not in buffer:
                        buffer.insert(random.randrange(len(buffer) + 1), selected_neighbor)
                    buffer.insert(random.randrange(len(buffer) + 1), current_obj)
                else:
                    open_objects[current_otype] = list(
                        set([x for x in open_objects[current_otype] if not x == current_obj]))
                    closed_objects[current_otype].append(current_obj)
            if len(buffer) == 0 and len(closed_objects[seed_type]) < number_of_objects:
                InitialSeedMaker.create_obj(buffer, seed_type, oid, open_objects, total_objects)
        self.generatedObjects = total_objects

    def __sort_buffer(self, buffer):
        obj: ObjectInstance
        buffer.sort(key=lambda obj: self.otypes.index(obj.otype))

    def __reindex_generated_objects(self):
        generated_objects = self.generatedObjects
        index = 1
        for otype in self.otypes:
            for obj in generated_objects[otype]:
                obj.oid = index
                index = index + 1

    def __make_prior_arrival_times_distributions(self):
        arrival_times_distributions = {}
        self.flattenedLogs = dict()
        logging.info("Making arrival rate distributions...")
        for otype in self.otypes:
            logging.info(otype + "...")
            flattened_log = pm4py.ocel_flattening(self.ocel, otype)
            flattened_log = flattened_log.sort_values(["time:timestamp"])
            self.flattenedLogs[otype] = flattened_log
            arrival_times = flattened_log.groupby("case:concept:name").first()["time:timestamp"]
            arrival_times = arrival_times.sort_values()
            arrival_times = arrival_times.apply(lambda row: row.timestamp())
            arrival_rates = arrival_times.diff()[1:]
            dist = ArrivalTimeDistribution(arrival_rates)
            arrival_times_distributions[otype] = dist
        self.arrivalTimesDistributions = arrival_times_distributions

    def __make_relative_arrival_times_distributions(self):
        log_based_relative_arrival_times = {
            otype: {
                any_otype: []
                for any_otype in self.otypes
            } for otype in self.otypes
        }
        arrival_times = dict()
        for otype in self.otypes:
            flattened_log = self.flattenedLogs[otype]
            ot_arrival_times = flattened_log.groupby("case:concept:name").first()["time:timestamp"]
            ot_arrival_times = ot_arrival_times.apply(lambda row: row.timestamp())
            arrival_times[otype] = ot_arrival_times
        global_model = self.trainingModelPreprocessor.globalObjectModel
        dists = dict()
        for otype in self.otypes:
            dists[otype] = dict()
            for obj, obj_model in global_model[otype].items():
                arrival_time = arrival_times[otype][obj]
                for any_otype in self.otypes:
                    log_based_rel_times = log_based_relative_arrival_times[otype][any_otype]
                    for related_obj in obj_model[any_otype]:
                        related_arrival_time = arrival_times[any_otype][related_obj]
                        relative_arrival_time = related_arrival_time - arrival_time
                        log_based_rel_times.append(relative_arrival_time)
            for any_otype in self.otypes:
                log_based_rel_times = log_based_relative_arrival_times[otype][any_otype]
                if log_based_rel_times:
                    dists[otype][any_otype] = ArrivalTimeDistribution(pd.Series(log_based_rel_times))
        self.relativeArrivalTimesDistributions = dists
        self.logBasedRelativeArrivalTimes = log_based_relative_arrival_times

    # TODO: assign arrival times relative to related objects
    def __make_arrival_stats(self):
        log_based_relative_arrival_times = self.logBasedRelativeArrivalTimes
        simulated_relative_arrival_times = {otype: {any_otype: [] for any_otype in self.otypes} for otype in
                                            self.otypes}
        for otype in self.otypes:
            obj: ObjectInstance
            for obj in self.generatedObjects[otype]:
                arrival_time = obj.time
                for any_otype in self.otypes:
                    for related_obj in obj.global_model[any_otype]:
                        related_arrival_time = related_obj.time
                        relative_arrival_time = related_arrival_time - arrival_time
                        simulated_relative_arrival_times[otype][any_otype].append(relative_arrival_time)
            log_based_arr_stats_path = os.path.join(self.sessionPath, "arrival_times_" + otype + "_log_based.pkl")
            simulated_arr_stats_path = os.path.join(self.sessionPath, "arrival_times_" + otype + "_simulated.pkl")
            with open(log_based_arr_stats_path, "wb") as wf:
                pickle.dump(log_based_relative_arrival_times[otype], wf)
            with open(simulated_arr_stats_path, "wb") as wf:
                pickle.dump(simulated_relative_arrival_times[otype], wf)

    def __assign_arrival_times(self):
        logging.info("Assigning Arrival Times...")
        seed_type = self.objectModelParameters.seedType
        seed_objects = list(self.generatedObjects[seed_type])
        running_timestamps = {otype: 0 for otype in self.otypes}
        buffer = list(seed_objects)
        all_objects = [obj for sl in self.generatedObjects.values() for obj in sl]
        handled_objs = set()
        current_obj: ObjectInstance
        related_obj: ObjectInstance
        while len(buffer) > 0:
            current_obj = buffer[0]
            buffer = buffer[1:]
            current_otype = current_obj.otype
            if current_otype == seed_type:
                current_obj.time = running_timestamps[current_otype]
                handled_objs.add(current_obj)
                time = round(self.arrivalTimesDistributions[seed_type].sample())
                running_timestamps[current_otype] = running_timestamps[current_otype] + time
            open_local_model = {
                obj for sl in current_obj.total_local_model.values()
                for obj in sl
                if obj not in handled_objs
            }
            for related_obj in open_local_model:
                related_type = related_obj.otype
                if related_type != seed_type:
                    relative_arrival_time = round(
                        self.relativeArrivalTimesDistributions[current_otype][related_type].sample())
                    related_obj.time = current_obj.time + relative_arrival_time
                handled_objs.add(related_obj)
                buffer = buffer + [related_obj]
            if len(buffer) == 0 and len(handled_objs) != len(all_objects):
                new_seed_obj = [x for x in all_objects if x not in handled_objs][0]
                new_seed_obj.time = round(self.arrivalTimesDistributions[new_seed_obj.otype].sample())
                buffer = [new_seed_obj]
        if len(handled_objs) != len(all_objects):
            raise ValueError("Not all objects have been assigned an arrival time")
        min_time = min(map(lambda obj: obj.time, all_objects))
        for obj in all_objects:
            obj.time = obj.time - min_time
        seed_objects.sort(key=lambda x: x.time)

    def __assign_arrival_times_by_bfs2(self):
        otypes = self.otypes
        arrival_times_distributions = self.arrivalTimesDistributions
        running_timestamps = {otype: 0 for otype in otypes}
        open_objects = set()
        for otype in self.otypes:
            open_objects.update(self.generatedObjects[otype])
        first_obj = random.sample(open_objects, 1)[0]
        open_objects.remove(first_obj)
        buffer = [first_obj]
        current_obj: ObjectInstance
        assigned_objects = set()
        while len(buffer) > 0:
            current_obj = buffer[0]
            current_otype = current_obj.otype
            if current_obj not in assigned_objects:
                arrival_time = running_timestamps[current_otype] + \
                               round(arrival_times_distributions[current_otype].sample())
                current_obj.set_timestamp(arrival_time)
                assigned_objects.add(current_obj)
                running_timestamps[current_otype] = arrival_time
            open_local_model = [obj for otype in self.otypes for obj in
                                current_obj.total_local_model[otype]
                                if obj in open_objects]
            if len(open_local_model) == 0:
                buffer = buffer[1:]
                if len(buffer) == 0 and len(open_objects) > 0:
                    buffer = random.sample(open_objects, 1)
                continue
            open_objects = open_objects.difference(open_local_model)
            buffer = buffer + open_local_model
        min_time = min(map(lambda obj: obj.time, [obj for sl in self.generatedObjects.values() for obj in sl]))
        for sl in self.generatedObjects.values():
            for obj in sl:
                obj.time = obj.time - min_time

    def __predict_neighbor(self, obj: ObjectInstance, neighbor_types, open_objects, oid: RunningId):
        supported_objs = {}
        new_objs = []
        parent_types = neighbor_types["parents"]
        child_types = neighbor_types["children"]
        neighbor_types = parent_types + child_types
        neighbor_types = [nt for nt in neighbor_types if not obj.locally_closed_types[nt]]
        random.shuffle(neighbor_types)
        for neighbor_otype in neighbor_types:
            # try new instance for that otype
            supported_objs[neighbor_otype] = []
            new_obj = ObjectInstance(neighbor_otype, oid.get())
            # choice: decide action based on direct support
            direct_support = self.__compute_pairwise_support(obj, new_obj)
            local_support = self.__compute_global_support(obj, new_obj)
            max_support = local_support
            new_objs.append(new_obj)
            supported_objs[neighbor_otype].append((new_obj, local_support))
            open_neighbors = open_objects[neighbor_otype]
            # avoid bias towards specific objects
            random.shuffle(open_objects[neighbor_otype])
            open_neighbors = list(filter(lambda on:
                                         # neighbor still open, but not connected to this object yet
                                         on not in obj.direct_object_model[neighbor_otype] and on not in
                                         obj.reverse_object_model[neighbor_otype],
                                         open_neighbors
                                         ))
            open_neighbor: ObjectInstance
            for open_neighbor in open_neighbors:
                global_support = self.__compute_global_support(obj, open_neighbor)
                supported_objs[neighbor_otype].append((open_neighbor, global_support))
                if global_support > max_support:
                    max_support = global_support
            rnd = random.random()
            # if rnd > max_support:
            if rnd > direct_support:
                obj.close_type(neighbor_otype)
                continue
            if not sum(list(map(lambda x: x[1], supported_objs[neighbor_otype]))) > 0:
                if direct_support > 0.99:
                    # enforce (contradicting supports, so prioritize local support)
                    supported_objs[neighbor_otype] = [(obj, 1) for (obj, x) in supported_objs[neighbor_otype]]
                    self.enforcements = self.enforcements + 1
                else:
                    obj.close_type(neighbor_otype)
                    continue
            probs = {o: p for (o, p) in supported_objs[neighbor_otype]}
            cum_dist = CumulativeDistribution(probs)
            selected_neighbor = cum_dist.sample()
            predicted_otype = selected_neighbor.otype
            mode = PredictionMode.NEW if selected_neighbor in new_objs else PredictionMode.APPEND
            reverse = True if predicted_otype in parent_types else False
            if mode == PredictionMode.NEW:
                oid.inc()
                open_objects[predicted_otype].append(selected_neighbor)
            prediction = ObjectLinkPrediction(predict=True, predicted_type=predicted_otype, mode=mode, reverse=reverse,
                                              selected_neighbor=selected_neighbor)
            # logging.info(f"{obj.otype} {str(obj.oid)}: {str(prediction.pretty_print()}"))
            return prediction
        return ObjectLinkPrediction(predict=False)

    def __compute_global_support(self, left_object: ObjectInstance, right_object: ObjectInstance):
        left_otype = left_object.otype
        right_otype = right_object.otype
        (left_side, right_side) = self.objectTypeGraph.get_component_split(left_otype, right_otype)
        execution_model_object: ObjectInstance
        left_global_model = [left_object]
        right_global_model = [right_object]
        for otype in left_side:
            global_model_of_type = [left_object.global_model[otype]]
            left_global_model += [el for sl in global_model_of_type for el in sl]
        for otype in right_side:
            global_model_of_type = [right_object.global_model[otype]]
            right_global_model += [el for sl in global_model_of_type for el in sl]
        global_support = 1.0
        for left_model_object in left_global_model:
            if left_model_object.otype in self.nonEmittingTypes:
                continue
            support = self.__compute_element_support(left_model_object, right_global_model)
            if support < global_support:
                global_support = support
        for right_model_object in right_global_model:
            if right_model_object.otype in self.nonEmittingTypes:
                continue
            support = self.__compute_element_support(right_model_object, left_global_model)
            if support < global_support:
                global_support = support
        return global_support

    def __compute_element_support(self, obj: ObjectInstance, object_model):
        element_support = 1
        ot = obj.otype
        for otype in self.otypes:
            current_number_at_obj = len(obj.global_model[otype])
            additions = len([any_obj for any_obj in object_model
                             if any_obj.otype == otype
                             and any_obj not in obj.global_model[otype]])
            for i in range(additions):
                support = obj.support_distributions[otype].get_support(current_number_at_obj + i + 1)
                element_support = support * element_support
        return element_support

    def __compute_pairwise_support(self, obj1: ObjectInstance, obj2: ObjectInstance):
        ot1 = obj1.otype
        ot2 = obj2.otype
        if obj1 in obj2.global_model[ot1]:
            if obj2 not in obj1.global_model[ot2]:
                raise ValueError("How can it be?")
            return 1.0
        if ot1 in self.nonEmittingTypes:
            support_for_obj2_at_obj1 = 1.0
        else:
            nof_ot2_at_obj1 = len(obj1.global_model[ot2])
            support_for_obj2_at_obj1 = obj1.support_distributions[ot2].get_support(nof_ot2_at_obj1 + 1)
        if ot2 in self.nonEmittingTypes:
            support_for_obj1_at_obj2 = 1.0
        else:
            nof_ot1_at_obj2 = len(obj2.global_model[ot1])
            support_for_obj1_at_obj2 = obj2.support_distributions[ot1].get_support(nof_ot1_at_obj2 + 1)
        # return support_for_obj1_at_obj2*support_for_obj2_at_obj1
        return (min(support_for_obj1_at_obj2, support_for_obj2_at_obj1))

    def get_response(self):
        response_dict = dict()
        for ot in self.otypes:
            response_dict[ot] = dict()
            generated_objects = self.generatedObjects[ot]
            response_dict[ot]["simulation_stats"] = self.__get_arrival_rate_mean_stdev(generated_objects)
            orig_mean = self.arrivalTimesDistributions[ot].mean
            orig_stdev = self.arrivalTimesDistributions[ot].stdev
            response_dict[ot]["original_stats"] = {
                "mean": orig_mean,
                "stdev": orig_stdev
            }
            response_dict[ot]["simulation_stats"]["relations"] = dict()
            for any_ot in self.otypes:
                relation_stats = self.__get_relation_mean_stdev(generated_objects, any_ot)
                response_dict[ot]["simulation_stats"]["relations"][any_ot] = relation_stats
            response_dict[ot]["simulation_stats"]["number_of_objects"] = len(generated_objects)
        return response_dict

    def __get_relation_mean_stdev(self, objs, otype):
        rel_cards = list(map(lambda obj: len(obj.total_local_model[otype]), objs))
        rel_cards = pd.Series(rel_cards)
        if len(rel_cards) == 0:
            return {
                "mean": 0,
                "stdev": 0
            }
        rel_card_mean = rel_cards.mean()
        rel_card_stdev = math.sqrt(rel_cards.var()) if len(rel_cards) > 1 else 0
        return {
            "mean": rel_card_mean,
            "stdev": rel_card_stdev
        }

    def __get_arrival_rate_mean_stdev(self, objs):
        if len(objs) < 2:
            return {
                "mean": 0,
                "stdev": 0
            }
        obj_arrivals = pd.Series(list(map(lambda obj: obj.time, objs))).sort_values()
        obj_arrival_rates = obj_arrivals.diff()[1:]
        mean = obj_arrival_rates.mean()
        stdev = math.sqrt(obj_arrival_rates.var())
        return {
            "mean": mean,
            "stdev": stdev
        }
