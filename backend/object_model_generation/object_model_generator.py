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
        schema_frequencies = {
            otype: dict()
            for otype in self.otypes
        }
        for otype, objs in self.generatedObjects.items():
            otype_model = {
                obj: [adj_obj
                      for adj_type in obj.total_local_model
                      for adj_obj in obj.total_local_model[adj_type]
                      ]
                for obj in objs
            }
            object_model.addModel(otype, otype_model)
            obj: ObjectInstance
            for obj in objs:
                for depth in obj.global_model:
                    for path, related_objs in obj.global_model[depth].items():
                        if path not in schema_frequencies[otype]:
                            schema_frequencies[otype][path] = {}
                        card = len(related_objs)
                        if card not in schema_frequencies[otype][path]:
                            schema_frequencies[otype][path][card] = 0
                        schema_frequencies[otype][path][card] = schema_frequencies[otype][path][card] + 1
        for otype, paths_dict in schema_frequencies.items():
            for path, cardinality_distribution in paths_dict.items():
                if len(path) < 1:
                    continue
                if len(cardinality_distribution) < 1:
                    continue
                print(path)
                min_card = min(cardinality_distribution)
                max_card = max(cardinality_distribution)
                x_axis = range(min_card, max_card + 1)
                total = sum(cardinality_distribution.values())
                simulated_schema_dist = list(map(lambda card: float(cardinality_distribution[card]) / total
                if card in cardinality_distribution else 0, x_axis))
                stats = {"simulated": simulated_schema_dist, "x_axis": x_axis}
                dist_path = os.path.join(self.sessionPath, str(path) + "_schema_dist_simulated.pkl")
                with open(dist_path, "wb") as wf:
                    pickle.dump(stats, wf)
        object_model.save_without_global_model()

    def __initialize_object_instance_class(self):
        ObjectInstance.set_(
            otypes=self.otypes,
            execution_model_paths=self.trainingModelPreprocessor.executionModelPaths,
            execution_model_depth=self.trainingModelPreprocessor.executionModelDepth,
            schema_distributions=self.trainingModelPreprocessor.schemaDistributions
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
            neighbor_types = object_type_graph.get_parent_and_child_otypes(current_otype)
            if current_otype not in self.nonEmittingTypes:
                prediction: ObjectLinkPrediction = self.__predict_neighbor(
                    current_obj, neighbor_types, open_objects, oid)
                if prediction.predict:
                    selected_neighbor = prediction.selected_neighbor
                    predicted_type = prediction.predicted_type
                    reverse = prediction.reverse
                    merge_map = prediction.mergeMap
                    if not reverse:
                        ObjectInstance.merge(current_obj, selected_neighbor, merge_map)
                    else:
                        ObjectInstance.merge(selected_neighbor, current_obj, merge_map)
                    if selected_neighbor not in total_objects[predicted_type]:
                        total_objects[predicted_type].append(selected_neighbor)
                    buffer = buffer + [current_obj]
                    if selected_neighbor not in buffer:
                        buffer = buffer + [selected_neighbor]
                        #buffer.insert(random.randrange(len(buffer) + 1), selected_neighbor)
                    buffer.insert(random.randrange(len(buffer) + 1), current_obj)
                else:
                    open_objects[current_otype] = list(
                        set([x for x in open_objects[current_otype] if not x == current_obj]))
                    closed_objects[current_otype].append(current_obj)
            if len(buffer) == 0 and len(closed_objects[seed_type]) < number_of_objects:
                InitialSeedMaker.create_obj(buffer, seed_type, oid, open_objects, total_objects)
            if len(total_objects[seed_type]) > number_of_objects:
                break
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
        # otype -> depths -> paths -> objs -> model
        global_model = self.trainingModelPreprocessor.globalObjectModel
        dists = dict()
        model_depth = 1
        for otype in self.otypes:
            dists[otype] = dict()
            for path, obj_models in global_model[otype][model_depth].items():
                for obj, obj_model in obj_models.items():
                    arrival_time = arrival_times[otype][obj]
                    any_otype = path[-1]
                    log_based_rel_times = log_based_relative_arrival_times[otype][any_otype]
                    for related_obj in obj_model:
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
                    path = (otype, any_otype)
                    if path not in obj.global_model[1]:
                        continue
                    for related_obj in obj.global_model[1][path]:
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
            local_support, local_merge_map, zero_ratio = self.__compute_global_support(obj, new_obj)
            local_candidate = (new_obj, local_support, local_merge_map, zero_ratio)
            max_support = local_support
            new_objs.append(new_obj)
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
            max_global_support = 0
            for open_neighbor in open_neighbors:
                global_support, merge_map, zero_ratio = self.__compute_global_support(obj, open_neighbor)
                supported_objs[neighbor_otype].append((open_neighbor, global_support, merge_map, zero_ratio))
                if global_support > max_support:
                    max_support = global_support
                if global_support > max_global_support:
                    max_global_support = global_support
            if local_support > max_global_support:
                # heuristic to prefer existing objects instead of creating new ones
                supported_objs[neighbor_otype].append(local_candidate)
            rnd = random.random()
            # if rnd > max_support:
            if rnd > direct_support:
                obj.close_type(neighbor_otype)
                continue
            if not sum(list(map(lambda x: x[1], supported_objs[neighbor_otype]))) > 0:
                if direct_support > 0.99:
                    # enforce (contradicting supports, so prioritize local support)
                    enforced_candidates = supported_objs[neighbor_otype]
                    if not any(candidate[0] == new_obj for candidate in enforced_candidates):
                        enforced_candidates += [local_candidate]
                    supported_objs[neighbor_otype] = [(obj, 1-zero_ratio, merge_map, zero_ratio)
                                                      for (obj, p, merge_map, zero_ratio) in enforced_candidates]
                    self.enforcements = self.enforcements + 1
                else:
                    obj.close_type(neighbor_otype)
                    continue
            merge_maps = {
                o: mm for (o, p, mm, zero_ratio) in supported_objs[neighbor_otype]
            }
            probs = {o: p for (o, p, mm, zero_ratio) in supported_objs[neighbor_otype]}
            try:
                cum_dist = CumulativeDistribution(probs)
            except:
                raise ValueError("Why?")
            selected_neighbor = cum_dist.sample()
            merge_map = merge_maps[selected_neighbor]
            predicted_otype = selected_neighbor.otype
            mode = PredictionMode.NEW if selected_neighbor in new_objs else PredictionMode.APPEND
            reverse = True if predicted_otype in parent_types else False
            if mode == PredictionMode.NEW:
                oid.inc()
                open_objects[predicted_otype].append(selected_neighbor)
            prediction = ObjectLinkPrediction(predict=True, predicted_type=predicted_otype, mode=mode, reverse=reverse,
                                              selected_neighbor=selected_neighbor, merge_map=merge_map)
            # logging.info(f"{obj.otype} {str(obj.oid)}: {str(prediction.pretty_print()}"))
            return prediction
        return ObjectLinkPrediction(predict=False)

    def __compute_global_support(self, left_object: ObjectInstance, right_object: ObjectInstance):
        left_otype = left_object.otype
        right_otype = right_object.otype
        paths = {}
        level_objs = {}
        level = 1
        path = tuple([left_otype, right_otype])
        level_objs[level] = dict()
        level_objs[level][path] = [[left_object], [right_object]]
        cut_index = 1
        # supports for objects on left margin to be evaluated: yes, right side: yes
        paths[level] = [(path, cut_index)]
        left_border_object: ObjectInstance
        right_border_object: ObjectInstance
        global_support = 1.0
        merge_map = dict()
        zero_supports = 0
        pairwise_supports_counts = 0
        while True:
            # evaluate current level supports
            for path, cut_index in paths[level]:
                if len(path) == 4:
                    print(path)
                if path == ('MATERIAL', 'LEAD_Plan Goods Issue', 'MATERIAL', 'LEAD_Create Purchase Order'):
                    print("hi")
                path_objects = level_objs[level][path]
                left_border_objects = path_objects[0]
                right_border_objects = path_objects[-1]
                # path objects are ordered according to the path
                reverse_path = list(path[:])
                reverse_path.reverse()
                reverse_path = tuple(reverse_path)
                reversed_path_objects = path_objects[:]
                reversed_path_objects.reverse()
                for left_border_object in left_border_objects:
                    support = self.__compute_element_support(left_border_object, path_objects, path, cut_index)
                    if support < global_support:
                        global_support = support
                    if support == 0:
                        zero_supports += 1
                    pairwise_supports_counts += 1
                    self.__update_merge_map(merge_map, left_border_object, path_objects, path, cut_index)
                for right_border_object in right_border_objects:
                    support = self.__compute_element_support(
                        right_border_object, reversed_path_objects, reverse_path, level + 1 - cut_index)
                    if support < global_support:
                        global_support = support
                    if support == 0:
                        zero_supports += 1
                    pairwise_supports_counts += 1
                    self.__update_merge_map(merge_map, right_border_object, reversed_path_objects, reverse_path, level + 1 - cut_index)
            if level == ObjectInstance.executionModelDepth:
                break
            # new step
            new_paths = []
            level_objs[level + 1] = dict()
            for path, cut_index in paths.get(level):
                path_objects = level_objs[level][path]
                left_border_type = path[0]
                right_border_type = path[-1]
                left_border_objects = path_objects[0]
                right_border_objects = path_objects[-1]
                if left_border_objects:
                    left_extensions = self.objectTypeGraph.get_neighbor_otypes(left_border_type)

                    for left_extension_type in left_extensions:
                        new_path = tuple([left_extension_type] + list(path))
                        new_paths.append((new_path, cut_index + 1))
                        left_margin_path = tuple([left_border_type, left_extension_type])
                        new_objs = []
                        for left_obj in left_border_objects:
                            new_objs += left_obj.global_model[1][left_margin_path]
                        if left_extension_type == right_otype and left_object in left_border_objects:
                            new_objs += [right_object]
                        if left_extension_type == left_otype and right_object in left_border_objects:
                            new_objs += [left_object]
                        level_objs[level + 1][new_path] = [list(set(new_objs))] + path_objects
                if right_border_objects:
                    right_extensions = self.objectTypeGraph.get_neighbor_otypes(right_border_type)
                    for right_extension_type in right_extensions:
                        new_path = tuple(list(path) + [right_extension_type])
                        new_paths.append((new_path, cut_index))
                        right_margin_path = tuple([right_border_type, right_extension_type])
                        new_objs = []
                        for right_obj in right_border_objects:
                            new_objs += right_obj.global_model[1][right_margin_path]
                        if right_extension_type == left_otype and right_object in right_border_objects:
                            new_objs += [left_object]
                        if right_extension_type == right_otype and left_object in right_border_objects:
                            new_objs += [right_object]
                        level_objs[level + 1][new_path] = path_objects + [list(set(new_objs))]
            level = level + 1
            paths[level] = new_paths
        zero_ratio = float(zero_supports) / pairwise_supports_counts
        return global_support, merge_map, zero_ratio

    def __compute_element_support(self, obj: ObjectInstance, path_objects, path, cut_index):
        otype = obj.otype
        if otype in self.nonEmittingTypes:
            return 1.0
        element_support = 1
        for depth in range(cut_index, len(path)):
            subpath = tuple(path[:depth + 1])
            current_objs = path_objects[depth]
            current_number_at_obj = len(obj.global_model[depth][subpath])
            additions = len([any_obj for any_obj in current_objs
                             if any_obj not in obj.global_model[depth][subpath]])
            for j in range(additions):
                support = obj.supportDistributions[subpath].get_support(current_number_at_obj + j + 1)
                element_support = min(support, element_support)
        return element_support

    def __update_merge_map(self, merge_map, obj: ObjectInstance, path_objects, path, cut_index):
        for depth in range(cut_index, len(path)):
            subpath = path[:depth + 1]
            current_objs = path_objects[depth]
            additions = [any_obj for any_obj in current_objs
                         if any_obj not in obj.global_model[depth][subpath]]
            if obj not in merge_map:
                merge_map[obj] = dict()
            merge_map[obj][subpath] = additions

    def __compute_pairwise_support(self, obj1: ObjectInstance, obj2: ObjectInstance):
        ot1 = obj1.otype
        ot2 = obj2.otype
        depth = 1
        if obj1 in obj2.global_model[depth][tuple([ot2, ot1])]:
            if obj2 not in obj1.global_model[depth][tuple([ot1, ot2])]:
                raise ValueError("How can it be?")
            return 1.0
        if ot1 in self.nonEmittingTypes:
            support_for_obj2_at_obj1 = 1.0
        else:
            nof_ot2_at_obj1 = len(obj1.global_model[depth][tuple([ot1, ot2])])
            support_for_obj2_at_obj1 = ObjectInstance.supportDistributions[ot1][tuple([ot1, ot2])].get_support(
                nof_ot2_at_obj1 + 1)
        if ot2 in self.nonEmittingTypes:
            support_for_obj1_at_obj2 = 1.0
        else:
            nof_ot1_at_obj2 = len(obj2.global_model[depth][tuple([ot2, ot1])])
            support_for_obj1_at_obj2 = ObjectInstance.supportDistributions[ot2][tuple([ot2, ot1])].get_support(
                nof_ot1_at_obj2 + 1)
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
