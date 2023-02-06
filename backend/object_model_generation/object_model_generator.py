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
        original_model = ObjectModel(session_path)
        object_model = ObjectModel(session_path)
        schema_frequencies = {
            otype: dict()
            for otype in self.otypes
        }
        original_models = {
            otype: {} for otype in self.otypes
        }
        multiply = 8
        for i in range(multiply):
            for otype, objs in self.trainingModelPreprocessor.totalObjectModel.items():
                arrival_times = dict(self.arrivalTimes[otype])
                for oid in objs:
                    obj_inst = ObjectInstance(otype, str(oid) + "_" + str(i))
                    time = round(float((arrival_times[oid])))
                    obj_inst.time = time
                    original_models[otype][ str(oid) + "_" + str(i) ] = obj_inst
        for otype, objs in self.trainingModelPreprocessor.directObjectModel.items():
            full_otype_model = {}
            for i in range(multiply):
                for oid, adj_objs in objs.items():
                    obj_inst: ObjectInstance = original_models[otype][ str(oid) + "_" + str(i) ]
                    all_adj_objs = []
                    for any_otype, any_objs in adj_objs.items():
                        for any_obj in any_objs:
                            any_obj_inst = original_models[any_otype][ str(any_obj) + "_" + str(i)]
                            obj_inst.direct_object_model[any_otype].add(any_obj_inst)
                            any_obj_inst.reverse_object_model[otype].add(obj_inst)
                            obj_inst.total_local_model[any_otype].add(any_obj_inst)
                            any_obj_inst.total_local_model[otype].add(obj_inst)
                            all_adj_objs.append(any_obj_inst)
                    full_otype_model[obj_inst] = all_adj_objs
            original_model.addModel( otype, full_otype_model)
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
        original_model.save_without_global_model(True)
        object_model.save_without_global_model(False)
        for otype, paths_dict in schema_frequencies.items():
            for path, cardinality_distribution in paths_dict.items():
                if len(path) < 1:
                    continue
                if len(cardinality_distribution) < 1:
                    continue
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



    def __initialize_object_instance_class(self):
        ObjectInstance.set_(
            otypes=self.otypes,
            execution_model_paths=self.trainingModelPreprocessor.executionModelPaths,
            execution_model_depth=self.trainingModelPreprocessor.executionModelDepth,
            execution_model_evaluation_depth=self.trainingModelPreprocessor.executionModelEvaluationDepth,
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
        forward_closed_objects = {otype: [] for otype in self.otypes}
        total_objects = {otype: [] for otype in self.otypes}
        buffer = [(InitialSeedMaker.create_obj(seed_type, oid, open_objects, total_objects), [seed_type])]
        self.enforcements = 0
        while len(buffer) > 0:
            current_obj, instantiation_path = buffer[0]
            buffer = buffer[1:]
            current_otype = current_obj.otype
            neighbor_types = object_type_graph.get_parent_and_child_otypes(current_otype)
            if current_otype not in self.nonEmittingTypes:
                prediction: ObjectLinkPrediction = self.__predict_neighbor(
                    current_obj, instantiation_path, neighbor_types, open_objects, oid)
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
                    buffer.insert(random.randrange(len(buffer) + 1), (selected_neighbor, instantiation_path + [predicted_type]))

            current_next_types = [ot for ot in object_type_graph.get_neighbor_otypes(current_otype)
                                  if ot not in instantiation_path]
            if all(current_obj.locally_closed_types[ot]
                   for ot in current_next_types):
                fcos = forward_closed_objects[current_otype]
                forward_closed_objects[current_otype] = list(set(fcos + [current_obj]))
                buffer = [x for x in buffer if not x[0] == current_obj]
            else:
                buffer.insert(random.randrange(len(buffer) + 1), (current_obj, instantiation_path))

            closed = float(sum(map(lambda otype: len(forward_closed_objects[otype]), self.otypes)))
            total = float(sum(map(lambda otype: len(total_objects[otype]), self.otypes)))
            if len(total_objects[seed_type]) > number_of_objects and closed/total > 0.98:
                break
            if len(buffer) == 0:
                buffer = [(InitialSeedMaker.create_obj(seed_type, oid, open_objects, total_objects), [seed_type])]
            if total % 100 == 0:
                print("Total nof objects: " + str((total)))
        print("enforcements: " + str(self.enforcements))
        self.generatedObjects = total_objects

    def __evaluate_local_closure(self, obj_a, obj_b):
        ot_a = obj_a.otype
        ot_b = obj_b.otype
        #new_a = ObjectInstance(ot_a, 0)
        new_b = ObjectInstance(ot_b, 0)
        rnd = random.random()
        y, direct_support_a, x, mm = self.__compute_global_support(obj_a, new_b)
        #y, direct_support_b, x, mm = self.__compute_global_support(obj_b, new_a)
        if rnd > direct_support_a:
            obj_a.close_type(ot_b)
        #if direct_support_b == 0:
        #if rnd > direct_support_b:
         #   obj_b.close_type(ot_a)

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

    def __make_prior_arrival_times_distributions2(self):
        arrival_times_distributions = {}
        self.flattenedLogs = dict()
        logging.info("Making arrival rate distributions...")
        self.arrivalTimes = {}
        for otype in self.otypes:
            logging.info(otype + "...")
            flattened_log = pm4py.ocel_flattening(self.ocel, otype)
            flattened_log = flattened_log.sort_values(["time:timestamp"])
            self.flattenedLogs[otype] = flattened_log
            arrival_times = flattened_log.groupby("case:concept:name").first()["time:timestamp"]
            arrival_times = arrival_times.sort_values()
            arrival_times = arrival_times.apply(lambda row: row.timestamp())
            self.arrivalTimes[otype] = arrival_times
            arrival_rates = arrival_times.diff()[1:]
            dist = ArrivalTimeDistribution(arrival_rates)
            arrival_times_distributions[otype] = dist
        self.arrivalTimesDistributions = arrival_times_distributions

    def __make_prior_arrival_times_distributions(self):
        arrival_times_distributions = {}
        self.flattenedLogs = dict()
        logging.info("Making arrival rate distributions...")
        self.arrivalTimes = {}
        for otype in self.otypes:
            logging.info(otype + "...")
            flattened_log = pm4py.ocel_flattening(self.ocel, otype)
            flattened_log = flattened_log.sort_values(["time:timestamp"])
            self.flattenedLogs[otype] = flattened_log
            arrival_times = flattened_log.groupby("case:concept:name").first()["time:timestamp"]
            arrival_times = arrival_times.sort_values()
            arrival_times = arrival_times.apply(lambda row: row.timestamp())
            self.arrivalTimes[otype] = arrival_times
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

    def __predict_neighbor(self, obj: ObjectInstance, instantiation_path, neighbor_types, open_objects, oid: RunningId):
        supported_objs = {}
        fallback_candidates = {}
        new_objs = []
        parent_types = neighbor_types["parents"]
        child_types = neighbor_types["children"]
        neighbor_types = parent_types + child_types
        neighbor_types = [nt for nt in neighbor_types if not obj.locally_closed_types[nt] and nt not in instantiation_path]
        #random.shuffle(neighbor_types)
        for neighbor_otype in neighbor_types:
            # try new instance for that otype
            supported_objs[neighbor_otype] = []
            fallback_candidates[neighbor_otype] = []
            new_obj = ObjectInstance(neighbor_otype, oid.get())
            # choice: decide action based on direct support
            global_support, direct_support, x, merge_map = self.__compute_global_support(obj, new_obj)
            direct_support = self.__compute_pairwise_support(obj, new_obj)
            rnd = random.random()
            if rnd > direct_support:
                obj.close_type(neighbor_otype)
                continue
            local_candidate = (new_obj, global_support, merge_map)
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
            found = False
            for open_neighbor in open_neighbors:
                global_support, direct_left_support, direct_right_support, merge_map = self.__compute_global_support(obj, open_neighbor)
                supported_objs[neighbor_otype].append((open_neighbor, global_support, merge_map))
                rnd = random.random()
                if rnd < global_support:
                    supported_objs[neighbor_otype] = [(open_neighbor, global_support, merge_map)]
                    found = True
                    break
                else:
                    fallback_candidates[neighbor_otype].append((open_neighbor, global_support, merge_map))
            if not found:
                supported_objs[neighbor_otype].append(local_candidate)
                fallback_candidates[neighbor_otype].append(local_candidate)
            if not sum(list(map(lambda x: x[1], supported_objs[neighbor_otype]))) > 0:
                if direct_support > 10.99:
                    # enforce (contradicting supports, so prioritize local support)
                    enforced_candidates = fallback_candidates[neighbor_otype]
                    supported_objs[neighbor_otype] = [(obj, 1, merge_map)
                                                      for (obj, p, merge_map) in enforced_candidates]
                    self.enforcements = self.enforcements + 1
                else:
                    obj.close_type(neighbor_otype)
                    continue
            merge_maps = {
                o: mm for (o, p, mm) in supported_objs[neighbor_otype]
            }
            probs = {o: p for (o, p, mm) in supported_objs[neighbor_otype]}
            cum_dist = CumulativeDistribution(probs)
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
            print(str(obj.oid) + " - " + predicted_otype + " " + str(selected_neighbor.oid))
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
        # extend to left side: yes, right side: yes
        paths[level] = [(path, True, True, cut_index)]
        left_border_object: ObjectInstance
        right_border_object: ObjectInstance
        global_support = 1.0
        direct_left_support = 1.0
        direct_right_support = 1.0
        merge_map = dict()
        while True:
            # evaluate current level supports
            for path, extend_left, extend_right, cut_index in paths[level]:
                # extend left: merge all left border objects with everything on the right side and vice versa
                # extend right: merge all right border objects with everything on the left side and vice versa
                path_objects = level_objs[level][path]
                left_border_objects = path_objects[0]
                right_border_objects = path_objects[-1]
                left_path_side = path[:cut_index]
                left_side_objects = path_objects[:cut_index]
                right_path_side = path[cut_index:]
                right_side_objects = path_objects[cut_index:]
                left_path_side_reverse = list(left_path_side[:])
                left_path_side_reverse.reverse()
                left_path_side_reversed = tuple(left_path_side_reverse)
                right_path_side_reverse = list(right_path_side[:])
                right_path_side_reverse.reverse()
                right_path_side_reversed = tuple(right_path_side_reverse)
                left_side_objects_reversed = left_side_objects[:]
                left_side_objects_reversed.reverse()
                right_side_objects_reversed = right_side_objects[:]
                right_side_objects_reversed.reverse()
                if extend_left:
                    for left_border_object in left_border_objects:
                        self.__update_merge_map(
                            merge_map, left_border_object, right_side_objects, left_path_side, right_path_side)
                        if level > ObjectInstance.executionModelDepth:
                            continue
                        if level > 1 and not (path == ('LEAD_Create Purchase Requisition', 'MATERIAL', 'LEAD_Receive Goods')
                                    or path == ('LEAD_Receive Goods', 'MATERIAL', 'LEAD_Create Purchase Requisition')):
                            support = 1
                        else:
                            support = self.__compute_element_support(
                                left_border_object, right_side_objects, left_path_side, right_path_side)
                        global_support = min(global_support, support)
                        if left_border_object == left_object:
                            direct_left_support = min(direct_left_support, support)
                    if level > ObjectInstance.executionModelDepth:
                        continue
                    support = self.__compute_extension_side_vs_other_side_elements_support(
                           left_border_objects, right_side_objects, left_path_side, right_path_side)
                    global_support = min(global_support, support)
                if extend_right:
                    for right_border_object in right_border_objects:
                        self.__update_merge_map(
                            merge_map, right_border_object, left_side_objects_reversed, right_path_side_reversed,
                            left_path_side_reversed)
                        if level > ObjectInstance.executionModelDepth:
                            continue
                        support = self.__compute_element_support( right_border_object, left_side_objects_reversed,
                                                                  right_path_side_reversed, left_path_side_reversed)
                        if right_border_object == right_object:
                            direct_right_support = min(direct_right_support, support)
                        global_support = min(global_support, support)
                    if level > ObjectInstance.executionModelDepth:
                        continue
                    support = self.__compute_extension_side_vs_other_side_elements_support( right_border_objects,
                        left_side_objects_reversed, right_path_side_reversed,left_path_side_reversed)
                    global_support = min(global_support, support)
            if level == ObjectInstance.executionModelEvaluationDepth:
                break
            # new step
            new_paths = []
            level_objs[level + 1] = dict()
            for path, extend_left, extend_right, cut_index in paths.get(level):
                path_objects = level_objs[level][path]
                left_border_type = path[0]
                right_border_type = path[-1]
                left_border_objects = path_objects[0]
                right_border_objects = path_objects[-1]
                if left_border_objects:
                    left_extensions = self.objectTypeGraph.get_neighbor_otypes(left_border_type)
                    for left_extension_type in left_extensions:
                        new_path = tuple([left_extension_type] + list(path))
                        new_paths.append((new_path, True, False, cut_index + 1))
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
                        new_paths.append((new_path, False, True, cut_index))
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
        return global_support, direct_left_support, direct_right_support, merge_map

    # o1: a, [[x1,x2],[x3,x4,x5]], [a,b,c], [d,e]
    def __compute_element_support(self, obj: ObjectInstance, other_side_objects, this_path_side, other_path_side):
        otype = obj.otype
        if otype in self.nonEmittingTypes:
            return 1.0
        element_support = 1
        # 0,d; 1,e
        for i, other_type in enumerate(other_path_side):
            depth = len(this_path_side) + i
            subpath = tuple(list(this_path_side) + list(other_path_side[:(i + 1)]))
            current_objs = other_side_objects[i]
            current_model = obj.global_model[depth][subpath]
            current_number_at_obj = len(current_model)
            additions = len([any_obj for any_obj in current_objs
                             if any_obj not in current_model])
            additions_support = 1
            for j in range(additions):
                additions_support = additions_support * obj.supportDistributions[subpath].get_support(
                    current_number_at_obj + j + 1)
            element_support = min(element_support, additions_support)
        return element_support

    # [x1,x2,x3], [[y1],[],[y2,y3,y4]] [a,b,c], [d,e,f]
    def __compute_extension_side_vs_other_side_elements_support(
            self, extension_side_objects, other_side_objects, extension_side_path, other_side_path):
        support = 1
        if not extension_side_objects:
            return support
        reversed_extension_side_path = list(extension_side_path[:])
        reversed_extension_side_path.reverse()
        for i, otype in enumerate(other_side_path):
            depth = len(extension_side_path) + i
            current_level_objects = other_side_objects[i]
            subpath = list(other_side_path[:(i + 1)])
            subpath.reverse()
            subpath = tuple(subpath + reversed_extension_side_path)
            for current_level_object in current_level_objects:
                current_model = current_level_object.global_model[depth][subpath]
                current_number_at_obj = len(current_model)
                additions = len([any_obj for any_obj in extension_side_objects
                                 if any_obj not in current_model])
                element_support = 1
                for j in range(additions):
                    element_support = element_support * current_level_object. \
                        supportDistributions[subpath].get_support(current_number_at_obj + j + 1)
                support = min(support, element_support)
        return support

    # {}, x1, [[y1],[],[y2,y3,y4]] [a,b,c], [d,e,f]
    def __update_merge_map(self, merge_map, obj, other_side_objects, this_path_side, other_path_side):
        for i, otype in enumerate(other_path_side):
            depth = len(this_path_side) + i
            subpath = tuple(list(this_path_side) + list(other_path_side[:(i + 1)]))
            reversed_subpath = list(subpath[:])
            reversed_subpath.reverse()
            reversed_subpath = tuple(reversed_subpath)
            current_objs = other_side_objects[i]
            additions = [any_obj for any_obj in current_objs
                         if any_obj not in obj.global_model[depth][subpath]]
            if obj not in merge_map:
                merge_map[obj] = dict()
            if subpath not in merge_map[obj]:
                merge_map[obj][subpath] = []
            merge_map[obj][subpath] = list(set(merge_map[obj][subpath] + additions))
            for addition in additions:
                if addition not in merge_map:
                    merge_map[addition] = dict()
                if reversed_subpath not in merge_map[addition]:
                    merge_map[addition][reversed_subpath] = []
                merge_map[addition][reversed_subpath] = list(set([obj] + merge_map[addition][reversed_subpath]))

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
        rel_card_stdev = math.sqrt(rel_cards.var()) if len(rel_cards) > 1 else 0.0
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
        stdev = math.sqrt(obj_arrival_rates.var()) if len(obj_arrival_rates) > 1 else 0.0
        return {
            "mean": mean,
            "stdev": stdev
        }
