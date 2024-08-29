import logging
import math
import random

import numpy as np
import pandas as pd
import pm4py
#from pyemd import emd

from object_model_generation.generator_parametrization import ParameterType, AttributeParameterization, ParameterMode
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
        self.generatorParametrization = training_model_preprocessor.generatorParametrization

    def generate(self):
        self.__initialize_object_instance_class()
        self.__run_generation()
        self.__assign_object_attributes()
        #self.__reindex_generated_objects()
        self.__make_generated_o2o()
        self.__make_arrival_times()

    def make_model_and_stats(self):
        session_path = self.sessionPath
        obj: ObjectInstance
        original_model = ObjectModel(session_path)
        generated_model = ObjectModel(session_path)
        schema_frequencies = {
            otype: dict()
            for otype in self.otypes
        }
        original_objs_dict = {}
        for otype, objs in self.trainingModelPreprocessor.objectModel.items():
            arrival_times = dict(self.trainingModelPreprocessor.arrivalTimes[otype])
            for oid in objs:
                obj_inst = ObjectInstance(otype, str(oid))
                time = round(float(arrival_times[oid]))
                obj_inst.time = time
                original_objs_dict[str(oid)] = obj_inst
        for otype, objs in self.trainingModelPreprocessor.objectModel.items():
            full_otype_model = {}
            for oid, adj_objs in objs.items():
                obj_inst = original_objs_dict[oid]
                all_adj_objs = []
                for any_otype, any_objs in adj_objs.items():
                    for any_obj in any_objs:
                        any_obj_inst = original_objs_dict[str(any_obj)]
                        obj_inst.objectModel[any_otype].add(any_obj_inst)
                        any_obj_inst.objectModel[otype].add(obj_inst)
                        all_adj_objs.append(any_obj_inst)
                full_otype_model[obj_inst] = all_adj_objs
            original_model.addModel( otype, full_otype_model)
        for otype, objs in self.generatedObjects.items():
            otype_model = {
                obj: [adj_obj
                      for adj_type in obj.objectModel
                      for adj_obj in obj.objectModel[adj_type]
                      ]
                for obj in objs
            }
            generated_model.addModel(otype, otype_model)
            obj: ObjectInstance
            for obj in objs:
                for depth in obj.global_model:
                    if depth == 0:
                        continue
                    for path, related_objs in obj.global_model[depth].items():
                        if str(path) not in schema_frequencies[otype]:
                            schema_frequencies[otype][str(path)] = {}
                        card = len(related_objs)
                        if card not in schema_frequencies[otype][str(path)]:
                            schema_frequencies[otype][str(path)][card] = 0
                        schema_frequencies[otype][str(path)][card] = schema_frequencies[otype][str(path)][card] + 1
        self.simulatedSchemaFrequencies = schema_frequencies
        self.originalModel = original_model
        self.generatedModel = generated_model
        for otype in self.otypes:
            card_params = self.generatorParametrization.get_parameters(otype, ParameterType.CARDINALITY.value)
            for path, card_param in card_params.items():
                simulated_data = self.simulatedSchemaFrequencies[otype][path]
                card_param : AttributeParameterization
                card_param.update_simulated_data(simulated_data)

    def save(self):
        self.originalModel.save_without_global_model(True)
        self.generatedModel.save_without_global_model(False)
        self.generatorParametrization.save(self.sessionPath)

    def __initialize_object_instance_class(self):
        # TODO: unify parameter-relevant fields / classes
        schema_distributions = {
            otype: otype_params[ParameterType.CARDINALITY]
            for otype, otype_params in self.generatorParametrization.parameters.items()
        }
        ObjectInstance.set_(
            otypes=self.otypes,
            execution_model_paths=self.trainingModelPreprocessor.executionModelPaths,
            execution_model_depth=self.trainingModelPreprocessor.executionModelDepth,
            execution_model_evaluation_depth=self.trainingModelPreprocessor.executionModelEvaluationDepth,
            schema_distributions=schema_distributions
        )

    def __run_generation(self):
        logging.info("Running Generation of Object Model...")
        object_model_parameters = self.objectModelParameters
        object_type_graph = self.objectTypeGraph
        seed_type = object_model_parameters.seedType
        self.__make_type_hierarchy(seed_type)
        number_of_objects = object_model_parameters.numberOfObjects
        self.nonEmittingTypes = object_model_parameters.nonEmittingTypes
        oid = RunningId()
        open_objects = {otype: [] for otype in self.otypes}
        closed_objects = {otype: [] for otype in self.otypes}
        total_objects = {otype: [] for otype in self.otypes}
        buffer = [InitialSeedMaker.create_obj(seed_type, oid, open_objects, total_objects)]
        self.enforcements = 0
        total_nof_objects = 1
        total_nof_closed_objects = 0
        while len(buffer) > 0:
            current_obj = buffer[0]
            buffer = buffer[1:]
            current_otype = current_obj.otype
            neighbor_types = object_type_graph.get_neighbors(current_otype)
            if current_otype not in self.nonEmittingTypes:
                prediction: ObjectLinkPrediction = self.__predict_neighbor(
                    current_obj, neighbor_types, open_objects, oid
                )
                if prediction.predict:
                    selected_neighbor = prediction.selected_neighbor
                    if prediction.mode == PredictionMode.NEW:
                        total_nof_objects += 1
                    predicted_type = prediction.predicted_type
                    merge_map = prediction.mergeMap
                    ObjectInstance.merge(selected_neighbor, current_obj, merge_map)
                    if selected_neighbor not in total_objects[predicted_type]:
                        total_objects[predicted_type].append(selected_neighbor)
                    if selected_neighbor not in buffer:
                        buffer = buffer + [selected_neighbor]
                        #buffer.insert(random.randrange(len(buffer) + 1), selected_neighbor)
                    #buffer.insert(random.randrange(len(buffer) + 1), current_obj)
                    buffer = [current_obj] + buffer
                else:
                    open_objects[current_otype] = list(
                        set([x for x in open_objects[current_otype] if not x == current_obj]))
                    closed_objects[current_otype].append(current_obj)
                    total_nof_closed_objects += 1
            if float(total_nof_closed_objects) / total_nof_objects > 0.995 and len(closed_objects[seed_type]) > number_of_objects:
                break
            if (len(buffer) == 0 and len(closed_objects[seed_type]) < number_of_objects):
                buffer = [InitialSeedMaker.create_obj(seed_type, oid, open_objects, total_objects)]
        self.generatedObjects     = total_objects
        self.generatedObjectsById = {}
        for ot_objs in total_objects.values():
            for obj in ot_objs:
                oid = obj.oid
                self.generatedObjectsById[oid] = obj


    def __evaluate_local_closure(self, obj_a, obj_b):
        ot_a = obj_a.otype
        ot_b = obj_b.otype
        # new_a = ObjectInstance(ot_a, 0)
        new_b = ObjectInstance(ot_b, 0)
        rnd = random.random()
        y, direct_support_a, x, mm = self.__compute_global_support(obj_a, new_b)
        # y, direct_support_b, x, mm = self.__compute_global_support(obj_b, new_a)
        if rnd > direct_support_a:
            obj_a.close_type(ot_b)
        # if direct_support_b == 0:
        # if rnd > direct_support_b:
        #   obj_b.close_type(ot_a)

    def __sort_buffer(self, buffer):
        obj: ObjectInstance
        buffer.sort(key=lambda obj: self.otypes.index(obj.otype))

    def __make_type_hierarchy(self, seed_type):
        otg = self.objectTypeGraph
        level = 0
        th = {seed_type: level}
        buffer =  [edge.source.name for node in otg.nodes for edge in node.incoming_edges if node.name == seed_type]
        buffer += [edge.target.name for node in otg.nodes for edge in node.outgoing_edges if node.name == seed_type]
        handled = {seed_type}
        while len(buffer):
            otypes = buffer[:]
            buffer = []
            level = level + 1
            for otype in otypes:
                handled.add(otype)
                th[otype] = level
                buffer += [edge.source.name for node in otg.nodes for edge in node.incoming_edges if node.name == otype]
                buffer += [edge.target.name for node in otg.nodes for edge in node.outgoing_edges if node.name == otype]
                buffer = [otype for otype in buffer if otype not in handled]
        self.typeHierarchy = th

    def __assign_object_attributes(self):
        obj: ObjectInstance
        for otype, objs in self.generatedObjects.items():
            attribute_parametrizations = self.generatorParametrization.get_parameters(otype, ParameterType.OBJECT_ATTRIBUTE.value)
            for attribute, attribute_parametrization in attribute_parametrizations.items():
                attribute_parametrization: AttributeParameterization
                for obj in objs:
                    value = attribute_parametrization.draw()
                    obj.assign_attribute(attribute, value)

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

    def __make_generated_o2o(self):
        o2o_src = []
        o2o_src_type = []
        o2o_trg = []
        o2o_trg_type = []
        for ot1, objs1 in self.generatedObjects.items():
            for o1 in objs1:
                o1: ObjectInstance
                for ot2, objs2 in o1.objectModel.items():
                    for o2 in objs2:
                        o2: ObjectInstance
                        o2o_src.append(o1.oid)
                        o2o_src_type.append(ot1)
                        o2o_trg.append(o2.oid)
                        o2o_trg_type.append(ot2)
        o2o = pd.DataFrame({
            "ocel_source_id": o2o_src,
            "ocel_source_type": o2o_src_type,
            "ocel_target_id": o2o_trg,
            "ocel_target_type": o2o_trg_type
        })
        self.o2o = o2o

    def __make_arrival_times(self):
        logging.info("Assigning Arrival Times...")
        seed_type = self.objectModelParameters.seedType
        seed_objects = list(self.generatedObjects[seed_type])
        # only for the seed type, all other arrive relative
        running_timestamp = 0
        buffer = list(seed_objects)
        all_objects = [obj for sl in self.generatedObjects.values() for obj in sl]
        handled_objs = set()
        current_obj: ObjectInstance
        related_obj: ObjectInstance
        seed_type_time_dist = self.trainingModelPreprocessor.generatorParametrization.get_parameters(
                    seed_type, ParameterType.TIMING.value, "Arrival Rates (independent)"
        )
        related_time_dists = {}
        # TODO: make this more safe (assumption that all relative arrival rate distributions for neighboring types exist)
        for otype in self.otypes:
            related_time_dists[otype] = {}
            time_dists = self.trainingModelPreprocessor.generatorParametrization.get_parameters(
                otype, ParameterType.TIMING.value)
            for any_type in self.otypes:
                attr = "Arrival Rates (relative to '" + any_type + "')"
                if attr in time_dists:
                    related_time_dists[otype][any_type] = time_dists[attr]
        while len(buffer) > 0:
            current_obj = buffer[0]
            buffer = buffer[1:]
            current_otype = current_obj.otype
            if current_otype == seed_type:
                current_obj.time = running_timestamp
                handled_objs.add(current_obj)
                time = round(seed_type_time_dist.draw())
                running_timestamp = running_timestamp + time
            open_local_model = list({
                obj for sl in current_obj.objectModel.values()
                for obj in sl
                if obj not in handled_objs
            })
            while len(open_local_model):
                related_obj = open_local_model[0]
                open_local_model = open_local_model[1:]
                related_type = related_obj.otype
                attr_par: AttributeParameterization = related_time_dists[related_type][current_otype]
                is_batch_arrival = attr_par.markedAsBatchArrival
                if related_type != seed_type:
                    relative_arrival_time = round(attr_par.draw())
                    if is_batch_arrival:
                        o2o = self.o2o
                        sibling_obj_ids = list(o2o[
                            (o2o["ocel_source_id"] == current_obj.oid) &
                            (o2o["ocel_target_type"] == related_type)
                        ]["ocel_target_id"].unique())
                        related_objs = [self.generatedObjectsById[oid] for oid in sibling_obj_ids]
                        open_local_model = [x for x in open_local_model if x.oid not in sibling_obj_ids]
                    else:
                        related_objs = [related_obj]
                    for related_obj_ in related_objs:
                        related_obj_.time = current_obj.time + relative_arrival_time
                        handled_objs.add(related_obj_)
                        if related_obj_ not in buffer:
                            buffer = buffer + [related_obj_]
        if len(handled_objs) != len(all_objects):
            raise ValueError("Not all objects have been assigned an arrival time")
        min_time = min(map(lambda obj: obj.time, handled_objs))
        for obj in handled_objs:
            obj.time = obj.time - min_time
        seed_objects.sort(key=lambda x: x.time)

    def __predict_neighbor(self, obj: ObjectInstance, neighbor_types, open_objects, rOID: RunningId):
        supported_objs = {}
        new_objs = []
        neighbor_types = [nt for nt in neighbor_types if not obj.locally_closed_types[nt]]
        random.shuffle(neighbor_types)
        for neighbor_otype in neighbor_types:
            if self.typeHierarchy[neighbor_otype] < self.typeHierarchy[obj.otype]:
                continue
            # try new instance for that otype
            supported_objs[neighbor_otype] = []
            new_obj = ObjectInstance(neighbor_otype, rOID.get())
            local_support, dls, rls, merge_map = self.__compute_global_support(obj, new_obj)
            max_support = local_support
            new_objs.append(new_obj)
            supported_objs[neighbor_otype].append((new_obj, local_support, merge_map))
            open_neighbors = open_objects[neighbor_otype]
            # avoid bias towards specific objects
            random.shuffle(open_objects[neighbor_otype])
            open_neighbors = list(filter(lambda on:
                # neighbor still open, but not connected to this object yet
                on not in obj.objectModel[neighbor_otype], open_neighbors)
            )
            open_neighbor: ObjectInstance
            for open_neighbor in open_neighbors:
                global_support, direct_left_support, direct_right_support, merge_map = self.__compute_global_support(
                    obj, open_neighbor)
                supported_objs[neighbor_otype].append((open_neighbor, global_support, merge_map))
                if global_support > max_support:
                    max_support = global_support
            rnd = random.random()
            direct_support = self.__compute_emit_support(obj, new_obj)
            if rnd > max_support and direct_support < 0.99:
                obj.close_type(neighbor_otype)
                continue
            if not sum(list(map(lambda x: x[1], supported_objs[neighbor_otype]))) > 0:
                obj.close_type(neighbor_otype)
                continue
            merge_maps = {o: mm for (o, supp, mm) in supported_objs[neighbor_otype]}
            probs = {o: supp for (o, supp, mm) in supported_objs[neighbor_otype]}
            cum_dist = CumulativeDistribution(probs)
            selected_neighbor = cum_dist.sample()
            merge_map = merge_maps[selected_neighbor]
            predicted_otype = selected_neighbor.otype
            mode = PredictionMode.NEW if selected_neighbor in new_objs else PredictionMode.APPEND
            if mode == PredictionMode.NEW:
                rOID.inc()
                open_objects[predicted_otype].append(selected_neighbor)
            prediction = ObjectLinkPrediction(
                predict=True, predicted_type=predicted_otype, mode=mode,
                selected_neighbor=selected_neighbor, merge_map=merge_map
            )
            # logging.info(f"{obj.otype} {str(obj.oid)}: {str(prediction.pretty_print()}"))
            print(obj.otype + " " + str(obj.oid) + ": " + str(prediction.pretty_print()))
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
        support_events = 0
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
                        else:
                            support, event_count = self.__compute_element_support(
                                left_border_object, right_side_objects, left_path_side, right_path_side)
                            support_events += event_count
                        global_support = min(global_support, support)
                        if left_border_object == left_object:
                            direct_left_support = min(direct_left_support, support)
                    if level > ObjectInstance.executionModelDepth:
                        continue
                    support, event_count = self.__compute_extension_side_vs_other_side_elements_support(
                        left_border_objects, right_side_objects, left_path_side, right_path_side)
                    support_events += event_count
                    global_support = min(global_support, support)
                if extend_right:
                    for right_border_object in right_border_objects:
                        self.__update_merge_map(
                            merge_map, right_border_object, left_side_objects_reversed, right_path_side_reversed,
                            left_path_side_reversed)
                        if level > ObjectInstance.executionModelDepth:
                            continue
                        support, event_count = self.__compute_element_support(right_border_object,
                                                                              left_side_objects_reversed,
                                                                              right_path_side_reversed,
                                                                              left_path_side_reversed)
                        support_events += event_count
                        if right_border_object == right_object:
                            direct_right_support = min(direct_right_support, support)
                        global_support = min(global_support, support)
                    if level > ObjectInstance.executionModelDepth:
                        continue
                    support, event_count = self.__compute_extension_side_vs_other_side_elements_support(
                        right_border_objects,
                        left_side_objects_reversed,
                        right_path_side_reversed,
                        left_path_side_reversed)
                    support_events += event_count
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
                    left_extensions = self.objectTypeGraph.get_neighbors(left_border_type)
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
                    right_extensions = self.objectTypeGraph.get_neighbors(right_border_type)
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
        if support_events == 0:
            raise ValueError("Undefined support for object connection")
        return global_support, direct_left_support, direct_right_support, merge_map


    def __compute_element_support(self, obj: ObjectInstance, other_side_objects, this_path_side, other_path_side):
        otype = obj.otype
        if otype in self.nonEmittingTypes:
            return 1.0
        element_support = 1
        event_count = 0
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
                if str(subpath) not in obj.supportDistributions:
                    continue
                event_count += 1
                additions_support = additions_support * obj.supportDistributions[str(subpath)].get_support(
                    current_number_at_obj + j + 1)
            element_support = min(element_support, additions_support)
        return element_support, event_count


    def __compute_extension_side_vs_other_side_elements_support(
            self, extension_side_objects, other_side_objects, extension_side_path, other_side_path):
        support = 1
        support_events = 0
        if not extension_side_objects:
            return support, support_events
        reversed_extension_side_path = list(extension_side_path[:])
        reversed_extension_side_path.reverse()
        for i, otype in enumerate(other_side_path):
            depth = len(extension_side_path) + i
            current_level_objects = other_side_objects[i]
            subpath = list(other_side_path[:(i + 1)])
            subpath.reverse()
            subpath = (tuple(subpath + reversed_extension_side_path))
            for current_level_object in current_level_objects:
                if str(subpath) not in current_level_object.supportDistributions:
                    continue
                support_events += 1
                current_model = current_level_object.global_model[depth][subpath]
                current_number_at_obj = len(current_model)
                additions = len([any_obj for any_obj in extension_side_objects
                                 if any_obj not in current_model])
                element_support = 1
                for j in range(additions):
                    element_support = element_support * current_level_object. \
                        supportDistributions[str(subpath)].get_support(current_number_at_obj + j + 1)
                support = min(support, element_support)
        return support, support_events


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


    def __compute_emit_support(self, existing_obj: ObjectInstance, new_obj: ObjectInstance):
        ot1 = existing_obj.otype
        ot2 = new_obj.otype
        depth = 1
        if existing_obj in new_obj.global_model[depth][tuple([ot2, ot1])]:
            return 1.0
        support_for_new_at_old = 1.0
        nof_new_at_old = len(existing_obj.global_model[depth][tuple([ot1, ot2])])
        if str(tuple([ot1, ot2])) in ObjectInstance.supportDistributions[ot1]:
            support_for_new_at_old = ObjectInstance.supportDistributions[ot1][str(tuple([ot1, ot2]))].get_support(
                nof_new_at_old + 1)
        support_for_old_at_new = 1.0
        nof_old_at_new = len(new_obj.global_model[depth][tuple([ot2, ot1])])
        if str(tuple([ot2, ot1])) in ObjectInstance.supportDistributions[ot2]:
            support_for_old_at_new = ObjectInstance.supportDistributions[ot2][str(tuple([ot2, ot1]))].get_support(
                nof_old_at_new + 1)
        return (min(support_for_old_at_new, support_for_new_at_old))

    def __get_distance_matrix(self, n):
        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(float(abs(i-j)))
            matrix.append(row)
        return np.array(matrix)


    def __get_object_graph_emc(self, depth):
        avg_emc = 0
        total = 0
        for otype in self.otypes:
            attr_params = self.generatorParametrization.parameters[otype][ParameterType.CARDINALITY].items()
            for key, attr_param in attr_params:
                if not (len(key.split("', '")) == depth + 1):
                    continue
                total += 1
                attr_param: AttributeParameterization
                log_based = attr_param.yAxes[ParameterMode.LOG_BASED]
                sim = attr_param.yAxes[ParameterMode.SIMULATED]
                n = len(log_based)
                distance_matrix = self.__get_distance_matrix(n)
                emc = emd(np.array(log_based), np.array(sim), distance_matrix)
                avg_emc += emc
        if total == 0:
            return 0
        return avg_emc / total

    def get_response(self):
        response_dict = dict()
        response_dict["numberOfObjects"] = {}
        for ot in self.otypes:
            generated_objects = self.generatedObjects[ot]
            response_dict["numberOfObjects"][ot] = len(generated_objects)
        #response_dict["earthMoversConformance"] = {}
        # TODO
        #for depth in range(1, 4):
            #object_graph_emc = self.__get_object_graph_emc(depth)
            #response_dict["earthMoversConformance"][depth] = object_graph_emc
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

    @classmethod
    def name(cls, session_path, name):
        generated_object_model: ObjectModel = ObjectModel.load(session_path, use_original=False, object_model_name="")
        generated_object_model.save(use_original=False, name=name)