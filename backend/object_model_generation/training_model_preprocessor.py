import os
import pickle

import pm4py

from object_model_generation.generator_parametrization import GeneratorParametrization, ParameterType
from object_model_generation.object_model import ObjectModel
from object_model_generation.object_model_parameters import ObjectModelParameters
from object_model_generation.object_type_graph import ObjectTypeGraph


class TrainingModelPreprocessor:

    @classmethod
    def load(cls, session_path):
        training_model_preprocessor_path = os.path.join(session_path, "training_model_preprocessor.pkl")
        training_model_preprocessor : TrainingModelPreprocessor = pickle.load(open(training_model_preprocessor_path, "rb"))
        generator_parametrization : GeneratorParametrization = GeneratorParametrization.load(session_path)
        training_model_preprocessor.generatorParametrization = generator_parametrization
        return training_model_preprocessor

    @classmethod
    def load_attribute_names(cls, session_path):
        attribute_names_path = os.path.join(session_path, "attribute_names.pkl")
        return pickle.load(open(attribute_names_path, "rb"))

    otypes: []
    objectModel: dict
    activityLeadingTypes: []
    activitySelectedTypes: []
    objectTypeGraph: ObjectTypeGraph
    generatorParametrization: GeneratorParametrization
    attributeNames: dict
    objectAttributeValueDistributions: dict
    cardinalityDistributions: dict

    def __init__(self, session_path, ocel, object_model_parameters: ObjectModelParameters):
        self.sessionPath = session_path
        self.ocel = ocel
        self.otypes = object_model_parameters.otypes
        self.activityLeadingTypes = object_model_parameters.activityLeadingTypes
        self.activitySelectedTypes = object_model_parameters.activitySelectedTypes
        #self.seedType = object_model_parameters.seedType
        self.executionModelDepth = object_model_parameters.executionModelDepth
        self.executionModelEvaluationDepth = object_model_parameters.executionModelEvaluationDepth
        df = ocel.get_extended_table()
        acts = self.activityLeadingTypes.keys()
        df = df[df["ocel:activity"].isin(acts)]
        self.df = df

    def build(self):
        self.__make_object_type_graph()
        self.__make_process_executions()
        self.__make_schema_distributions()
        self.__make_object_attribute_value_distributions()
        self.__make_timing_information()
        self.__make_attribute_names()
        self.__initialize_generator_parametrization()

    def save(self):
        self.generatorParametrization.save(self.sessionPath)
        with open(os.path.join(self.sessionPath, "attribute_names.pkl"), "wb") as wf:
            pickle.dump(self.attributeNames, wf)
        with open(os.path.join(self.sessionPath, "training_model_preprocessor.pkl"), "wb") as wf:
            pickle.dump(self, wf)
        with open(os.path.join(self.sessionPath, "object_type_graph.pkl"), "wb") as wf:
            pickle.dump(self.objectTypeGraph, wf)

    def __make_object_type_graph(self):
        object_type_graph = ObjectTypeGraph(self.otypes)
        object_type_graph.add_nodes_by_names(self.otypes)
        otg_edges = {}
        for act in self.activitySelectedTypes:
            leading_type = self.activityLeadingTypes[act]
            other_selected_types = [type for type in self.activitySelectedTypes[act] if not type == leading_type]
            if leading_type not in otg_edges:
                otg_edges[leading_type] = set()
            otg_edges[leading_type].update(other_selected_types)
        for leading_type, other_types in otg_edges.items():
            for other_type in other_types:
                object_type_graph.add_edge_by_names(leading_type, other_type)
        object_type_graph.make_info()
        self.objectTypeGraph = object_type_graph

    def __make_process_executions(self):
        self.__make_object_model()
        object_model = self.objectModel
        otypes = self.otypes
        local_schemata = {
            otype: dict()
            for otype in otypes
        }
        self.globalModelDepths = {
            otype: dict()
            for otype in otypes
        }
        process_executions = {
            otype: {
                depth: dict()
                for depth in range(self.executionModelEvaluationDepth + 1)
            }
            for otype in self.otypes
        }
        global_schemata = {
            otype: {
                depth: dict()
                for depth in range(self.executionModelEvaluationDepth + 1)
            }
            for otype in self.otypes
        }
        execution_model_paths = {otype: dict() for otype in self.otypes}
        for otype in self.otypes:
            current_depth = 0
            paths = [tuple([otype])]
            while current_depth <= self.executionModelEvaluationDepth:
                execution_model_paths[otype][current_depth] = paths[:]
                last_paths = paths[:]
                paths = []
                for path in last_paths:
                    process_executions[otype][current_depth][path] = dict()
                    global_schemata[otype][current_depth][path] = dict()
                    last_otype = path[-1]
                    next_otypes = self.objectTypeGraph.get_neighbors(last_otype)
                    for next_otype in next_otypes:
                        paths.append(tuple(list(path) + [next_otype]))
                current_depth = current_depth + 1
        self.executionModelPaths = execution_model_paths
        for otype, obj_models in object_model.items():
            for obj, obj_model in obj_models.items():
                self.__make_global_schema(otype, obj, process_executions, global_schemata)
                local_schemata[otype][obj] = {
                    any_otype: len(object_model[otype][obj][any_otype])
                    for any_otype in self.otypes
                }
        self.globalObjectModel = process_executions
        self.globalSchemata = global_schemata
        self.localSchemata = local_schemata

    def __make_global_schema(self, otype, obj, process_executions, global_schemata):
        object_model = self.objectModel
        # execution: depth -> otype-path -> set of objects
        execution = dict()
        schema = dict()
        current_depth = 0
        path = tuple([otype])
        execution[current_depth] = {path: [obj]}
        schema[current_depth] = {path: 1}
        process_executions[otype][current_depth][path][obj] = [obj]
        global_schemata[otype][current_depth][path][obj] = 1
        while current_depth < self.executionModelEvaluationDepth:
            current_model = execution[current_depth]
            execution[current_depth + 1] = dict()
            schema[current_depth + 1] = dict()
            for path, model_objs in current_model.items():
                last_otype = path[-1]
                next_otypes = self.objectTypeGraph.get_neighbors(last_otype)
                for next_otype in next_otypes:
                    new_execution_model = [object_model[last_otype][model_obj][next_otype]
                                           for model_obj in model_objs]
                    new_execution_model = [model_obj for sl in new_execution_model
                                           for model_obj in sl]
                    new_execution_model = list(set(new_execution_model))
                    new_path = tuple(list(path) + [next_otype])
                    execution[current_depth + 1][new_path] = new_execution_model
                    schema[current_depth + 1][new_path] = len(new_execution_model)
                    process_executions[otype][current_depth + 1][new_path][obj] = new_execution_model
                    global_schemata[otype][current_depth + 1][new_path][obj] = len(new_execution_model)
            current_depth = current_depth + 1

    def __make_schema_distributions(self):
        self.flatLocalSchemata = {
            otype: {
                obj: tuple(
                    list(map(lambda any_otype: schema[any_otype] if any_otype in schema else 0, self.otypes))
                )
                for obj, schema in obj_schemata.items()
            }
            for otype, obj_schemata in self.localSchemata.items()
        }
        # otype -> depth -> path -> cardinality -> frequency
        schema_distributions = {otype: dict() for otype in self.otypes}
        # otype -> depth -> path -> obj -> cardinality
        for otype, depths_with_paths in self.globalSchemata.items():
            schema_distributions[otype] = dict()
            for depth, paths_with_obj_cards in depths_with_paths.items():
                if depth < 1:
                    continue
                for path, obj_cards in paths_with_obj_cards.items():
                    path_key = str(path)
                    schema_distributions[otype][path_key] = []
                    for card in obj_cards.values():
                        schema_distributions[otype][path_key].append(card)
        self.cardinalityDistributions = schema_distributions

    def __make_object_attribute_value_distributions(self):
        self.objectAttributeValueDistributions = dict()
        objects_df = self.ocel.objects
        attribute_names = [col for col in objects_df.columns if not col.startswith("ocel:")]
        oav_dists = {}
        for otype in self.otypes:
            oav_dists[otype] = {}
            for attr in attribute_names:
                otype_attr_support = objects_df[objects_df["ocel:type"] == otype][objects_df[attr].notnull()]
                if len(otype_attr_support) == 0:
                    continue
                oav_dists[otype][attr] = []
                for value in otype_attr_support[attr].values:
                    value_supp = otype_attr_support[otype_attr_support[attr] == value]
                    oav_dists[otype][attr] += len(value_supp)*[value]
        self.objectAttributeValueDistributions = oav_dists

    def __make_timing_information(self):
        self.timingDistributions = {
            otype: dict()
            for otype in self.otypes
        }
        self.__make_prior_arrival_times_distributions()
        self.__make_relative_arrival_times_distributions()

    def __make_prior_arrival_times_distributions(self):
        self.flattenedLogs = dict()
        self.arrivalTimes = {}
        for otype in self.otypes:
            flattened_log = pm4py.ocel_flattening(self.ocel, otype)
            flattened_log = flattened_log.sort_values(["time:timestamp"])
            self.flattenedLogs[otype] = flattened_log
            arrival_times = flattened_log.groupby("case:concept:name").first()["time:timestamp"]
            arrival_times = arrival_times.sort_values()
            arrival_times = arrival_times.apply(lambda row: row.timestamp())
            self.arrivalTimes[otype] = arrival_times
            arrival_rates = arrival_times.diff()[1:].values
            attr = "Arrival Rates (independent)"
            self.timingDistributions[otype][attr] = list(arrival_rates)

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
        global_model = self.globalObjectModel
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
                    attr = "Arrival Rates (relative to '" + otype + "')"
                    self.timingDistributions[any_otype][attr] = list(log_based_rel_times)
        self.relativeArrivalTimesDistributions = dists
        self.logBasedRelativeArrivalTimes = log_based_relative_arrival_times

    def __make_attribute_names(self):
        attributeNames = dict()
        attributeNames[ParameterType.CARDINALITY] = {}
        for otype, path_dict in self.cardinalityDistributions.items():
            attributeNames[ParameterType.CARDINALITY][otype] = []
            for path in path_dict.keys():
                attributeNames[ParameterType.CARDINALITY][otype].append(str(path))
        attributeNames[ParameterType.OBJECT_ATTRIBUTE] = {
            otype: list(self.objectAttributeValueDistributions[otype].keys())
            for otype in self.otypes
        }
        attributeNames[ParameterType.TIMING] = {
            otype: ["absoluteArrival"] for otype in self.otypes
        }
        self.attributeNames = attributeNames

    def __make_object_model(self):
        otypes = self.otypes
        object_model = {otype: dict() for otype in otypes}
        self.df.apply(
            lambda row: self.__update_object_model(
                row, object_model, otypes
            ), axis=1)
        self.objectModel = object_model

    def __update_object_model(self, row, object_model, otypes):
        event_objects = []
        activity = row["ocel:activity"]
        leading_type = self.activityLeadingTypes[activity]
        leading_objects = row["ocel:type:" + leading_type]
        if not isinstance(leading_objects, list):
            # TODO
            return
        if not len(leading_objects) == 1:
            raise ValueError("Event does not have exactly one object of specified leading type")
        leading_object = leading_objects[0]
        if leading_object not in object_model[leading_type]:
            object_model[leading_type][leading_object] = {
                any_otype: set()
                for any_otype in otypes
            }
        for otype in otypes:
            if otype == leading_type:
                continue
            object_col = "ocel:type:" + otype
            otype_objects = row[object_col]
            if isinstance(otype_objects, list):
                event_objects += [(oid, otype) for oid in otype_objects]
        for i, (event_object, otype) in enumerate(event_objects):
            object_model[leading_type][leading_object][otype].add(event_object)
            if event_object not in object_model[otype]:
                object_model[otype][event_object] = {
                    any_otype: set()
                    for any_otype in otypes
                }
            object_model[otype][event_object][leading_type].add(leading_object)

    def __initialize_generator_parametrization(self):
        generator_parametrization = GeneratorParametrization(
            self.otypes, self.cardinalityDistributions, self.objectAttributeValueDistributions, self.timingDistributions
        )
        self.generatorParametrization = generator_parametrization

