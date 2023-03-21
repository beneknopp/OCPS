import os
import pickle

import pandas as pd

from object_model_generation.object_instance import ObjectInstance
from object_model_generation.object_model import ObjectModel
from object_model_generation.object_model_parameters import ObjectModelParameters
from object_model_generation.object_type_graph import ObjectTypeGraph
from object_model_generation.stats_mode import StatsMode


class TrainingModelPreprocessor:

    @classmethod
    def load(cls, session_path):
        training_model_preprocessor_path = os.path.join(session_path, "training_model_preprocessor.pkl")
        return pickle.load(open(training_model_preprocessor_path, "rb"))

    @classmethod
    def load_attribute_names(cls, session_path):
        attribute_names_path = os.path.join(session_path, "attribute_names.pkl")
        return pickle.load(open(attribute_names_path, "rb"))

    @classmethod
    def load_object_attribute_value_distributions(cls, session_path, mode: StatsMode = StatsMode.LOG_BASED):
        object_attribute_value_distributions_filename = "object_attribute_value_distributions_" + str(mode.value) + ".pkl"
        object_attribute_value_distributions_path = os.path.join(session_path, object_attribute_value_distributions_filename)
        if not os.path.isfile(object_attribute_value_distributions_path):
            return None
        return pickle.load(open(object_attribute_value_distributions_path, "rb"))

    @classmethod
    def load_schema_distributions(cls, session_path, mode: StatsMode = StatsMode.LOG_BASED):
        schema_distributions_filename = "schema_distributions_" + str(mode.value) + ".pkl"
        schema_distributions_path = os.path.join(session_path, schema_distributions_filename)
        if not os.path.isfile(schema_distributions_path):
            return None
        return pickle.load(open(schema_distributions_path, "rb"))

    otypes: []
    activityLeadingTypes: []
    activitySelectedTypes: []
    objectTypeGraph: ObjectTypeGraph
    attributeNames: dict
    schemaDistributions: dict

    def __init__(self, session_path, ocel, object_model_parameters: ObjectModelParameters):
        self.sessionPath = session_path
        self.ocel = ocel
        self.otypes = object_model_parameters.otypes
        self.activityLeadingTypes = object_model_parameters.activityLeadingTypes
        self.activitySelectedTypes = object_model_parameters.activitySelectedTypes
        self.seedType = object_model_parameters.seedType
        self.executionModelDepth = object_model_parameters.executionModelDepth
        self.executionModelEvaluationDepth = object_model_parameters.executionModelEvaluationDepth
        df = ocel.get_extended_table()
        acts = self.activityLeadingTypes.keys()
        df = df[df["ocel:activity"].isin(acts)]
        self.df = df

    def build(self):
        self.__make_object_type_graph()
        self.__make_process_executions()
        # TODO: factor out
        self.__make_schema_distributions()
        self.__make_object_attribute_value_distributions()
        self.__make_attribute_names()

    def save(self):
        mode = StatsMode.LOG_BASED
        with open(os.path.join(self.sessionPath, "attribute_names.pkl"), "wb") as wf:
            pickle.dump(self.attributeNames, wf)
        with open(os.path.join(self.sessionPath, "object_attribute_value_distributions_"+str(mode.value)+".pkl"), "wb") as wf:
            pickle.dump(self.objectAttributeValueDistributions, wf)
        with open(os.path.join(self.sessionPath, "schema_distributions_"+str(mode.value)+".pkl"), "wb") as wf:
            pickle.dump(self.schemaDistributions, wf)
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
        object_model = self.totalObjectModel
        otypes = self.otypes
        local_schemata = {
            otype: dict()
            for otype in otypes
        }
        self.globalModelDepths = {
            otype: dict()
            for otype in otypes
        }
        process_executions = {otype: {depth: dict() for depth in range(self.executionModelEvaluationDepth + 1)} for otype in
                              self.otypes}
        global_schemata = {otype: {depth: dict() for depth in range(self.executionModelEvaluationDepth + 1)} for otype in
                           self.otypes}
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
                    next_otypes = self.objectTypeGraph.get_neighbor_otypes(last_otype)
                    for next_otype in next_otypes:
                        paths.append(tuple(list(path) + [next_otype]))
                current_depth = current_depth + 1
        self.executionModelPaths = execution_model_paths
        total_object_model = self.totalObjectModel
        for otype, obj_models in object_model.items():
            for obj, obj_model in obj_models.items():
                self.__make_global_schema(otype, obj, process_executions, global_schemata)
                local_schemata[otype][obj] = {
                    any_otype: len(total_object_model[otype][obj][any_otype])
                    for any_otype in self.otypes
                }
        self.globalObjectModel = process_executions
        self.globalSchemata = global_schemata
        self.localSchemata = local_schemata

    def __make_global_schema(self, otype, obj, process_executions, global_schemata):
        total_object_model = self.totalObjectModel
        # execution: depth -> otype-path -> set of objects
        execution = dict()
        schema = dict()
        current_model = total_object_model[otype][obj]
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
                next_otypes = self.objectTypeGraph.get_neighbor_otypes(last_otype)
                for next_otype in next_otypes:
                    new_execution_model = [total_object_model[last_otype][model_obj][next_otype]
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

    def __save_original_marking(self):
        original_model = ObjectModel(self.sessionPath)
        original_objects = {}


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
                    schema_distributions[otype][path_key] = dict()
                    for card in obj_cards.values():
                        if card not in schema_distributions[otype][path_key]:
                            schema_distributions[otype][path_key][card] = 0
                        schema_distributions[otype][path_key][card] = schema_distributions[otype][path_key][card] + 1
        self.schemaDistributions = schema_distributions

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
                oav_dists[otype][attr] = {}
                for value in otype_attr_support[attr].values:
                    value_supp = otype_attr_support[otype_attr_support[attr] == value]
                    oav_dists[otype][attr][value] = len(value_supp)
        self.objectAttributeValueDistributions = oav_dists

    def __make_attribute_names(self):
        attributeNames = dict()
        attributeNames["cardinality"] = {}
        for otype, path_dict in self.schemaDistributions.items():
            attributeNames["cardinality"][otype] = []
            for path in path_dict.keys():
                attributeNames["cardinality"][otype].append(str(path))
        attributeNames["objectAttribute"] = {
            otype: list(self.objectAttributeValueDistributions[otype].keys())
            for otype in self.otypes
        }
        attributeNames["timing"] = {
            otype: ["absoluteArrival"] for otype in self.otypes
        }
        self.attributeNames = attributeNames

    def __make_object_model(self):
        otypes = self.otypes
        direct_object_model = {otype: dict() for otype in otypes}
        reverse_object_model = {otype: dict() for otype in otypes}
        total_object_model = {otype: dict() for otype in otypes}
        self.df.apply(
            lambda row: self.__update_object_model(
                row, direct_object_model, reverse_object_model, total_object_model, otypes
            ), axis=1)
        self.directObjectModel = direct_object_model
        self.reverseObjectModel = reverse_object_model
        self.totalObjectModel = total_object_model

    def __update_object_model(self, row, direct_object_model, reverse_object_model, total_object_model, otypes):
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
        if leading_object not in direct_object_model[leading_type]:
            direct_object_model[leading_type][leading_object] = {
                any_otype: set()
                for any_otype in otypes
            }
        if leading_object not in reverse_object_model[leading_type]:
            reverse_object_model[leading_type][leading_object] = {
                any_otype: set()
                for any_otype in otypes
            }
        if leading_object not in total_object_model[leading_type]:
            total_object_model[leading_type][leading_object] = {
                any_otype: set()
                for any_otype in otypes
            }
        for otype in [otype for otype in otypes if not otype == leading_type]:
            object_col = "ocel:type:" + otype
            otype_objects = row[object_col]
            if isinstance(otype_objects, list):
                event_objects += [(oid, otype) for oid in otype_objects]
        for i, (event_object, otype) in enumerate(event_objects):
            direct_object_model[leading_type][leading_object][otype].add(event_object)
            total_object_model[leading_type][leading_object][otype].add(event_object)
            if event_object not in direct_object_model[otype]:
                direct_object_model[otype][event_object] = {
                    any_otype: set()
                    for any_otype in otypes
                }
            if event_object not in reverse_object_model[otype]:
                reverse_object_model[otype][event_object] = {
                    any_otype: set()
                    for any_otype in otypes
                }
            if event_object not in total_object_model[otype]:
                total_object_model[otype][event_object] = {
                    any_otype: set()
                    for any_otype in otypes
                }
            reverse_object_model[otype][event_object][leading_type].add(leading_object)
            total_object_model[otype][event_object][leading_type].add(leading_object)

    @classmethod
    def make_chart_data(cls, labels, log_dist, modeled_dist = None, sim_dist = None):
        stats = {}
        # dict from attr name to values to freq
        for label in labels:
            stats[label] = {}
            val_freqs = [
                ("log_based",
                 pd.DataFrame(list(log_dist[label].items()), columns=["value", "frequency"]))
            ]
            if modeled_dist is not None:
                val_freqs.append((
                    "modeled",
                    pd.DataFrame(list(modeled_dist[label].items()), columns=["value", "frequency"])
                ))
            if sim_dist is not None:
                val_freqs.append((
                    "simulated",
                    pd.DataFrame(list(sim_dist[label].items()), columns=["value", "frequency"])
                ))
            # TODO
            glob_min_val = 10000000
            glob_max_val = -100000000
            for mode, val_freq_dict in val_freqs:
                min_val = min(val_freq_dict["value"])
                max_val = max(val_freq_dict["value"])
                glob_min_val = min(glob_min_val, min_val)
                glob_max_val = max(glob_max_val, max_val)
            x_axis = range(glob_min_val, glob_max_val + 1)
            for mode, val_freq_dict in val_freqs:
                total = sum(val_freq_dict["frequency"])
                chart_data = list(map(
                    lambda card: sum(val_freq_dict[val_freq_dict["value"] == card]["frequency"]) / total,
                    x_axis))
                stats[label][mode] = chart_data
            x_axis = [str(i) for i in range(glob_min_val, glob_max_val + 1)]
            stats[label]["x_axis"] = x_axis
        return stats

