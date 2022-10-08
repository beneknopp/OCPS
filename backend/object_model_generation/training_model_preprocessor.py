import os
import pickle

import pandas as pd

from object_model_generation.object_model_parameters import ObjectModelParameters
from object_model_generation.object_type_graph import ObjectTypeGraph


class TrainingModelPreprocessor:

    @classmethod
    def load(cls, session_path):
        training_model_preprocessor_path = os.path.join(session_path, "training_model_preprocessor.pkl")
        return pickle.load(open(training_model_preprocessor_path, "rb"))

    otypes: []
    activityLeadingTypes: []
    activitySelectedTypes: []
    objectTypeGraph: ObjectTypeGraph

    def __init__(self, session_path, ocel, object_model_parameters: ObjectModelParameters):
        self.sessionPath = session_path
        self.ocel = ocel
        self.otypes = object_model_parameters.otypes
        self.activityLeadingTypes = object_model_parameters.activityLeadingTypes
        self.activitySelectedTypes = object_model_parameters.activitySelectedTypes
        self.seedType = object_model_parameters.seedType
        self.executionModelDepth = object_model_parameters.executionModelDepth
        df = ocel.get_extended_table()
        acts = self.activityLeadingTypes.keys()
        df = df[df["ocel:activity"].isin(acts)]
        self.df = df

    def build(self):
        self.__make_object_type_graph()
        self.__make_process_executions()
        self.__make_schema_distributions()

    def save(self):
        # otype -> depth -> path -> cardinality -> frequency
        for otype, depths_with_paths in self.schemaDistributions.items():
            for depth, paths_with_cardfreqs in depths_with_paths.items():
                for path, card_freqs in paths_with_cardfreqs.items():
                    if len(path) < 1:
                        continue
                    card_freqs = list(card_freqs.items())
                    if len(card_freqs) < 1:
                        continue
                    print(path)
                    card_freqs = pd.DataFrame(card_freqs, columns=["cardinality", "frequency"])
                    # for card, freq in card_freqs.items():
                    min_card = min(card_freqs["cardinality"])
                    max_card = max(card_freqs["cardinality"])
                    x_axis = range(min_card, max_card + 1)
                    total = sum(card_freqs["frequency"])
                    log_based_schema_dist = list(map(
                        lambda card: sum(card_freqs[card_freqs["cardinality"] == card]["frequency"]) / total,
                        x_axis))
                    stats = {"log_based": log_based_schema_dist, "x_axis": x_axis}
                    dist_path = os.path.join(self.sessionPath, str(path) + "_schema_dist.pkl")
                    with open(dist_path, "wb") as wf:
                        pickle.dump(stats, wf)
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
        process_executions = {
            otype: dict()
            for otype in otypes
        }
        global_schemata = {
            otype: dict()
            for otype in otypes
        }
        local_schemata = {
            otype: dict()
            for otype in otypes
        }
        self.globalModelDepths = {
            otype: dict()
            for otype in otypes
        }
        process_executions = {otype: {depth: dict() for depth in range(self.executionModelDepth + 1)} for otype in
                              self.otypes}
        global_schemata = {otype: {depth: dict() for depth in range(self.executionModelDepth + 1)} for otype in
                           self.otypes}
        execution_model_paths = {otype: dict() for otype in self.otypes}
        for otype in self.otypes:
            current_depth = 0
            paths = [tuple([otype])]
            while current_depth <= self.executionModelDepth:
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
        while current_depth < self.executionModelDepth:
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
                schema_distributions[otype][depth] = dict()
                for path, obj_cards in paths_with_obj_cards.items():
                    schema_distributions[otype][depth][path] = dict()
                    for card in obj_cards.values():
                        if card not in schema_distributions[otype][depth][path]:
                            schema_distributions[otype][depth][path][card] = 0
                        schema_distributions[otype][depth][path][card] = schema_distributions[otype][depth][path][card] + 1
        self.schemaDistributions = schema_distributions

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
