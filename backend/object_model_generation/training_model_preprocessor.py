import os
import pickle

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
        df = ocel.get_extended_table()
        acts = self.activityLeadingTypes.keys()
        df = df[df["ocel:activity"].isin(acts)]
        self.df = df

    def build(self):
        self.__make_object_type_graph()
        self.__make_leading_type_process_executions()
        self.__make_schema_distributions()

    def save(self):
        for otype, obj_schemata in self.flatGlobalSchemata.items():
            for i, any_otype in enumerate(self.otypes):
                otype_relations = dict()
                for obj, schema in obj_schemata.items():
                    card = list(schema)[i]
                    if card not in otype_relations:
                        otype_relations[card] = 0
                    otype_relations[card] = otype_relations[card] + 1
                min_card = min(list(otype_relations.keys()))
                max_card = max(list(otype_relations.keys()))
                x_axis = range(min_card, max_card + 1)
                total = sum(otype_relations.values())
                log_based_schema_dist = list(map(lambda card: float(otype_relations[card]) / total
                if card in otype_relations else 0, x_axis))
                stats = {"log_based": log_based_schema_dist, "x_axis": x_axis}
                dist_path = os.path.join(self.sessionPath, otype + "_to_" + any_otype + "_schema_dist.pkl")
                with open(dist_path, "wb") as wf:
                    pickle.dump(stats, wf)
        with open(os.path.join(self.sessionPath, "training_model_preprocessor.pkl"), "wb") as wf:
            pickle.dump(self, wf)

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

    def __make_leading_type_process_executions(self):
        self.__make_object_model()
        object_model = self.totalObjectModel
        otypes = self.otypes
        leading_type_process_executions = {
            otype: dict()
            for otype in otypes
        }
        leading_type_global_schemata = {
            otype: dict()
            for otype in otypes
        }
        leading_type_local_schemata = {
            otype: dict()
            for otype in otypes
        }
        self.globalModelDepths = {
            otype: dict()
            for otype in otypes
        }
        total_object_model = self.totalObjectModel
        for otype in otypes:
            any_otypes = [
                any_otype
                for any_otype in otypes
                if not any_otype == otype
            ]
            for obj in object_model[otype]:
                execution, schema = self.__make_global_schema(otype, obj, any_otypes)
                leading_type_process_executions[otype][obj] = execution
                leading_type_global_schemata[otype][obj] = schema
                leading_type_local_schemata[otype][obj] = {
                    any_otype: len(total_object_model[otype][obj][any_otype])
                    for any_otype in self.otypes
                }
        self.globalObjectModel = leading_type_process_executions
        self.globalSchemata = leading_type_global_schemata
        self.localSchemata = leading_type_local_schemata

    def __make_schema_distributions(self):
        self.flat_local_schemata = {
            otype: {
                obj: tuple(
                    list(map(lambda any_otype: schema[any_otype] if any_otype in schema else 0, self.otypes))
                )
                for obj, schema in obj_schemata.items()
            }
            for otype, obj_schemata in self.localSchemata.items()
        }
        self.flatGlobalSchemata = {
            otype: {
                obj: tuple(
                    list(map(lambda any_otype: schema[any_otype] if any_otype in schema else 0, self.otypes))
                )
                for obj, schema in obj_schemata.items()
            }
            for otype, obj_schemata in self.globalSchemata.items()
        }
        self.schemaDistributions = {
            otype: {
                schema: len([v for v in (obj_flat_schemata.values()) if v == schema])
                for schema in set(obj_flat_schemata.values())
            }
            for otype, obj_flat_schemata in self.flatGlobalSchemata.items()
        }

    def __make_object_model(self):
        otypes = self.otypes
        direct_object_model = {otype: dict() for otype in otypes}
        reverse_object_model = {otype: dict() for otype in otypes}
        total_object_model = {otype: dict() for otype in otypes}
        self.df.apply(lambda row: self.__update_object_model(
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

    def __make_global_schema(self, otype, obj, any_otypes):
        object_model = self.totalObjectModel
        execution = {
            any_otype: set()
            for any_otype in self.otypes
        }
        current_model = object_model[otype][obj]
        min_path_lengths = {
            otype: 0
        }
        level = 1
        buffer = [
            (any_obj, any_otype, level)
            for any_otype in any_otypes
            for any_obj in current_model[any_otype]
        ]
        min_path_lengths.update({
            any_otype: level
            for (any_obj, any_otype, level) in buffer
        })
        while len(buffer) > 0:
            (any_obj, any_otype, lvl) = buffer[-1]
            buffer = buffer[:-1]
            execution[any_otype].add(any_obj)
            any_model = object_model[any_otype][any_obj]
            buffer_delta = []
            for any_next_otype in any_model:
                if any_next_otype == otype or len(any_model[any_next_otype]) == 0:
                    continue
                if any_next_otype not in min_path_lengths:
                    min_path_lengths[any_next_otype] = lvl + 1
                if min_path_lengths[any_next_otype] != lvl + 1:
                    continue
                for any_next_obj in any_model[any_next_otype]:
                    if any_next_obj not in execution[any_next_otype] \
                            and (any_next_obj, any_next_otype, lvl + 1) not in buffer:
                        buffer_delta.append((any_next_obj, any_next_otype, lvl + 1))
            buffer += buffer_delta
        schema = {
            any_otype: len(execution[any_otype])
            for any_otype in any_otypes
        }
        # TODO
        self.globalModelDepths[otype] = min_path_lengths
        return execution, schema
