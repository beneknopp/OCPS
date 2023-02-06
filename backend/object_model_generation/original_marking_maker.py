import os
import pickle

from input_ocel_processing.process_config import ProcessConfig
from object_model_generation.object_instance import ObjectInstance
from object_model_generation.object_model import ObjectModel
from object_model_generation.object_type_graph import ObjectTypeGraph


class OriginalMarkingMaker():

    def __init__(self, ocel, process_config: ProcessConfig):
        self.ocel = ocel
        self.df = ocel.get_extended_table()
        self.processConfig = process_config
        self.otypes = process_config.otypes
        self.activityLeadingTypes = process_config.activityLeadingTypes

    def __update_object_model(self, row, direct_object_model, otypes):
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
        for otype in [otype for otype in otypes if not otype == leading_type]:
            object_col = "ocel:type:" + otype
            otype_objects = row[object_col]
            if isinstance(otype_objects, list):
                event_objects += [(oid, otype) for oid in otype_objects]
        for i, (event_object, otype) in enumerate(event_objects):
            direct_object_model[leading_type][leading_object][otype].add(event_object)
            if event_object not in direct_object_model[otype]:
                direct_object_model[otype][event_object] = {
                    any_otype: set()
                    for any_otype in otypes
                }

    def make(self):
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


    def save(self):
        obj: ObjectInstance
        object_model = ObjectModel(self.processConfig.session_path)
        schema_frequencies = {
            otype: dict()
            for otype in self.otypes
        }
        for otype, objs in self.originalObjects.items():
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
        object_model.save_without_global_model(True)