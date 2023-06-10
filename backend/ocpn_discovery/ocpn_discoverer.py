import json
import os

import pm4py
from pm4py.objects.ocel.obj import OCEL

from .net_utils import Place, Transition, TransitionType, Arc, NetProjections

from ocpa.objects.log.importer.ocel import factory as ocel_import_factory
from ocpa.visualization.oc_petri_net import factory as ocpn_vis_factory
from eval.evaluators import ocpn_to_ocel
from ocpa.algo.discovery.ocpn import algorithm as ocpn_discovery_factory

class OCPN_DTO:

    def __init__(self):
        pass


class OCPN_Discoverer:
    ocel: OCEL
    precision = ""
    fitness = ""

    def __init__(self, session_path):
        self.sessionPath = session_path
        self.file_path = os.path.join(session_path, "postprocessed_input.jsonocel")
        self.ocel = pm4py.read_ocel(self.file_path)

    def discover(self, activity_selected_types):
        otypes = list(set([otype for otype_list in activity_selected_types.values() for otype in otype_list]))
        self.otypes = otypes
        types_selected_activity = {
            otype: [act for act in activity_selected_types if otype in activity_selected_types[act]] + ["END_" + otype]
            for otype in otypes
        }
        #ocel = pm4py.filter_ocel_object_types_allowed_activities(ocel, types_selected_activity)
        ocel: OCEL = ocel_import_factory.apply(file_path=self.file_path)
        ocpn = ocpn_discovery_factory.apply(ocel, parameters={"debug": False})
        #ocpn_dict = pm4py.discover_oc_petri_net(ocel)
        self.ocpn_dict = ocpn
        self.ocel = ocel
        self.__extract_net(ocpn)
        self.__identify_variable_arcs()
        self.__extend_with_start_and_end()
        self.__make_projections()

    def evaluate(self):
        original_file_path = os.path.join(self.sessionPath, "input.jsonocel")
        original_ocel = ocel_import_factory.apply(file_path=original_file_path)
        # TODO: proper projection of padded net (self.ocpn) to original types (non "LEAD_...")
        # here we boldly assume that the application of the same discovery method to the original ocel yields the same result
        projected_ocpn = ocpn_discovery_factory.apply(original_ocel)
        precision, fitness = ocpn_to_ocel(ocel=original_ocel, ocpn=projected_ocpn)
        eval_path = os.path.join(self.sessionPath, "ocpn_eval.txt")
        with open(eval_path, "w") as wf:
            wf.write("precision=" + str(precision) + ";fitness=" + str(fitness))
        self.precision = precision
        self.fitness = fitness

    def save(self):
        self.__save_projections()
        # for debugging
        self.__save_json_export()

    def __extract_net(self, ocpn_dict):
        self.places = dict()
        self.transitions = dict()
        self.arcs = dict()
        arc_id = 1
        for i, otype in enumerate(self.otypes):
            net = ocpn_dict.nets[otype][0]
            places_ = net.places
            transitions_ = net.transitions
            arcs_ = net.arcs
            for place_ in places_:
                name = place_.name + "_" + str(i)
                is_initial = len(place_.in_arcs) == 0
                is_final = len(place_.out_arcs) == 0
                place = Place(name, otype, is_initial, is_final)
                self.places[name] = place
            for transition_ in transitions_:
                label = transition_.label
                id = self.__get_transition_id(transition_, i)
                transition_type = TransitionType.SILENT if label is None else TransitionType.ACTIVITY
                if id not in self.transitions:
                    transition = Transition(id, label, transition_type)
                    self.transitions[id] = transition
            for arc_ in arcs_:
                name = "a_" + str(arc_id)
                source_ = arc_.source
                target_ = arc_.target
                try_source_name_as_place = source_.name + "_" + str(i)
                source_is_place = try_source_name_as_place in self.places
                if source_is_place:
                    source_name = try_source_name_as_place
                    source = self.places[source_name]
                    target_id = self.__get_transition_id(arc_.target, i)
                    target = self.transitions[target_id]
                else:
                    target_name = target_.name + "_" + str(i)
                    target = self.places[target_name]
                    source_id = self.__get_transition_id(arc_.source, i)
                    source = self.transitions[source_id]
                arc = Arc(name, source, target, is_variable_arc=False)
                arc_id = arc_id + 1
                self.arcs[name] = arc

    def __get_transition_id(self, transition_, otype_index):
        label = transition_.label
        return label if label is not None else transition_.name + "_" + str(otype_index)

    def __identify_variable_arcs(self):
        ocel = self.ocel
        variable_acts_per_type = {
            otype: []
            for otype in self.otypes
        }
        df = ocel.log.log
        #df = ocel.get_extended_table()
        for otype in self.otypes:
            #val_col = "ocel:type:" + otype
            count_col = otype + ":count"
            #df[count_col] = df[val_col].apply(lambda val: self.__list_length(val))
            df[count_col] = df[otype].apply(lambda val: self.__list_length(val))
        #unique_otype_occurrences_at_acts = df.groupby("ocel:activity", as_index=False) \
        unique_otype_occurrences_at_acts = df.groupby("event_activity", as_index=False) \
            .agg(self.__count_unique_otype_occurrences)
        #any_otype_occurrences_at_acts = df.groupby("ocel:activity", as_index=False) \
        any_otype_occurrences_at_acts = df.groupby("event_activity", as_index=False) \
            .agg(self.__count_any_otype_occurrences)
        for otype in self.otypes:
            candidate_var_acts = unique_otype_occurrences_at_acts[
                unique_otype_occurrences_at_acts[otype + ":count"] < 0.99
                ]["event_activity"].values
                #]["ocel:activity"].values
            candidate_var_acts = [
                act for act in candidate_var_acts if act in
                                                     any_otype_occurrences_at_acts[
                                                         any_otype_occurrences_at_acts[otype + ":count"] > 0.01
                                                        ]["event_activity"].values
                                                         #]["ocel:activity"].values
            ]
            for act in candidate_var_acts:
                variable_acts_per_type[otype].append(act)
        arc: Arc
        for otype, variable_acts in variable_acts_per_type.items():
            for act in variable_acts:
                otype_input_arcs_to_act = list(filter(lambda arc: isinstance(arc.source, Place) and
                                                                  arc.source.otype == otype and arc.target.label == act,
                                                      self.arcs.values()))
                otype_output_arcs_from_act = list(filter(lambda arc: isinstance(arc.target, Place) and
                                                                     arc.target.otype == otype and arc.source.label == act,
                                                         self.arcs.values()))
                for arc in otype_input_arcs_to_act + otype_output_arcs_from_act:
                    arc.isVariableArc = True

    def __extend_with_start_and_end(self):
        for otype in self.otypes:
            old_initial_place = list(filter(lambda p: p.isInitial and p.otype == otype, self.places.values()))[0]
            if len(old_initial_place.outgoingArcs) > 1 or \
                    any(arc.transEnd.transitionType == TransitionType.ACTIVITY for arc in
                        old_initial_place.outgoingArcs):
                old_initial_place.isInitial = False
                initial_transition_id = "START_" + otype
                initial_transition = Transition(initial_transition_id, initial_transition_id, TransitionType.INITIAL)
                initial_place_id = "new_source_" + otype
                new_initial_place = Place(initial_place_id, otype, is_initial=True, is_final=False)
                arc_count = len(self.arcs)
                arc1 = Arc("a_" + str(arc_count + 1), new_initial_place, initial_transition, False)
                arc2 = Arc("a_" + str(arc_count + 2), initial_transition, old_initial_place, False)
                self.places[initial_place_id] = new_initial_place
                self.transitions[initial_transition_id] = initial_transition
                self.arcs[arc1.id] = arc1
                self.arcs[arc2.id] = arc2
            else:
                initial_transition: Transition = old_initial_place.outgoingArcs[0].transEnd
                initial_transition.transitionType = TransitionType.INITIAL
            ##########################
            old_final_place = list(filter(lambda p: p.isFinal and p.otype == otype, self.places.values()))[0]
            if len(old_final_place.incomingArcs) > 1 or \
                    any(arc.transEnd.transitionType == TransitionType.ACTIVITY for arc in old_final_place.incomingArcs):
                old_final_place.isFinal = False
                final_transition_id = "END_" + otype
                final_transition = Transition(final_transition_id, final_transition_id, TransitionType.FINAL)
                final_place_id = "new_sink_" + otype
                new_final_place = Place(final_place_id, otype, is_initial=False, is_final=True)
                arc_count = len(self.arcs)
                arc3 = Arc("a_" + str(arc_count + 1), old_final_place, final_transition, False)
                arc4 = Arc("a_" + str(arc_count + 2), final_transition, new_final_place, False)
                self.places[final_place_id] = new_final_place
                self.transitions[final_transition_id] = final_transition
                self.arcs[arc3.id] = arc3
                self.arcs[arc4.id] = arc4
            else:
                final_transition: Transition = old_final_place.incomingArcs[0].transEnd
                final_transition.transitionType = TransitionType.FINAL
                final_transition.id = "END_" + otype
                final_transition.label = "END_" + otype

    def __list_length(self, val):
        if isinstance(val, list):
            return len(val)
        return 0

    def __count_unique_otype_occurrences(self, series):
        count = series.count()
        try:
            return float(len(series[series == 1])) / count
        except:
            return 0

    def __count_any_otype_occurrences(self, series):
        count = series.count()
        try:
            float(len(series[series > 0])) / count
        except:
            return 0

    def __make_projections(self):
        self.netProjections = NetProjections(self.places, self.otypes)

    def __save_projections(self):
        self.netProjections.save(self.sessionPath)

    def __save_json_export(self):
        json_export = {}
        places = []
        for place in self.places.values():
            places.append({
                "id": place.id,
                "otype": place.otype,
                "isInitial": place.isInitial,
                "isFinal": place.isFinal
            })
        transitions = []
        for transition in self.transitions.values():
            transitions.append({
                "id": transition.id,
                "label": transition.label,
                "transitionType": transition.transitionType.value
            })
        arcs = []
        for arc in self.arcs.values():
            arcs.append({
                "id": arc.id,
                "source": arc.source.id,
                "target": arc.target.id,
                "isVariableArc": arc.isVariableArc
            })
        json_export["places"] = places
        json_export["transitions"] = transitions
        json_export["arcs"] = arcs
        with open(os.path.join(self.sessionPath, "ocpn_export.json"), "w") as write_file:
            json.dump(json_export, write_file, indent=4)

    def export(self):
        place: Place
        transition: Transition
        arc: Arc
        places = list(map(lambda place: place.as_json(), self.places.values()))
        transitions = list(map(lambda transition: transition.as_json(), self.transitions.values()))
        arcs = list(map(lambda arc: arc.as_json(), self.arcs.values()))
        return {"places": places, "transitions": transitions, "arcs": arcs,
                "precision": self.precision , "fitness": self.fitness}
