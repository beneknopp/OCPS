import json
import os
from datetime import datetime

import pm4py

from utils.running_id import RunningId


class InputOCELPreprocessor:

    def __init__(self, session_path, file_name, file):
        file_path = os.path.join(session_path, file_name)
        file.save(file_path)
        self.sessionPath = session_path
        self.ocel = pm4py.read_ocel(file_path)
        self.df = self.ocel.get_extended_table()
        self.otypes = self.ocel.globals["ocel:global-log"]["ocel:object-types"]
        self.realOtypes = self.otypes[:]
        self.acts = list(self.df["ocel:activity"].unique())

    def preprocess(self):
        self.__make_padding_types()
        self.__identify_leading_groups()
        self.__make_allowed_otypes_info()

    def get_activity_leading_otype_candidates(self):
        return self.activity_leading_otype_candidates

    def get_activity_allowed_otypes(self):
        return self.activity_allowed_otypes

    def get_activity_leading_type_groups(self):
        return {}

    def __make_padding_types(self):
        df = self.df
        self.paddingTypes = []
        for act in self.acts:
            running_id = RunningId()
            padding_type_name = "LEAD_" + act
            df["ocel:type:" + padding_type_name] = df.apply(
                lambda row:
                [padding_type_name + str(running_id.get_and_inc())]
                if row["ocel:activity"] == act else [], axis=1
            )
            self.otypes.append(padding_type_name)
            self.paddingTypes.append(padding_type_name)
        ocel = dict()
        ocel["ocel:global-event"] = {"ocel:activity": "__INVALID__"}
        ocel["ocel:global-object"] = {"ocel:type": "__INVALID__"}
        ocel["ocel:global-log"] = {
            "ocel:attribute-names": {},
            "ocel:object-types": self.otypes,
            "ocel:version": "1.0",
            "ocel:ordering": "timestamp"
        }
        ocel["ocel:events"] = {}
        ocel["ocel:objects"] = {}
        self.ocel = ocel
        df.apply(lambda row: self.__pad_event(row), axis=1)

    def __pad_event(self, row):
        events = self.ocel["ocel:events"]
        event = {}
        event["ocel:activity"] = row["ocel:activity"]
        event["ocel:timestamp"] = row["ocel:timestamp"]
        objs_by_type = {
            otype: row["ocel:type:" + otype]
            for otype in self.otypes
        }
        for otype in self.otypes:
            l = objs_by_type[otype]
            objs_by_type[otype] = l if isinstance(l, list) else []
        obj_ids = [obj_id for sl in objs_by_type.values() for obj_id in sl]
        event["ocel:omap"] = [str(obj_id) for obj_id in obj_ids]
        event["ocel:vmap"] = {}
        events[len(events) + 1] = event
        for otype, obj_ids in objs_by_type.items():
            for obj_id in obj_ids:
                if str(obj_id) not in self.ocel["ocel:objects"]:
                    self.ocel["ocel:objects"][str(obj_id)] = {"ocel:type": otype, "ocel:ovmap": {}}

    def __identify_leading_groups(self):
        iterator = self.df.iterrows()
        nextline = next(iterator, None)
        # otype -> id -> set of ids
        object_models = {
            otype: dict()
            for otype in self.paddingTypes
        }
        while nextline is not None:
            index, event = nextline
            self.__update_object_models(event, object_models)
            nextline = next(iterator, None)
        subsumed_by = {pt: set() for pt in self.paddingTypes}
        om_map = {otype: dict() for otype in self.paddingTypes}
        for padding_type1 in self.paddingTypes:
            for padding_type2 in self.paddingTypes:
                if padding_type1 == padding_type2:
                    continue
                if all(
                        any(om2 == om1 for om2 in object_models[padding_type2].values())
                        for om1 in object_models[padding_type1].values()
                ):
                    subsumed_by[padding_type1].add(padding_type2)
                    for om1_lead, om1 in object_models[padding_type1].items():
                        om2_lead = [om2_lead for om2_lead, om2 in object_models[padding_type2].items() if om2 == om1][0]
                        if om1_lead not in om_map[padding_type1]:
                            om_map[padding_type1][om1_lead] = dict()
                            om_map[padding_type1][om1_lead][padding_type2] = om2_lead
        equal_leads = {act: set() for act in self.acts}
        subsumed_leads = {act: set() for act in self.acts}
        for act1 in self.acts:
            for act2 in self.acts:
                if act1 == act2:
                    continue
                pad1 = "LEAD_" + act1
                pad2 = "LEAD_" + act2
                act1_leads = [pad1] + list(subsumed_by[pad1])
                act2_leads = [pad2] + list(subsumed_by[pad2])
                act1_leads.sort()
                act2_leads.sort()
                if act1_leads == act2_leads:
                    equal_leads[act1].add(act2)
                    equal_leads[act2].add(act1)
                if all(ot in act1_leads for ot in act2_leads):
                    subsumed_leads[act2].add(act1)
                if all(ot in act2_leads for ot in act1_leads):
                    subsumed_leads[act1].add(act2)
        assigned_acts = set()
        group_index = 1
        group_assignments = dict()
        group_leads = dict()
        for act in self.acts:
            if act not in assigned_acts:
                group = [act] + list(subsumed_leads[act])
                expanded_group = group[:]
                for any_act in group:
                    if any_act in group_assignments:
                        any_act_group = group_assignments[any_act]
                        other_acts = [other_act for other_act in self.acts if other_act in group_assignments
                                      and group_assignments[other_act] == any_act_group]
                        expanded_group += other_acts
                group = set(expanded_group)
                assigned_acts.update(group)
                for any_act in group:
                    group_assignments[any_act] = group_index
                group_leads[group_index] = act
                group_index = group_index + 1
        self.activity_leading_otype_candidates = {
            act: ["LEAD_" + group_leads[group_assignments[act]]] for act in self.acts
        }
        for event_id, event in self.ocel["ocel:events"].items():
            act = event["ocel:activity"]
            timestamp = event["ocel:timestamp"].timestamp()
            event["ocel:timestamp"] = datetime.fromtimestamp(int(timestamp)).strftime(
                '%Y-%m-%d %H:%M:%S')
            padding_type = "LEAD_" + act
            old_lead_obj = [oid for oid in event["ocel:omap"] if oid[:5] == "LEAD_"][0]
            group = group_assignments[act]
            group_leading_act = group_leads[group]
            leading_padding_type = "LEAD_" + group_leading_act
            if leading_padding_type != padding_type:
                group_lead_obj = om_map[padding_type][old_lead_obj][leading_padding_type]
                event["ocel:omap"] = [oid for oid in event["ocel:omap"] if oid != old_lead_obj] + [group_lead_obj]
                if old_lead_obj in self.ocel["ocel:objects"]:
                    del self.ocel["ocel:objects"][old_lead_obj]
        self.ocel["ocel:global-log"]["ocel:object-types"] = self.realOtypes + ["LEAD_" + leading_act for leading_act in
                                                                           group_leads.values()]
        ocel_path = os.path.join(self.sessionPath, "padded_ocel.jsonocel")
        with open(ocel_path, "w") as write_file:
            json.dump(self.ocel, write_file, indent=4)
        self.ocel = pm4py.read_ocel(ocel_path)
        self.df = self.ocel.get_extended_table()
        self.otypes = self.ocel.globals["ocel:global-log"]["ocel:object-types"]
        self.realOtypes = self.otypes[:]
        self.acts = list(self.df["ocel:activity"].unique())

    def __update_object_models(self, event, object_models):
        act = event["ocel:activity"]
        padding_type = "LEAD_" + act
        lead_obj = event["ocel:type:" + padding_type][0]
        object_model = []
        for any_otype in self.realOtypes:
            any_objs = event["ocel:type:" + any_otype]
            if isinstance(any_objs, list):
                object_model += any_objs
            object_model.sort()
        object_model = tuple(object_model)
        object_models[padding_type][lead_obj] = object_model

    def __list_length(self, val):
        if isinstance(val, list):
            return len(val)
        return 0

    def __count_otype_occurrences(self, series):
        count = series.count()
        return float(len(series[series > 0])) / count if count > 0 else 0

    def __count_unique_otype_occurrences(self, series):
        count = series.count()
        return float(len(series[series == 1])) / count if count > 0 else 0

    def __make_allowed_otypes_info(self):
        df = self.df
        activity_allowed_otypes = {
            act: []
            for act in df["ocel:activity"].values
        }
        for otype in self.otypes:
            val_col = "ocel:type:" + otype
            count_col = otype + ":count"
            df[count_col] = df[val_col].apply(lambda val: self.__list_length(val))
        otype_occurrences_at_acts = df.groupby("ocel:activity", as_index=False) \
            .agg(self.__count_otype_occurrences)
        for otype in self.otypes:
            taking_part_in = otype_occurrences_at_acts[
                otype_occurrences_at_acts[otype + ":count"] > 0.01
                ]["ocel:activity"].values
            for act in taking_part_in:
                activity_allowed_otypes[act].append(otype)
        self.activity_allowed_otypes = activity_allowed_otypes

    def write_state(self):
        otypes_path = os.path.join(self.sessionPath, "otypes")
        with open(otypes_path, "w") as wf:
            wf.write(",".join(self.otypes))

    def get_otypes(self):
        return self.otypes

    def get_acts(self):
        return self.acts
