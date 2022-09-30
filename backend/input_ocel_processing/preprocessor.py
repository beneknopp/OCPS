import os

import pm4py


class InputOCELPreprocessor:
    session_path: str

    def __init__(self, session_path, file_name, file):
        file_path = os.path.join(session_path, file_name)
        file.save(file_path)
        self.session_path = session_path
        self.ocel = pm4py.read_ocel(file_path)
        self.df = self.ocel.get_extended_table()
        self.otypes = self.ocel.globals["ocel:global-log"]["ocel:object-types"]
        self.acts = list(self.df["ocel:activity"].unique())

    def preprocess(self):
        self.__make_activity_otype_info()

    def get_activity_leading_otype_candidates(self):
        return self.activity_leading_otype_candidates

    def get_activity_allowed_otypes(self):
        return self.activity_allowed_otypes

    def get_activity_leading_type_groups(self):
        return self.activityLeadingTypeGroups

    def __make_activity_otype_info(self):
        df = self.df
        preliminary_activity_leading_otype_candidates = {
            act: []
            for act in df["ocel:activity"].values
        }
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
        unique_otype_occurrences_at_acts = df.groupby("ocel:activity", as_index=False) \
            .agg(self.__count_unique_otype_occurrences)
        for otype in self.otypes:
            taking_part_in = otype_occurrences_at_acts[
                otype_occurrences_at_acts[otype + ":count"] > 0.01
                ]["ocel:activity"].values
            candidate_leads = unique_otype_occurrences_at_acts[
                unique_otype_occurrences_at_acts[otype + ":count"] > 0.99
                ]["ocel:activity"].values
            for act in taking_part_in:
                activity_allowed_otypes[act].append(otype)
            for act in candidate_leads:
                preliminary_activity_leading_otype_candidates[act].append(otype)
        self.activity_allowed_otypes = activity_allowed_otypes
        self.preliminary_activity_leading_otype_candidates = preliminary_activity_leading_otype_candidates
        self.__iterate_for_leading_type_candidate_identification()

    def __iterate_for_leading_type_candidate_identification(self):
        iterator = self.df.iterrows()
        nextline = next(iterator, None)
        # otype -> id -> set of ids
        candidate_models = {
            act: {
                otype: dict()
                for otype in self.preliminary_activity_leading_otype_candidates[act]
            }
            for act in self.acts
        }
        while nextline is not None:
            index, event = nextline
            self.__update_candidate_models(event, candidate_models)
            nextline = next(iterator, None)
        activity_leading_type_groups = dict()
        for otype in self.otypes:
            # partner types can lead the same activities
            partners = []
            # ...
            anti_partners = []
            candidates = [act for act, ots in self.preliminary_activity_leading_otype_candidates.items()
                          if otype in ots]
            for act in candidates:
                other_candidates = [any_act for any_act in candidates if any_act != act]
                if not other_candidates:
                    partners.append([act])
                for any_act in other_candidates:
                    if any(act in group and any_act in group for group in partners):
                        continue
                    if any(act in group and any_act in group for group in anti_partners):
                        continue
                    are_partners = True
                    for obj in candidate_models[act][otype]:
                        obj_model_1 = candidate_models[act][otype][obj]
                        if obj in candidate_models[any_act][otype]:
                            obj_model_2 = candidate_models[any_act][otype][obj]
                            if obj_model_1 != obj_model_2:
                                are_partners = False
                    if are_partners:
                        box = partners
                    else:
                        box = anti_partners
                    existing_groups = [group for group in box if act in group or any_act in group]
                    if existing_groups:
                        group = existing_groups[0]
                        box.remove(group)
                        new_group = list(set(group + [act, any_act]))
                        box += [new_group]
                    else:
                        group = [act, any_act]
                        box += [group]
            activity_leading_type_groups[otype] = list(partners)
        self.activityLeadingTypeGroups = activity_leading_type_groups
        self.activity_leading_otype_candidates = self.preliminary_activity_leading_otype_candidates

    def __update_candidate_models(self, event, candidate_models):
        act = event["ocel:activity"]
        lt_candidates = self.preliminary_activity_leading_otype_candidates[act][:]
        for otype in lt_candidates:
            lead_objs_at_event = event["ocel:type:" + otype]
            if not isinstance(lead_objs_at_event, list):
                continue
            lead_obj = lead_objs_at_event[0]
            candidate_model = []
            for any_otype in self.otypes:
                any_objs = event["ocel:type:" + any_otype]
                if isinstance(any_objs, list):
                    candidate_model += any_objs
            candidate_model.sort()
            if lead_obj in candidate_models[act][otype]:
                if candidate_model != candidate_models[act][otype][lead_obj]:
                    # otype is not a leading type candidate
                    self.preliminary_activity_leading_otype_candidates[act].remove(otype)
            else:
                candidate_models[act][otype][lead_obj] = candidate_model

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

    def get_otypes(self):
        return self.otypes

    def get_acts(self):
        return self.acts

    def write_state(self):
        otypes_path = os.path.join(self.session_path, "otypes")
        with open(otypes_path, "w") as wf:
            wf.write(",".join(self.otypes))
