import os
import pickle


class ProcessConfig:
    otypes: list
    acts: list
    activitySelectedTypes: dict
    activityLeadingTypes: dict
    otypeLeadingActivities: dict
    nonEmittingTypes: list
    useOriginalMarking: bool

    @classmethod
    def load(cls, session_path):
        process_config_path = os.path.join(session_path, "process_config.pkl")
        return pickle.load(open(process_config_path, "rb"))

    def __init__(self, config_dto, session_path):
        self.session_path = session_path
        self.acts = config_dto["acts"]
        self.nonEmittingTypes = config_dto["non_emitting_types"]
        self.activitySelectedTypes = config_dto["activity_selected_types"]
        self.activityLeadingTypes = config_dto["activity_leading_type_selections"]
        # TODO: why are there duplicates?
        self.otypes = list(set([otype for otype in config_dto["otypes"]
                       if any(otype in self.activitySelectedTypes[act] for act in self.acts)]))
        self.otypeLeadingActivities = {
            otype: [act for act, ot in self.activityLeadingTypes.items() if ot == otype]
            for otype in self.otypes
        }
        self.useOriginalMarking = True

    def save(self):
        process_config_path = os.path.join(self.session_path, "process_config.pkl")
        with open(process_config_path, "wb") as write_file:
            pickle.dump(self, write_file)

    @classmethod
    def update_use_original_marking(cls, session_path, use_original_marking):
        process_config = cls.load(session_path)
        process_config.useOriginalMarking = use_original_marking
        process_config.save()

    @classmethod
    def update_non_emitting_types(cls, session_path, non_emitting_types_str):
        process_config: ProcessConfig = cls.load(session_path)
        process_config.nonEmittingTypes = non_emitting_types_str.split(",") if len(non_emitting_types_str) > 0 else []
        process_config.save()
