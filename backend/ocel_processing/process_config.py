import os
import pickle

# TODO: factor out or make marking specific
class ProcessConfig:
    otypes: list
    acts: list
    activitySelectedTypes: dict
    activityLeadingTypes: dict
    otypeLeadingActivities: dict
    nonEmittingTypes: list
    useOriginalMarking: bool
    simulCount : int
    simulTypeCount: dict
    clockOffset: int

    @classmethod
    def load(cls, session_path):
        process_config_path = os.path.join(session_path, "process_config.pkl")
        return pickle.load(open(process_config_path, "rb"))

    def __init__(self, session_path):
        self.sessionPath = session_path
        self.raw_ocel_path = None
        self.acts = None
        self.clockOffset = None
        self.simulCount = None
        self.nonEmittingTypes = None
        self.activitySelectedTypes = None
        self.activityLeadingTypes = None
        self.otypeLeadingActivities = None
        self.otypes = None

    def init_config(self, config_dto):
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
        self.simulCount = 0
        self.simulTypeCount = {
            otype: 0
            for otype in self.otypes
        }
        self.useOriginalMarking = True

    def save(self):
        process_config_path = os.path.join(self.sessionPath, "process_config.pkl")
        with open(process_config_path, "wb") as write_file:
            pickle.dump(self, write_file)

    @classmethod
    def update_use_original_marking(cls, session_path, use_original_marking):
        process_config = cls.load(session_path)
        process_config.useOriginalMarking = use_original_marking
        process_config.save()

    @classmethod
    def update_raw_ocel_path(cls, session_path, raw_ocel_path):
        process_config = cls.load(session_path)
        process_config.raw_ocel_path = raw_ocel_path
        process_config.save()

    @classmethod
    def update_object_types(cls, session_path, otypes: list[str]):
        process_config: ProcessConfig = cls.load(session_path)
        process_config.otypes = otypes
        process_config.save()

    @classmethod
    def update_non_emitting_types(cls, session_path, non_emitting_types_str):
        process_config: ProcessConfig = cls.load(session_path)
        process_config.nonEmittingTypes = non_emitting_types_str.split(",") if len(non_emitting_types_str) > 0 else []
        process_config.save()

    #TODO right now only total object count is used
    @classmethod
    def update_simul_type_count(cls, session_path, otype, count):
        process_config: ProcessConfig = cls.load(session_path)
        process_config.simulTypeCount[otype] = count
        process_config.save()

    @classmethod
    def update_simul_count(cls, session_path, count):
        process_config: ProcessConfig = cls.load(session_path)
        process_config.simulCount = count
        process_config.save()

    @classmethod
    def get_simul_count(cls, session_path):
        process_config: ProcessConfig = cls.load(session_path)
        count = process_config.simulCount
        return count

    @classmethod
    def get_raw_ocel_path(cls, session_path):
        process_config: ProcessConfig = cls.load(session_path)
        raw_ocel_path = process_config.raw_ocel_path
        return raw_ocel_path