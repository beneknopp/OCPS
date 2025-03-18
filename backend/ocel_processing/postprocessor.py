import os
import pm4py

from ocel_processing.load_ocel import load_ocel
from ocel_processing.process_config import ProcessConfig


class InputOCELPostprocessor:

    def __init__(self, session_path, process_config: ProcessConfig):
        self.session_path = session_path
        raw_ocel_path = process_config.raw_ocel_path
        self.ocel = load_ocel(raw_ocel_path)
        self.process_config = process_config

    def postprocess(self):
        process_config = self.process_config
        ocel = self.ocel
        postprocessed_ocel_path = os.path.join(self.session_path, "postprocessed_input.sqlite")
        otypes = process_config.otypes
        activity_selected_types = process_config.activitySelectedTypes
        postprocessed_ocel = pm4py.filter_ocel_object_types_allowed_activities(ocel, {
            otype: [act for act, types in activity_selected_types.items() if otype in types]
            for otype in otypes
        })
        if os.path.exists(postprocessed_ocel_path):
            os.remove(postprocessed_ocel_path)
        pm4py.write_ocel2(postprocessed_ocel, postprocessed_ocel_path)
        return postprocessed_ocel

    #TODO
    def get_clock_offset(self):
        return int(min(self.ocel.get_extended_table()["ocel:timestamp"].values))
