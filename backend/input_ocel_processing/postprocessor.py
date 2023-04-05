import os
import pm4py

from input_ocel_processing.process_config import ProcessConfig


class InputOCELPostprocessor:

    def __init__(self, session_path, process_config: ProcessConfig):
        self.session_path = session_path
        raw_ocel_path = os.path.join(session_path, "input.jsonocel")
        self.ocel = pm4py.read_ocel(raw_ocel_path)
        self.process_config = process_config

    def postprocess(self):
        process_config = self.process_config
        ocel = self.ocel
        postprocessed_ocel_path = os.path.join(self.session_path, "postprocessed_input.jsonocel")
        otypes = process_config.otypes
        activity_selected_types = process_config.activitySelectedTypes
        postprocessed_ocel = pm4py.filter_ocel_object_types_allowed_activities(ocel, {
            otype: [act for act, types in activity_selected_types.items() if otype in types]
            for otype in otypes
        })
        pm4py.write_ocel(postprocessed_ocel, postprocessed_ocel_path)
        for otype in otypes:
            flog = pm4py.ocel.ocel_flattening(postprocessed_ocel, otype)
            flat_path = os.path.join(self.session_path, "flattened_" + otype + ".xes")
            pm4py.write_xes(flog, flat_path)
        return postprocessed_ocel

    def make_default_distributions(self):
        self.__
        pass
