import json
import os
import pickle
from datetime import datetime

import pm4py
#from pm4py import filter_ocel_object_types
from pm4py.ocel import ocel_flattening

from input_ocel_processing.process_config import ProcessConfig


class OcelMaker:
    ocel: {}

    @classmethod
    def load(cls, session_path):
        ocel_maker_path = os.path.join(session_path, "ocel_maker.pkl")
        return pickle.load(open(ocel_maker_path, "rb"))

    def save(self):
        ocel_maker_path = os.path.join(self.sessionPath, "ocel_maker.pkl")
        with open(ocel_maker_path, "wb") as write_file:
            pickle.dump(self, write_file)

    def __init__(self, session_path, objects, attribute_names, timestamp_offset):
        self.sessionPath = session_path
        self.processConfig = ProcessConfig.load(session_path)
        self.timestampOffset = timestamp_offset
        ocel = dict()
        ocel["ocel:global-event"] = {"ocel:activity": "__INVALID__"}
        ocel["ocel:global-object"] = {"ocel:type": "__INVALID__"}
        ocel["ocel:global-log"] = {
            "ocel:attribute-names": attribute_names,
            "ocel:object-types": self.processConfig.otypes,
            "ocel:version": "1.0",
            "ocel:ordering": "timestamp"
        }
        ocel["ocel:events"] = {}
        ocel["ocel:objects"] = objects
        ocel["ocel:global-log"]["ocel:leading-types"] = self.processConfig.activityLeadingTypes
        self.ocel = ocel

    def add_event(self, activity, timestamp, omap, vmap):
        events = self.ocel["ocel:events"]
        event = {}
        event["ocel:activity"] = activity
        event["ocel:timestamp"] = datetime.fromtimestamp(int(timestamp) + self.timestampOffset).strftime(
            '%Y-%m-%d %H:%M:%S')
        event["ocel:omap"] = omap
        event["ocel:vmap"] = vmap
        events[len(events) + 1] = event

    def write_ocel(self):
        if not self.ocel["ocel:events"]:
            return
        ocel_path = os.path.join(self.sessionPath, "simulated_ocel_raw.jsonocel")
        with open(ocel_path, "w") as write_file:
            json.dump(self.ocel, write_file, indent=4)
        # TODO: constructor ocel from json
        simulated_ocel = pm4py.read_ocel(ocel_path)
        otype_acts = pm4py.ocel.ocel_object_type_activities(simulated_ocel)
        allowed_otype_acts = { otype : acts for otype, acts in otype_acts.items() if not otype.startswith("LEAD_") }
        filtered_simulated_ocel = pm4py.filtering.filter_ocel_object_types_allowed_activities(simulated_ocel, allowed_otype_acts)
        nof_objs = len(filtered_simulated_ocel.objects)
        ocel_path = os.path.join(self.sessionPath, "simulated_ocel_origMarking=" + str(self.processConfig.useOriginalMarking).lower() + \
                                 "_nofObjects=" + str(nof_objs) + ".jsonocel")
        pm4py.write_ocel(filtered_simulated_ocel, ocel_path)
        ProcessConfig.update_simul_count(self.sessionPath, nof_objs)
        for otype in self.processConfig.otypes:
            flattened_otype_simulated = ocel_flattening(simulated_ocel, otype)
            flat_path = os.path.join(self.sessionPath, 'flattened_' + otype + '_simulated_' + str(nof_objs) + '.xes')
            pm4py.write_xes(flattened_otype_simulated, flat_path)
            ProcessConfig.update_simul_type_count(self.sessionPath, otype, nof_objs)
