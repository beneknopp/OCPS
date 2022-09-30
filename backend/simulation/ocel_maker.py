import json
import os
import pickle
from datetime import datetime

import pm4py

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
        ocel_path = os.path.join(self.sessionPath, "simulated_ocel.jsonocel")
        with open(ocel_path, "w") as write_file:
            json.dump(self.ocel, write_file, indent=4)
        simulated_ocel = pm4py.read_ocel(ocel_path)
        for otype in self.processConfig.otypes:
            flattened_otype_simulated = pm4py.ocel_flattening(simulated_ocel, otype)
            flat_path = os.path.join(self.sessionPath, 'flattened_' + otype + '_simulated.xes')
            pm4py.write_xes(flattened_otype_simulated, flat_path)
