import math
import os

import numpy as np
import pm4py

from input_ocel_processing.process_config import ProcessConfig
from object_model_generation.training_model_preprocessor import TrainingModelPreprocessor
from pm4py.statistics.variants.log import get as variants_module
from pm4py.algo.evaluation.earth_mover_distance import algorithm as emd_evaluator


class SimulationEvaluator:

    sessionPath: str
    processConfig: ProcessConfig
    trainingModelPreprocessor : TrainingModelPreprocessor
    originalOcel: any
    originalFlatLogs: dict
    originalFlatLogDataFrames: dict
    originalActivityDelays: dict
    originalCycleTimes : dict
    simulatedOcel: any
    simulatedFlatLogs: dict
    simulatedFlatLogDataFrames: dict
    simulatedActivityDelays: dict
    simulatedCycleTimes: dict
    earthMoversConformances: dict

    def __init__(self, session_path):
        self.sessionPath = session_path
        self.processConfig = ProcessConfig.load(session_path)
        self.otypes = self.processConfig.otypes
        #self.trainingModelPreprocessor = TrainingModelPreprocessor.load(session_path)
        self.originalActivityDelays = dict()
        self.originalCycleTimes = dict()
        self.simulatedActivityDelays = dict()
        self.simulatedCycleTimes = dict()
        self.earthMoversConformances = dict()

    def evaluate(self):
        self.__load_data()
        for otype in self.otypes:
            self.__compute_otype_delays(otype)
            self.__compute_emd_conformance(otype)

    def __compute_otype_delays(self, otype):
        original_flog = self.originalFlatLogDataFrames[otype]
        simulated_flog = self.simulatedFlatLogDataFrames[otype]
        print("Computing timing information for original '" + otype + "' data")
        original_act_delays, original_cycle_times = self.__compute_delays(original_flog)
        print("Computing timing information for simulated '" + otype + "' data")
        simulated_act_delays, simulated_cycle_times = self.__compute_delays(simulated_flog)
        self.originalActivityDelays[otype] = original_act_delays
        self.originalCycleTimes[otype] = original_cycle_times
        self.simulatedActivityDelays[otype] = simulated_act_delays
        self.simulatedCycleTimes[otype] = simulated_cycle_times

    def __compute_emd_conformance(self, otype):
        print("Computing earth mover's distance for '" + otype + "'")
        original_flog = self.originalFlatLogDataFrames[otype]
        simulated_flog = self.simulatedFlatLogDataFrames[otype]
        original_flog_1 = pm4py.convert_to_event_log(original_flog)
        simulated_flog_1 = pm4py.convert_to_event_log(simulated_flog)
        original_language = variants_module.get_language(original_flog_1)
        simulated_language = variants_module.get_language(simulated_flog_1)
        emd = emd_evaluator.apply(simulated_language, original_language)
        self.earthMoversConformances[otype] = 1-emd

    def export(self):
        return {
            "otypes": self.otypes,
            "originalCycleTimes": self.originalCycleTimes,
            "simulatedCycleTimes": self.simulatedCycleTimes,
            "earthMoversConformances": self.earthMoversConformances
        }

    def __load_data(self):
        original_ocel_path = os.path.join(self.sessionPath, "postprocessed_input.jsonocel")
        self.originalOcel = pm4py.read_ocel(original_ocel_path)
        original_flat_logs = {}
        original_flat_frames = {}
        for otype in self.otypes:
            flog_path = os.path.join(self.sessionPath, "flattened_" + otype + ".xes")
            original_flog = pm4py.read_xes(flog_path)
            original_flat_logs[otype] = original_flog
            original_frame = pm4py.convert_to_dataframe(original_flog)
            original_frame = original_frame.sort_values(["case:concept:name", "time:timestamp"])
            original_flat_frames[otype] = original_frame
        self.originalFlatLogs = original_flat_logs
        self.originalFlatLogDataFrames = original_flat_frames
        simul_count = self.processConfig.get_simul_count(self.sessionPath)
        simulated_ocel_path = os.path.join(self.sessionPath, "simulated_ocel_origMarking=" +\
                                 str(self.processConfig.useOriginalMarking).lower() +\
                                 "_nofObjects=" + str(simul_count) + ".jsonocel")
        self.simulatedOcel = pm4py.read_ocel(simulated_ocel_path)
        simulated_flat_logs = {}
        simulated_flat_frames = {}
        for otype in self.otypes:
            flog_path = os.path.join(self.sessionPath, 'flattened_' + otype + '_simulated_' + str(simul_count) + '.xes')
            simulated_flog = pm4py.read_xes(flog_path)
            simulated_flat_logs[otype] = simulated_flog
            simulated_frame = pm4py.convert_to_dataframe(simulated_flog)
            simulated_frame = simulated_frame.sort_values(["case:concept:name", "time:timestamp"])
            simulated_flat_frames[otype] = simulated_frame
        self.simulatedFlatLogs = simulated_flat_logs
        self.simulatedFlatLogDataFrames = simulated_flat_frames

    def __compute_delays(self, log):
        activity_delays = {
            act: []
            for act in set(log["concept:name"].values)
        }
        cycle_times = []
        log["delay"] = 0
        log["int:timestamp"] = log.apply(lambda row: int(row["time:timestamp"].timestamp()), axis=1)
        iterator = log.iterrows()
        index, line = next(iterator, None)
        lastline, lastindex = line, index
        nextline = next(iterator, None)
        firstline = line
        while nextline is not None:
            index, line = nextline
            if line["case:concept:name"] == lastline["case:concept:name"]:
                # Update DELAY
                delay = line["int:timestamp"] - lastline["int:timestamp"]
                log.at[lastindex, 'delay'] = delay
                activity_delays[lastline["concept:name"]].append(delay)
            else:
                activity_delays[lastline["concept:name"]].append(0)
                cycle_time = lastline["int:timestamp"] - firstline["int:timestamp"]
                cycle_times.append(cycle_time)
                firstline = line
            lastline, lastindex = line, index
            nextline = next(iterator, None)
        activity_delay_stats = {
            act: {
                "mean": np.mean(delays),
                "stdev": math.sqrt(np.var(delays)),
            }
            for act, delays in activity_delays.items()
        }
        cycle_times_stats = {
            "mean": np.mean(cycle_times),
            "stdev": math.sqrt(np.var(cycle_times)),
        }
        return activity_delay_stats, cycle_times_stats