import math
import os
import pickle
from datetime import timedelta

import numpy as np
import pm4py

from input_ocel_processing.process_config import ProcessConfig
from object_model_generation.training_model_preprocessor import TrainingModelPreprocessor
from pm4py.statistics.variants.log import get as variants_module
from pm4py.algo.evaluation.earth_mover_distance import algorithm as emd_evaluator


class SimulationRunEvaluation:
    otype: str
    # steps -> eval info
    simulatedActivityDelays: dict
    simulatedA2ADelays: dict
    simulatedCycleTimes: dict
    earthMoversConformances: dict

    def __init__(self, otype: str, simulated_activity_delays, simulated_a2a_delays, simulated_cycle_times,
                 earth_movers_conformances):
        self.otype = otype
        self.simulatedActivityDelays = simulated_activity_delays
        self.simulatedA2ADelays = simulated_a2a_delays
        self.simulatedCycleTimes = simulated_cycle_times
        self.earthMoversConformances = earth_movers_conformances

    def get(self, stats_type):
        if stats_type == "actdelays":
            return self.simulatedActivityDelays, self.simulatedA2ADelays
        if stats_type == "cycletimes":
            return self.simulatedCycleTimes
        if stats_type == "earthmovers":
            return self.earthMoversConformances
        raise AttributeError()


class SimulationEvaluator:
    sessionPath: str
    processConfig: ProcessConfig
    trainingModelPreprocessor: TrainingModelPreprocessor
    originalOcel: any
    originalFlatLogs: dict
    originalFlatLogDataFrames: dict
    originalActivityDelays: dict
    originalCycleTimes: dict
    originalLanguages: dict
    simulatedOcel: any
    simulatedFlatLogs: dict
    simulatedFlatLogDataFrames: dict
    simulatedActivityDelays: dict
    simulatedCycleTimes: dict
    earthMoversConformances: dict
    GRID = 10

    def __init__(self, session_path, selected_object_models):
        self.sessionPath = session_path
        self.processConfig = ProcessConfig.load(session_path)
        self.otypes = self.processConfig.otypes
        self.selectedObjectModels = selected_object_models
        self.evaluations = dict()
        self.a2aSupport = {otype: [] for otype in self.otypes}

    def save(self):
        path = os.path.join(self.sessionPath, "simulation_evaluations.pkl")
        with open(path, "wb") as write_file:
            pickle.dump(self, write_file)

    @classmethod
    def load(cls, session_path):
        path = os.path.join(session_path, "simulation_evaluations.pkl")
        return pickle.load(open(path, "rb"))

    def __merge_axes_info(self):
        merged_axes = []
        for ticks in self.stepTicks.values():
            merged_axes += ticks
        merged_axes = sorted(list(set(merged_axes)))
        return merged_axes

    # TODO: some refactoring / restructuring / reengineering
    def get(self, otype, stats_type):
        # barChartData: {[attribute: string]: {data: number[], label: string}[]}
        # mbarChartLabels: {[attribute: string]: string[]} = {};
        axes = self.__merge_axes_info()
        if stats_type == "actdelays":
            actTypes = self.processConfig.activitySelectedTypes
            acts = [act for act, types in actTypes.items() if otype in types]
            a2as = self.a2aSupport[otype]
            a2as_pretty = {(act1, act2): "'" + act1 + "' to '" + act2 + "'" for act1, act2 in a2as}
            attributes = acts + list(a2as_pretty.values())
            chart_data = {attribute: [] for attribute in acts + list(a2as_pretty.values())}
            for model in self.selectedObjectModels:
                act_data_, a2a_data_ = self.simulationRunEvaluations[model][otype].get(stats_type)
                # TODO: properly store attributes
                for attribute in acts:
                    chart_entry_act = {"data": [0.0], "label": model, "type": "line"}
                    for i in range(1, len(axes)):
                        tick = axes[i]
                        if tick not in act_data_:
                            chart_entry_act["data"].append(chart_entry_act["data"][i - 1])
                        else:
                            chart_entry_act["data"].append(act_data_[tick][attribute])
                    chart_data[attribute].append(chart_entry_act)
                for attribute in a2as:
                    chart_entry_a2a = {"data": [0.0], "label": model, "type": "line"}
                    for i in range(1, len(axes)):
                        tick = axes[i]
                        if tick not in a2a_data_ or attribute not in a2a_data_[tick]:
                            chart_entry_a2a["data"].append(chart_entry_a2a["data"][i - 1])
                        else:
                            chart_entry_a2a["data"].append(a2a_data_[tick][attribute])
                    chart_data[a2as_pretty[attribute]].append(chart_entry_a2a)
            input_act_data_ = self.originalActivityDelays[otype]
            input_a2a_data_ = self.originalA2ADelays[otype]
            for attribute in acts:
                chart_entry_act = {
                    "data": [], "label": "Input Data", "type": "line"
                }
                act_datum = input_act_data_[attribute]
                for i in range(len(axes)):
                    chart_entry_act["data"].append(act_datum)
                chart_data[attribute].append(chart_entry_act)
            for attribute in a2as:
                chart_entry_a2a = {
                    "data": [], "label": "Input Data", "type": "line"
                }
                if attribute not in input_a2a_data_:
                    continue
                a2a_datum = input_a2a_data_[attribute]
                for i in range(len(axes)):
                    chart_entry_a2a["data"].append(a2a_datum)
                chart_data[a2as_pretty[attribute]].append(chart_entry_a2a)

        elif stats_type == "cycletimes" or stats_type == "earthmovers":
            chart_data = {otype: []}
            attributes = [otype]
            for model in self.selectedObjectModels:
                data_ = self.simulationRunEvaluations[model][otype].get(stats_type)
                datum0 = 0.0
                # datum0 = "00:00:00:00" if stats_type == "cycletimes" else 0.0
                chart_entry = {
                    "data": [datum0],
                    "label": model,
                    "type": "line"
                }
                for i in range(1, len(axes)):
                    tick = axes[i]
                    if tick not in data_:
                        chart_entry["data"].append(chart_entry["data"][i - 1])
                        continue
                    # chart_entry["data"].append(self.__int_to_timedelta_str(data_[tick]))
                    chart_entry["data"].append(data_[tick])
                chart_data[otype].append(chart_entry)
            if stats_type == "cycletimes":
                input_data_ = self.originalCycleTimes
                chart_entry = {
                    "data": [], "label": "Input Data", "type": "line"
                }
                datum = input_data_[otype]
                for i in range(len(axes)):
                    chart_entry["data"].append(datum)
                chart_data[otype].append(chart_entry)
        else:
            raise AttributeError()
        axes = {attribute: [str(i) for i in axes] for attribute in attributes}
        response = {"axes": axes, "chartData": chart_data}
        return response

    def __make_step_ticks(self):
        self.stepTicks = {}
        logs_path = os.path.join(self.sessionPath, "simulated_logs")
        for model in self.selectedObjectModels:
            path = os.path.join(logs_path, model)
            path = os.path.join(path, "simulated_ocel.jsonocel")
            log = pm4py.read_ocel(path)
            frame = log.get_extended_table()
            max_step = max(map(lambda step: int(step), frame["STEPS"].values))
            # TODO
            #granularity = 10.0 if max_step < 250 else 100.0
            tick_width =  round((float(max_step) / float(self.GRID)))
            ticks = [0]
            for i in range(1, self.GRID):
                ticks.append(round(i * tick_width))
            # ticks.append(max_step)
            self.stepTicks[model] = ticks

    def __compute_emd_conformance(self, otype, model, simulated_flog, step_ticks):
        simulated_flog_formatted = pm4py.convert_to_event_log(simulated_flog)
        original_language = self.originalLanguages[otype]
        emconfs = {}
        if len(step_ticks) == 0:
            raise AttributeError()
        for tick in [step_ticks[1]] + [step_ticks[-1]]:
            # for tick in step_ticks[1:]:
            log_prefix = pm4py.filter_event_attribute_values(simulated_flog_formatted, "STEPS", [str(i) for i in range(tick)])
            simulated_language = variants_module.get_language(log_prefix)
            emdist = emd_evaluator.apply(simulated_language, original_language)
            emconf = 1 - emdist
            emconfs[tick] = emconf
        return emconfs

    def __load_flog_and_frame(self, original: bool, otype: str, model=""):
        if original:
            flog_path = os.path.join(self.sessionPath, "flattened_" + otype + ".xes")
        else:
            flog_fname = 'flattened_' + otype + '_simulated.xes'
            flog_path = os.path.join(self.sessionPath, "simulated_logs")
            flog_path = os.path.join(flog_path, model)
            flog_path = os.path.join(flog_path, flog_fname)
        flog = pm4py.read_xes(flog_path)
        frame = pm4py.convert_to_dataframe(flog)
        frame = frame.sort_values(["case:concept:name", "time:timestamp"])
        return flog, frame

    def __load_original_data(self):
        original_ocel_path = os.path.join(self.sessionPath, "postprocessed_input.jsonocel")
        self.originalOcel = pm4py.read_ocel(original_ocel_path)
        original_flat_logs = {}
        original_flat_frames = {}
        original_languages = {}
        for otype in self.otypes:
            original_flog, original_frame = self.__load_flog_and_frame(original=True, otype=otype)
            original_flat_logs[otype] = original_flog
            original_flat_frames[otype] = original_frame
            original_flog_formatted = pm4py.convert_to_event_log(original_flog)
            original_language = variants_module.get_language(original_flog_formatted)
            original_languages[otype] = original_language
        self.originalFlatLogs = original_flat_logs
        self.originalFlatLogDataFrames = original_flat_frames
        self.originalLanguages = original_languages

    def evaluate(self):
        self.__load_original_data()
        self.originalActivityDelays = dict()
        self.originalA2ADelays = dict()
        self.originalCycleTimes = dict()
        for otype in self.otypes:
            frame = self.originalFlatLogDataFrames[otype]
            activity_delay_stats, a2a_delay_stats, cycle_times_stats = self.__compute_delays(frame, otype)
            self.originalActivityDelays[otype] = activity_delay_stats
            self.originalA2ADelays[otype] = a2a_delay_stats
            self.originalCycleTimes[otype] = cycle_times_stats
        model_evals = {}
        self.__make_step_ticks()
        for model in self.selectedObjectModels:
            step_ticks = self.stepTicks[model]
            otype_evals = {}
            for otype in self.otypes:
                flog, frame = self.__load_flog_and_frame(original=False, otype=otype, model=model)
                act_delays, a2a_delays, cycle_times = self.__compute_delays(frame, otype, step_ticks)
                emconf = self.__compute_emd_conformance(otype, model, flog, step_ticks)
                otype_eval = SimulationRunEvaluation(otype, act_delays, a2a_delays, cycle_times, emconf)
                otype_evals[otype] = otype_eval
            model_evals[model] = otype_evals
        self.simulationRunEvaluations = model_evals

    def __compute_delays(self, log, otype, ticks=None):
        acts = set(log["concept:name"].values)
        activity_delays = {
            act: []
            for act in acts
        }
        a2a_delays = {
            (act1, act2): []
            for act1 in acts for act2 in acts
        }
        cycle_times = []
        log["delay"] = 0
        log["int:timestamp"] = log.apply(lambda row: int(row["time:timestamp"].timestamp()), axis=1)
        if ticks is None:
            log["STEPS"] = -1
        else:
            ticks = sorted(ticks)
        iterator = log.iterrows()
        index, line = next(iterator, None)
        lastline, lastindex = line, index
        steps = int(line["STEPS"])
        firsttime = line["int:timestamp"]
        nextline = next(iterator, None)
        laststeps = steps
        lastcase = lastline["case:concept:name"]
        lastact = lastline["concept:name"]
        lasttime = lastline["int:timestamp"]
        while nextline is not None:
            index, line = nextline
            act = line["concept:name"]
            case = line["case:concept:name"]
            time = line["int:timestamp"]
            steps = int(line["STEPS"])
            if case == lastcase:
                # Update DELAY
                delay = time - lasttime
                log.at[lastindex, 'delay'] = delay
                activity_delays[lastact].append((delay, laststeps))
                a2a_delays[(lastact, act)].append((delay, laststeps))
            else:
                activity_delays[lastact].append((0, laststeps))
                a2a_delays[(lastact, act)].append((0, laststeps))
                cycle_time = lasttime - firsttime
                cycle_times.append((cycle_time, steps))
                firsttime = line["int:timestamp"]
            lastline, lastindex = line, index
            lastact = act
            lastcase = case
            laststeps = steps
            lasttime = time
            nextline = next(iterator, None)
        a2a_support = [a2a for a2a, vals in a2a_delays.items() if len(vals) > 0]
        self.a2aSupport[otype] = list(set(self.a2aSupport[otype] + a2a_support))
        a2a_delays = {a2a: vals for a2a, vals in a2a_delays.items() if a2a in a2a_support}
        if ticks is None:
            activity_delay_stats = self.__make_delay_data(activity_delays)
            a2a_delay_stats = self.__make_delay_data(a2a_delays)
            cycle_times_stats = np.mean(list(map(lambda cycletime_and_step: cycletime_and_step[0], cycle_times)))
            return activity_delay_stats, a2a_delay_stats, cycle_times_stats
        activity_delay_stats = self.__make_delay_data(activity_delays, ticks)
        a2a_delay_stats = self.__make_delay_data(a2a_delays, ticks)
        cycle_times_stats_temp = {
            0: (0, 0)
        }
        ### now use ticks to project frame and incrementally compute mean
        for i in range(len(ticks) - 1):
            from_tick = ticks[i]
            to_tick = ticks[i + 1]
            cycle_data = filter(lambda datum: from_tick < datum[1] <= to_tick, cycle_times)
            cycles_only = list(map(lambda datum: datum[0], cycle_data))
            nof_cycles_delta = len(cycles_only)
            cycle_mean_total, nof_cycles = cycle_times_stats_temp[from_tick]
            if nof_cycles_delta == 0:
                cycle_times_stats_temp[to_tick] = cycle_mean_total, nof_cycles
            else:
                new_nof_cycles = nof_cycles + nof_cycles_delta
                cycle_mean_delta = np.mean(cycles_only)
                new_cycle_mean = (cycle_mean_total * nof_cycles + nof_cycles_delta * cycle_mean_delta) / new_nof_cycles
                cycle_times_stats_temp[to_tick] = (new_cycle_mean, new_nof_cycles)

        cycle_times_stats = {
            tick: cycle_time
            for tick, (cycle_time, counter) in cycle_times_stats_temp.items()
        }
        return activity_delay_stats, a2a_delay_stats, cycle_times_stats

    def __make_delay_data(self, delay_dict, ticks=None):
        if ticks is None:
            delay_stats = {
                act: np.mean(list(map(lambda delay_and_step: delay_and_step[0], delays)))
                for act, delays in delay_dict.items()
            }
            return delay_stats
        delay_stats_temp = {
            0: {
                key: (0, 0)
                for key in delay_dict.keys()
            }
        }
        for i in range(len(ticks) - 1):
            from_tick = ticks[i]
            to_tick = ticks[i + 1]
            delay_stats_temp[to_tick] = {
                key: (0, 0)
                for key in delay_stats_temp[from_tick].keys()
            }
            delay_data = {
                act: filter(lambda datum: from_tick < datum[1] <= to_tick, delays)
                for act, delays in delay_dict.items()
            }
            delays_only = {
                act: list(map(lambda datum: datum[0], data))
                for act, data in delay_data.items()
            }
            for key, delays in delays_only.items():
                nof_delays_delta = len(delays)
                delay_mean_total, nof_delays = delay_stats_temp[from_tick][key]
                if nof_delays_delta == 0:
                    delay_stats_temp[to_tick][key] = delay_mean_total, nof_delays
                else:
                    new_nof_delays = nof_delays + nof_delays_delta
                    delay_mean_delta = np.mean(delays)
                    new_delay_mean = (
                                                 delay_mean_total * nof_delays + nof_delays_delta * delay_mean_delta) / new_nof_delays
                    delay_stats_temp[to_tick][key] = (new_delay_mean, new_nof_delays)
            delay_stats = {
                tick: {
                    act: delay
                    for act, (delay, counter) in activity_delays.items()
                }
                for tick, activity_delays in delay_stats_temp.items()
            }
            return delay_stats

    # TODO: safety for days with 3 digits and more beauty
    def __int_to_timedelta_str(self, int_timedelta_in_seconds):
        td = timedelta(seconds=int_timedelta_in_seconds)
        str_td = str(td)
        if str_td.find(".") > 0:
            str_td = str_td.split(".")[0]
        parts = str_td.split(" days, ")
        if len(parts) == 1:
            parts = parts[0].split(" day, ")
            if len(parts) == 1:
                days = "0"
                hours, minutes, seconds = parts[0].split(":")
            else:
                days = "1"
                hours, minutes, seconds = parts[1].split(":")
        else:
            days = parts[0]
            hours, minutes, seconds = parts[1].split(":")
        if int(days) < 10:
            days = "0" + days
        if int(hours) < 10:
            hours = "0" + hours
        if int(minutes) < 10:
            minutes = "0" + minutes
        if int(seconds) < 10:
            seconds = "0" + seconds
        return days + ":" + hours + ":" + minutes + ":" + seconds
