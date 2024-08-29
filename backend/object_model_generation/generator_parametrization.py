import math
import os
import pickle
from enum import Enum

import numpy as np
from scipy import stats
from scipy.stats import norm, poisson

from utils.cumulative_distribution import CumulativeDistribution


class ParameterMode(Enum):
    LOG_BASED = "LOG_BASED"
    MODELED = "MODELED"
    SIMULATED = "SIMULATED"


class ParameterType(Enum):
    CARDINALITY = "CARDINALITY"
    OBJECT_ATTRIBUTE = "OBJECT_ATTRIBUTE"
    TIMING = "TIMING"


class ModelType(Enum):
    CUSTOM = "CUSTOM"
    NORMAL = "NORMAL"
    #UNIFORM = "UNIFORM" maybe later
    EXPONENTIAL = "EXPONENTIAL"

class DataType(Enum):
    INTEGER = "INTEGER"
    CONTINUOUS = "CONTINUOUS"
    CATEGORICAL = "CATEGORICAL"

class Predictor:

    modelType: ModelType
    dataType: DataType

    def __init__(self, data_type: DataType):
        self.dataType = data_type
        pass

    def init_custom(self, weighted_items: dict):
        self.modelType = ModelType.CUSTOM
        pmf: CumulativeDistribution = CumulativeDistribution(weighted_items)
        self.pmf = pmf

    def init_normal(self, mu, std):
        self.modelType = ModelType.NORMAL
        self.mu = mu
        self.std = std

    def init_exponential(self, scale, loc):
        self.modelType = ModelType.EXPONENTIAL
        self.scale = scale
        self.loc = loc

    def sample(self):
        if self.modelType == ModelType.CUSTOM:
            pmf: CumulativeDistribution = self.pmf
            return pmf.sample()
        if self.modelType == ModelType.NORMAL:
            s = -1
            i = 0
            while s < 0:
                if i > 100000:
                    raise ValueError("Timeout in sampling from distribution.")
                s = np.random.normal(self.mu, self.std, 1)[0]
                if self.dataType != DataType.INTEGER:
                    break
                i += 1
            return s
        if self.modelType == ModelType.EXPONENTIAL:
            distr = stats.expon(self.scale, self.loc)
            s = distr.rvs(size=1)[0]
            return s

class Modeler():

    parameters: str
    modelType: ModelType
    dataType: DataType
    predictor: Predictor
    bins: int

    def __init__(self, model_type: ModelType, data_type: DataType):
        self.modelType = model_type
        self.dataType = data_type
        self.parameters = ""

    def fit_data(self, data: []):
        if self.modelType == ModelType.CUSTOM:
            self.__fit_custom(data)
        elif self.modelType == ModelType.NORMAL:
            self.__fit_normal(data)
        elif self.modelType == ModelType.EXPONENTIAL:
            self.__fit_exponential(data)
        else:
            raise AttributeError()
        return self.parameters

    def __fit_custom(self, data):
        sorted_vals = sorted(list(set(data)))
        parameters = ""
        n = len(data)
        if self.dataType == DataType.CONTINUOUS:
            parameters += "bins=50; "
            bins = 50
            self.bins = bins
            data = sorted(data)
            maxvalue = max(data)
            minvalue = min(data)
            i = 0
            j = 0
            try:
                w = (float(maxvalue) - float(minvalue)) / bins
            except:
                print("")
            binmax = float(minvalue) + w
            weighted_vals = {
                float(minvalue) + round((k+1)*w*100)/100 : 0
                for k in range(bins)
            }
            bin_centers = sorted(list(weighted_vals.keys()))
            while i < n:
                current_value = float(data[i])
                while binmax < current_value - 0.001:
                    binmax += w
                    j += 1
                bin_center = bin_centers[j]
                weighted_vals[bin_center] = weighted_vals[bin_center] + 1
                i += 1
            for val, freq in weighted_vals.items():
                ratio = float(freq)/n
                rounded_ratio = round(ratio * 100) / 100
                parameters += str(float(val)) + ": " + str(rounded_ratio) + "; "
        else:
            weighted_vals = {}
            for val in sorted_vals:
                ratio = float(data.count(val))/n
                rounded_ratio = round(ratio*100) / 100
                parameters += str(val) + ": " + str(rounded_ratio) + "; "
                weighted_vals[val] = ratio
        self.defaultAxis = sorted(list(weighted_vals.keys()))
        self.predictor = Predictor(self.dataType)
        from collections import Counter
        self.predictor.init_custom(
            Counter(sorted(data))
        )
        self.parameters = parameters[:-2]

    def __fit_normal(self, data):
        mu, std = norm.fit(data)
        self.predictor = Predictor(self.dataType)
        self.predictor.init_normal(mu, std)
        self.parameters = "mu: " + str(mu) + "; std: " + str(std)

    def __fit_exponential(self, data):
        params = stats.expon.fit(data)
        scale, loc = params
        self.predictor = Predictor(self.dataType)
        self.predictor.init_exponential(scale, loc)
        self.parameters = "scale: " + str(scale) + "; loc: " + str(loc)

    def map_axis(self, ticks=None):
        if self.modelType == ModelType.CUSTOM:
            mapped_ticks = self.__map_custom(ticks)
        elif self.modelType == ModelType.NORMAL:
            mapped_ticks = self.__map_normal(ticks)
        elif self.modelType == ModelType.EXPONENTIAL:
            mapped_ticks = self.__map_exponential(ticks)
        else:
            raise AttributeError()
        return mapped_ticks

    def __map_custom(self, ticks):
        mapped_ticks = []
        if self.dataType == DataType.CONTINUOUS:
            parameters = self.parameters.split("; ")
            bins = int(parameters[0].split("=")[1])
            val_freqs = [valfreq.split(": ") for valfreq in parameters[1:]]
        else:
            val_freqs = [valfreq.split(": ") for valfreq in self.parameters.split("; ")]
        vals_to_freqs = {(valfreq[0]) : float(valfreq[1]) for valfreq in val_freqs}
        for tick in ticks:
            if str(tick) in vals_to_freqs:
                mapped_ticks.append(vals_to_freqs[str(tick)])
            else:
                mapped_ticks.append(0)
        return mapped_ticks

    def __map_normal(self, ticks):
        mapped_ticks = []
        mu, std = [float(param.split(": ")[1]) for param in self.parameters.split(";")]
        for tick in ticks:
            val = norm(mu, std).pdf(tick)
            mapped_ticks.append(val)
        return mapped_ticks

    def __map_uniform(self, ticks):
        mapped_ticks = []
        mu, std = [float(param.split(": ")[1]) for param in self.parameters.split(";")]
        for tick in ticks:
            val = norm(mu, std).pdf(tick)
            mapped_ticks.append(val)
        return mapped_ticks

    def __map_exponential(self, ticks):
        w = ticks[1] - ticks[0]
        bin_edges = [tick - w/2 for tick in ticks]
        bin_edges += [ticks[-1] + w/2]
        scale, loc = [float(param.split(": ")[1]) for param in self.parameters.split(";")]
        exponential_dist = stats.expon(scale, loc)
        cdfs = exponential_dist.cdf(bin_edges)
        mapped_ticks = [cdfs[i+1] - cdfs[i] for i in range(len(ticks))]
        return mapped_ticks

    def draw(self):
        return self.predictor.sample()

class AttributeParameterization():

    label: str
    dataType: DataType
    # final (log data)
    modeler: Modeler
    parameterType: ParameterType
    includeModeled: bool
    includeSimulated: bool
    fittingModel: str = "---"
    # only for arrival rates
    markedAsBatchArrival: bool
    # log_based statistics & model curve (if included) & simulated curve (if included)
    xAxis: []
    yAxes: dict
    data: []

    def __init__(self, label, data, parameter_type: ParameterType, include_modeled=False, include_simulated=False,
                 data_type: DataType = None, marked_as_batch_arrival=None):
        self.data = data
        self.label = label
        self.parameterType = parameter_type
        self.includeModeled = include_modeled
        self.includeSimulated = include_simulated
        if marked_as_batch_arrival is not None:
            self.markedAsBatchArrival = marked_as_batch_arrival
        self.__initialize_modeler(data_type)
        self.__initialize_chart_data()

    def __initialize_modeler(self, data_type: DataType):
        model_type = self.__fit_initial_model_type()
        if data_type is None:
            data_type = self.__fit_data_type()
        self.dataType = data_type
        modeler = Modeler(model_type, data_type)
        modeler.fit_data(self.data)
        self.modeler = modeler
        self.fittingModel = modeler.modelType

    def __fit_initial_model_type(self):
        return ModelType.CUSTOM

    def __fit_data_type(self):
        if all(type(i) == int for i in self.data):
            return DataType.INTEGER
        else:
            isFloat = True
            for i in self.data:
                try:
                    float(i)
                except:
                    isFloat = False
                if not isFloat:
                    break
            if isFloat:
                return DataType.CONTINUOUS
        return DataType.CATEGORICAL

    def __initialize_chart_data(self):
        data = self.data
        if len(data) == 0:
            raise ValueError("Empty list of attribute values")
        if self.dataType is DataType.CATEGORICAL:
            x_axis = list(set(data))
        elif self.dataType is DataType.CONTINUOUS:
            x_axis = self.__make_x_axis_continuous(data)
        else:
            min_val = math.floor(min(data))
            max_val = math.ceil(max(data)) + 1
            x_axis = range(min_val, max_val + 1)
        self.xAxis = x_axis
        self.yAxes = {}
        self.__set_log_based_chart_data()
        self.__set_model_chart_data()

    def __set_log_based_chart_data(self):
        if self.dataType is DataType.CONTINUOUS:
            mapped_x_axis = self.__map_x_axis_continuous()
        else:
            vals = set(self.data)
            total = len(self.data)
            mapped_x_axis = []
            if not self.dataType == DataType.CONTINUOUS:
                freqs = {val: self.data.count(val) for val in vals}
                for tick in self.xAxis:
                    if tick in vals:
                        mapped_x_axis.append(float(freqs[tick]) / total)
                    else:
                        mapped_x_axis.append(0.0)
        self.yAxes[ParameterMode.LOG_BASED] = mapped_x_axis

    def __make_x_axis_continuous(self, data):
        min_val = math.floor(float(min(data)))
        max_val = math.ceil(float(max(data)))
        W = max_val - min_val
        w = W / 20
        self.xAxisTickWidth = w
        ticks = [round(100*(min_val + (2*k+1)*w/2))/100 for k in range(20) ]
        return ticks

    def __map_x_axis_continuous(self):
        vals = self.data
        ticks = self.xAxis
        nof_ticks = len(ticks)
        nof_vals = len(vals)
        bin_counts = [0 for i in ticks]
        for val_ in vals:
            val = float(val_)
            i = 0
            while i < nof_ticks and ticks[i] < val:
                i = i + 1
            if i == 0:
                bin_counts[i] = bin_counts[i] + 1
            elif i == nof_ticks:
                bin_counts[-1] = bin_counts[-1] + 1
            else:
                if abs(val - ticks[i-1]) < abs(val - ticks[i]):
                    bin_counts[i-1] = bin_counts[i-1] + 1
                else:
                    bin_counts[i] = bin_counts[i] + 1
        mapped_x_axis = [float(x) / nof_vals for x in bin_counts]
        return mapped_x_axis

    def __set_model_chart_data(self):
        mapped_x_axis = self.modeler.map_axis(self.xAxis)
        self.yAxes[ParameterMode.MODELED] = mapped_x_axis

    def export(self):
        fitting_model = self.fittingModel
        if fitting_model == ModelType.CUSTOM:
            fitting_model = "Custom"
        elif fitting_model == ModelType.NORMAL:
            fitting_model = "Normal"
        elif fitting_model == ModelType.EXPONENTIAL:
            fitting_model = "Exponential"
        json_export = {}
        #json_export["label"] = self.label
        json_export["xAxis"] = [str(i) for i in self.xAxis]
        json_export["yAxes"] = {
            mode.value: y_axis#[str(i) for i in y_axis]
            for mode, y_axis in self.yAxes.items()
        }
        json_export["includeModeled"] = self.includeModeled
        json_export["includeSimulated"] = self.includeSimulated
        json_export["parameters"] = self.modeler.parameters
        json_export["fittingModel"] = fitting_model
        return json_export

    def switch_model_type(self, model_type):
        data = self.data
        data_type = self.modeler.dataType
        modeler = Modeler(model_type, data_type)
        modeler.fit_data(data)
        self.modeler = modeler
        self.__set_model_chart_data()

    def change_parameters(self, parameters_str):
        self.modeler.parameters = parameters_str
        self.__set_model_chart_data()

    def get_modeled_frequency_distribution(self):
        # TODO: adjust ticks/x-axis to comprise modeled distribution
        dist = {
            self.xAxis[i]: self.yAxes[ParameterMode.MODELED][i]
            for i in range(len(self.xAxis))
        }
        return dist

    def update_XAxis(self, new_ticks):
        min_tick = min(min(self.xAxis), min(new_ticks))
        max_tick = max(max(self.xAxis), max((new_ticks)))
        self.xAxis = range(min_tick, max_tick+1)
        self.__set_log_based_chart_data()
        self.__set_model_chart_data()

    def update_simulated_data(self, simulated_data):
        values = list(simulated_data.keys())
        total = sum(simulated_data.values())
        if len(values) == 0:
            raise ValueError("Empty list of attribute values")
        mapped_x_axis =[]
        self.update_XAxis(simulated_data.keys())
        for tick in self.xAxis:
            if tick in values:
                mapped_x_axis.append(float(simulated_data[tick])/total)
            else:
                mapped_x_axis.append(0)
        self.yAxes[ParameterMode.SIMULATED] = mapped_x_axis
        self.includeSimulated = True

    def draw(self):
        return self.modeler.draw()

class GeneratorParametrization():

    parameters: dict
    otypes: []

    @classmethod
    def load(cls, session_path):
        gen_par_path = os.path.join(session_path, "generator_parametrization.pkl")
        return pickle.load(open(gen_par_path, "rb"))

    def __init__(self, otypes, cardinality_dists, object_attribute_dists, timing_dists):
        self.otypes = otypes
        self.__initialize_parameters(cardinality_dists, object_attribute_dists, timing_dists)

    def save(self, session_path):
        gen_par_path = os.path.join(session_path, "generator_parametrization.pkl")
        with open(gen_par_path, "wb") as write_file:
            pickle.dump(self, write_file)

    def switch_model_type(self, otype: str, parameter_type_str: str, attribute: str, model_type: ModelType):
        parameter_type = ParameterType(parameter_type_str)
        attr_params: AttributeParameterization = self.parameters[otype][parameter_type][attribute]
        attr_params.switch_model_type(model_type)

    def __initialize_parameters(self, cardinality_dists, object_attribute_dists, timing_dists):
        parameters = {
            otype: {
                parameter_type: dict()
                for parameter_type in ParameterType
            }
            for otype in self.otypes
        }
        cardinality_type = ParameterType.CARDINALITY
        object_attribute_type = ParameterType.OBJECT_ATTRIBUTE
        timing_type = ParameterType.TIMING
        for otype in self.otypes:
            for path_to_cards in cardinality_dists[otype].items():
                path, data = path_to_cards
                include_modeled = len(path.split(",")) < 3
                attr_par = AttributeParameterization(
                    label=path, data=data, parameter_type=cardinality_type, include_modeled=include_modeled,
                    data_type = DataType.INTEGER)
                parameters[otype][cardinality_type][path] = attr_par
            for attr_data in object_attribute_dists[otype].items():
                    attr, data = attr_data
                    include_modeled = True
                    attr_par = AttributeParameterization(
                        label=attr, data=data, parameter_type=object_attribute_type, include_modeled=include_modeled
                    )
                    parameters[otype][object_attribute_type][attr] = attr_par
            for timing_data in timing_dists[otype].items():
                    attr, data = timing_data
                    include_modeled = attr == "Arrival Rates (independent)"
                    attr_par = AttributeParameterization(
                        label=attr, data=data, parameter_type=timing_type, include_modeled=include_modeled,
                        data_type=DataType.CONTINUOUS, marked_as_batch_arrival=False)
                    parameters[otype][timing_type][attr] = attr_par
        self.parameters = parameters

    def get_parameters(self, otype = "", parameter_type = "", attribute = ""):
        if len(otype) == 0:
            return self.parameters
        if len(parameter_type) == 0:
            return self.parameters[otype]
        if len(attribute) == 0:
            return self.parameters[otype][ParameterType(parameter_type)]
        return self.parameters[otype][ParameterType(parameter_type)][attribute]

    def export_parameters(self, otype: str, parameter_type_str: str, attribute: str = ""):
        parameter_type = ParameterType(parameter_type_str)
        parameters = self.parameters[otype][parameter_type]
        attr_par: AttributeParameterization
        if len(attribute) > 0:
            attr_par = parameters[attribute]
            return attr_par.export()
        parameters_export = {
            attr: attr_par.export()
            for attr, attr_par in parameters.items()
        }
        return parameters_export

    def select_for_training(self, otype: str, parameter_type_str: str, attribute: str, selected: bool):
        parameter_type = ParameterType(parameter_type_str)
        attr_par: AttributeParameterization = self.parameters[otype][parameter_type][attribute]
        attr_par.includeModeled = selected

    def mark_as_batch_arrival(self, otype:str, attribute: str, selected: bool):
        parameter_type = ParameterType.TIMING
        attr_par: AttributeParameterization = self.parameters[otype][parameter_type][attribute]
        attr_par.markedAsBatchArrival = selected

    def switch_fitting_model(self, otype: str, parameter_type_str: str, attribute: str, fitting_model_str: str):
        parameter_type = ParameterType(parameter_type_str)
        model_type = ModelType(fitting_model_str)
        attr_par: AttributeParameterization = self.parameters[otype][parameter_type][attribute]
        attr_par.switch_model_type(model_type)
        self.parameters[otype][parameter_type][attribute] = attr_par

    def change_parameters(self, otype: str, parameter_type_str: str, attribute: str, parameters_str: str):
        parameter_type = ParameterType(parameter_type_str)
        attr_par: AttributeParameterization = self.parameters[otype][parameter_type][attribute]
        attr_par.change_parameters(parameters_str)
        self.parameters[otype][parameter_type][attribute] = attr_par
