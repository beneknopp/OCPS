import math
import os
import pickle
from enum import Enum

import pandas as pd
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm, poisson


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
    POISSON = "POISSON"


class Modeler():

    parameters: str
    modelType: ModelType

    def __init__(self, model_type: ModelType):
        self.modelType = model_type
        self.parameters = ""

    def fit_data(self, data: []):
        if self.modelType == ModelType.CUSTOM:
            self.__fit_custom(data)
        elif self.modelType == ModelType.NORMAL:
            self.__fit_normal(data)
        elif self.modelType == ModelType.POISSON:
            self.__fit_poisson(data)
        else:
            raise AttributeError()
        return self.parameters

    def __fit_custom(self, data):
        sorted_vals = sorted(list(set(data)))
        parameters = ""
        n = len(data)
        for val in sorted_vals:
            ratio = float(data.count(val))/n
            rounded_ratio = round(ratio*100) / 100
            parameters += str(val) + ": " + str(ratio) + "; "
        self.parameters = parameters[:-2]

    def __fit_normal(self, data):
        mu, std = norm.fit(data)
        self.parameters = "mu: " + str(mu) + "; std: " + str(std)

    def __fit_poisson(self, data):
        minval = math.floor(min(data))
        maxval = math.ceil(max(data))
        # the bins have to be kept as a positive integer because poisson is a positive integer distribution
        bins = np.arange(minval, maxval) - 0.5
        entries, bin_edges, patches = plt.hist(data, bins=bins, density=True, label='Data')
        # calculate bin centers
        middles_bins = (bin_edges[1:] + bin_edges[:-1]) * 0.5
        parameters, cov_matrix = curve_fit(lambda k, lamb: poisson.pmf(k, lamb), middles_bins, entries)
        self.parameters = "lambda: " + str(parameters[0])

    def map_axis(self, ticks):
        if self.modelType == ModelType.CUSTOM:
            mapped_ticks = self.__map_custom(ticks)
        elif self.modelType == ModelType.NORMAL:
            mapped_ticks = self.__map_normal(ticks)
        elif self.modelType == ModelType.POISSON:
            mapped_ticks = self.__map_poisson(ticks)
        else:
            raise AttributeError()
        return mapped_ticks

    def __map_custom(self, ticks):
        mapped_ticks = []
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

    def __map_poisson(self, ticks):
        if any(x < 0 for x in ticks):
            raise AttributeError("Invalid arguments for Poisson distribution")
        mapped_ticks = []
        lamb = float(self.parameters.split(";")[0].split(": ")[1])
        for tick in ticks:
            val = poisson(lamb).pmf(tick)
            mapped_ticks.append(val)
        return mapped_ticks


class AttributeParameterization():

    label: str
    # final (log data)
    modeler: Modeler
    parameterType: ParameterType
    includeModeled: bool
    includeSimulated: bool
    # log_based statistics & model curve (if included) & simulated curve (if included)
    xAxis: []
    yAxes: dict
    data: []

    def __init__(self, label, data, parameter_type: ParameterType, include_modeled=False, include_simulated=False):
        self.data = data
        self.label = label
        self.parameterType = parameter_type
        self.includeModeled = include_modeled
        self.includeSimulated = include_simulated
        self.__initialize_modeler(ModelType.CUSTOM)
        self.__initialize_chart_data()

    def __initialize_modeler(self, model_type):
        modeler = Modeler(model_type)
        modeler.fit_data(self.data)
        self.modeler = modeler

    def __initialize_chart_data(self):
        yAxes = {}
        data = self.data
        if len(data) == 0:
            raise ValueError("Empty list of attribute values")
        if type(data[0]) == str:
            x_axis = list(set(data))
        else:
            min_val = math.floor(min(data))
            max_val = math.ceil(max(data)) + 1
            x_axis = range(min_val, max_val + 1)
        mapped_x_axis = self.modeler.map_axis(x_axis)
        yAxes[ParameterMode.LOG_BASED] = mapped_x_axis
        self.xAxis = x_axis
        self.yAxes = yAxes
        self.__set_model_chart_data()

    def __set_model_chart_data(self):
        mapped_x_axis = self.modeler.map_axis(self.xAxis)
        self.yAxes[ParameterMode.MODELED] = mapped_x_axis

    def export(self):
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
        return json_export

    def switch_model_type(self, model_type):
        data = self.data
        modeler = Modeler(model_type)
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
        for otype, path_to_cards in cardinality_dists.items():
            for path, data in path_to_cards.items():
                attr_par = AttributeParameterization(label=path, data=data, parameter_type=cardinality_type)
                parameters[otype][cardinality_type][path] = attr_par
        for otype, attr_data in object_attribute_dists.items():
            for attr, data in attr_data.items():
                attr_par = AttributeParameterization(label=attr, data=data, parameter_type=object_attribute_type)
                parameters[otype][object_attribute_type][attr] = attr_par
        self.parameters = parameters

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