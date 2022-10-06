import math

import pandas as pd
import scipy.integrate as integrate
from scipy import stats


class SupportDistribution:
    __pdf: any
    cdf_steps: dict

    def __init__(self, path, otype_schemata):
        self.__initialized = False
        self.path = path
        self.otype_schemata = otype_schemata
        self.__make_prior_distribution()
        self.__initialized = True

    def __make_prior_distribution(self):
        items = [(card, freq) for card, freq in self.otype_schemata.items()]
        card_frequencies = pd.DataFrame(items, columns=["cardinality", "frequency"])
        total = sum(card_frequencies["frequency"])
        mean = sum(card_frequencies["cardinality"] * card_frequencies["frequency"]) / total
        card_frequencies["p"] = card_frequencies["frequency"] / total
        variance = sum((card_frequencies["cardinality"] - mean).pow(2) * card_frequencies["p"])
        stdev = math.sqrt(variance)
        self.variance = variance
        if variance < 0.1:
            self.support_steps = self.__make_support_steps_for_single_value(mean)
            return
        self.support_steps = self.__make_support_steps_by_schema_frequencies(card_frequencies)
        # pdf = self.__get_pdf(mean, variance)
        # self.support_steps = self.__make_support_steps(pdf, mean, stdev, schema_frequencies)

    def __get_pdf(self, mean, variance):
        factor = 1 / (math.sqrt(2 * math.pi * variance))
        return lambda x: factor * math.exp(-math.pow(x - mean, 2) / (2 * variance))

    def __make_support_steps(self, pdf, mean, schema_frequencies):
        steps = [i for i in range(round(mean))]
        # probability as integral over pdf
        prob_fun = lambda step: integrate.quad(pdf, step - 0.5, step + 0.5)[0]
        probs = []
        for step in steps:
            probs.append(prob_fun(step))
        # go on computing probabilities until threshold met
        i = steps[-1]
        while probs[i] > 0.00001:
            i = i + 1
            p = prob_fun(i)
            probs.append(p)
            steps.append(i)
        if len(schema_frequencies[schema_frequencies["schema"] == 0]) == 0:
            probs[0] = 0.0
        probs = list(map(lambda prob: prob / sum(probs), probs))
        support_product = 1
        support_steps = [1.0]
        for i in steps[1:]:
            pi_1 = probs[i - 1]
            support_i = 1 - pi_1 / support_product
            support_product = support_i * support_product
            support_steps.append(support_i)
        return support_steps

    def __make_support_steps_by_schema_frequencies(self, schema_frequencies):
        # index 0: the support of having 0 objects is 1
        schemata = schema_frequencies["cardinality"].values
        ps = [0.0] * (max(schemata) + 1)
        support_steps = [1.0] + [0] * (max(schemata) + 1)
        support_product = 1
        xs = []
        for i in [i for i in range(len(ps))]:
            x = schema_frequencies[schema_frequencies["cardinality"] == i]
            if len(x) == 0:
                pi = 0
            else:
                pi = float(x["p"])
            ps[i] = pi
            support_i1 = 1 - pi / support_product
            support_product = support_i1 * support_product
            support_steps[i + 1] = support_i1
        return support_steps

    def __make_support_steps_for_single_value(self, mean):
        support_steps = {i: 1 for i in range(round(mean - 0.1) + 1)}
        support_steps[round(mean) + 1] = 0
        return support_steps

    def has_variance_below(self, threshold):
        return self.variance < threshold

    def get_support(self, x):
        if x >= len(self.support_steps):
            return 0.0
        return self.support_steps[x]
