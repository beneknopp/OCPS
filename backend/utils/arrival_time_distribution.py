import math

import numpy as np
from scipy import integrate
from scipy import stats

from utils.cumulative_distribution import CumulativeDistribution


class ArrivalTimeDistribution:
    GRANULARITY = 10000

    def __init__(self, arrival_times):
        self.__arrival_times = arrival_times
        self.mean = arrival_times.mean()
        self.variance = arrival_times.var()
        self.stdev = math.sqrt(self.variance)
        self.__make_distribution()

    def __make_distribution(self):
        self.__distribution = lambda x: round(self.mean)

    def __get_pdf(self):
        a, loc, scale = stats.skewnorm.fit(self.__arrival_times)
        skewnorm = stats.skewnorm(a, loc, scale)
        pdf = skewnorm.pdf
        return pdf

    def sample(self):
        return self.__distribution(1)
