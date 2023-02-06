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
        self.stdev = math.sqrt(self.variance) if self.variance > 0 else 0.0
        self.__make_distribution()

    def __make_distribution(self):
        #self.__distribution = lambda x: round(self.mean)
        cdist = CumulativeDistribution({time: 1 for time in self.__arrival_times.values})
        self.__distribution = lambda x: cdist.sample()

    def __get_pdf(self):
        a, loc, scale = stats.skewnorm.fit(self.__arrival_times)
        skewnorm = stats.skewnorm(a, loc, scale)
        pdf = skewnorm.pdf
        return pdf

    def sample(self):
        return self.__distribution(1)
