from random import random


class CumulativeDistribution:

    def __init__(self, weighted_items: dict):
        self.__make_probabilities(weighted_items)
        self.__make_cumulative_distribution()

    def __make_probabilities(self, weighted_items):
        items_weights_list = list(map(lambda k: (k, weighted_items[k]), weighted_items))
        total = 0
        for key, count in items_weights_list:
            total += count
        probabilities = list(map(lambda key_cum_count: (key_cum_count[0], float(key_cum_count[1]) / float(total)),
                                 items_weights_list
                                 ))
        probabilities.sort(key=lambda it: -it[1])
        self.probabilities = probabilities

    def __make_cumulative_distribution(self):
        probabilities = self.probabilities
        cum_p = 0.0
        cum_prob_dist = []
        for key, p in probabilities:
            cum_prob_dist.append((key, cum_p + p))
            cum_p += p
        cum_prob_dist[-1] = (cum_prob_dist[-1][0], 1.0)
        self.cumulative_distribution = cum_prob_dist

    def sample(self):
        rnd = random()
        s = next(el for el in self.cumulative_distribution if el[1] >= rnd)[0]
        return s

    def sample_n(self, n):
        return [self.sample() for i in range(n)]
