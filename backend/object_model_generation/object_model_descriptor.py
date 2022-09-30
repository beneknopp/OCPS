import os

from object_model_generation.object_instance import ObjectInstance
from object_model_generation.object_model import ObjectModel


class ObjectModelDescriptor:

    def __init__(self, session_path):
        self.sessionPath = session_path
        otypes_path = os.path.join(self.sessionPath, 'otypes')
        with open(otypes_path) as read_file:
            self.otypes = read_file.read().split(",")

    def describe(self, otype):
        object_model: ObjectModel = ObjectModel.load(self.sessionPath)
        objects = object_model.objectsByType[otype]
        ot_card_counts = {otype: dict() for otype in self.otypes}
        ot_cards = {otype: set() for otype in self.otypes}
        ret = {otype: dict() for otype in self.otypes}
        obj: ObjectInstance
        for ot in self.otypes:
            for obj in objects.keys():
                ot_card = len(obj.global_model[ot])
                if ot_card not in ot_card_counts[ot]:
                    ot_card_counts[ot][ot_card] = 1
                ot_card_counts[ot][ot_card] = ot_card_counts[ot][ot_card] + 1
                ot_cards[ot].add(ot_card)
            min_card = min(ot_cards[ot])
            max_card = max(ot_cards[ot])
            x_axis = range(min_card, max_card + 1)
            log_based_card_freqs = list(map(lambda i: ot_card_counts[ot][i] if i in ot_card_counts[ot] else 0, x_axis))
            log_based = [float(x) / float(sum(log_based_card_freqs)) for x in log_based_card_freqs]
            simulated = log_based[:]
            x_axis = [str(i) for i in x_axis]
            ret[ot] = {"x_axis": x_axis, "log_based": log_based, 'simulated': simulated}
        return ret
