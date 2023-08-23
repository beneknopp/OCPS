from enum import Enum

from object_model_generation.object_instance import ObjectInstance


class PredictionMode(Enum):
    NEW = "NEW"
    APPEND = "APPEND"


class ObjectLinkPrediction:
    predict: bool
    predicted_type: str
    mode: PredictionMode
    selected_neighbor: ObjectInstance

    def __init__(self, predict=False, predicted_type=None, mode=None, selected_neighbor=None,
                 merge_map=None):
        self.predict = predict
        self.predicted_type = predicted_type
        self.mode = mode
        self.selected_neighbor = selected_neighbor
        self.mergeMap = merge_map

    def pretty_print(self):
        return str({'predict': self.predict, 'predicted_type': self.predicted_type,
                    'mode': self.mode.value, 'selected_neighbor': self.selected_neighbor.oid})
