from enum import Enum

from object_model_generation.object_instance import ObjectInstance


class PredictionMode(Enum):
    NEW = "NEW"
    APPEND = "APPEND"


class ObjectLinkPrediction:
    predict: bool
    predicted_type: str
    mode: PredictionMode
    reverse: bool
    selected_neighbor: ObjectInstance

    def __init__(self, predict=False, predicted_type=None, mode=None, reverse=None, selected_neighbor=None):
        self.predict = predict
        self.predicted_type = predicted_type
        self.mode = mode
        self.reverse = reverse
        self.selected_neighbor = selected_neighbor

    def pretty_print(self):
        return str({'predict': self.predict, 'predicted_type': self.predicted_type, 'mode': self.mode.value,
                    'reverse': self.reverse, 'selected_neighbor': self.selected_neighbor.oid})
