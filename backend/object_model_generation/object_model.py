import os
import pickle


class ObjectModel:

    @classmethod
    def load(cls, session_path):
        object_model_path = os.path.join(session_path, "objects.pkl")
        return pickle.load(open(object_model_path, "rb"))

    objectsByType: dict
    objectsById: dict

    def __init__(self, session_path):
        self.sessionPath = session_path
        self.objectsByType = dict()
        self.objectsById = dict()

    def addModel(self, otype, model):
        self.objectsByType[otype] = model
        self.objectsById.update({
            obj.oid: obj
            for obj in model
        })

    def save(self):
        object_model_path = os.path.join(self.sessionPath, "objects.pkl")
        with open(object_model_path, "wb") as write_file:
            pickle.dump(self, write_file)