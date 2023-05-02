import os
import pickle

from object_model_generation.object_instance import ObjectInstance


class ObjectModel:

    @classmethod
    def load(cls, session_path, use_original, object_model_name = ""):
        path = session_path
        if use_original:
            object_model_path = os.path.join(path, "objects_original.pkl")
        else:
            if len(object_model_name) > 0:
                path = os.path.join(path, "objects")
                path = os.path.join(path, object_model_name)
            object_model_path = os.path.join(path, "objects.pkl")
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

    def save(self, use_original=False, name=""):
        path = self.sessionPath
        if len(name) > 0:
            if use_original:
                raise AttributeError()
            path = os.path.join(path, "objects")
            path = os.path.join(path, name)
            os.mkdir(path)
        if use_original:
            object_model_path = os.path.join(path, "objects_original.pkl")
        else:
            object_model_path = os.path.join(path, "objects.pkl")
        with open(object_model_path, "wb") as write_file:
            pickle.dump(self, write_file)

    # TODO: tried with copy.deepcopy so to not change this object, but failed due to memory shortage
    def save_without_global_model(self, use_original):
        #copied = copy.deepcopy(self)
        obj: ObjectInstance
        for oid, obj in self.objectsById.items():
            obj.global_model = None
        self.save(use_original)


