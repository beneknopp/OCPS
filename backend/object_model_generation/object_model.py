import os
import pickle

from object_model_generation.object_instance import ObjectInstance


class ObjectModel:

    @classmethod
    def load(cls, session_path, use_original):
        use_original_appendix = "_original" if use_original else ""
        object_model_path = os.path.join(session_path, "objects" + use_original_appendix + ".pkl")
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

    def save(self, use_original):
        use_original_appendix = "_original" if use_original else ""
        object_model_path = os.path.join(self.sessionPath, "objects" + use_original_appendix + ".pkl")
        with open(object_model_path, "wb") as write_file:
            pickle.dump(self, write_file)

    # TODO: tried with copy.deepcopy so to not change this object, but failed due to memory shortage
    def save_without_global_model(self, use_original):
        #copied = copy.deepcopy(self)
        obj: ObjectInstance
        for oid, obj in self.objectsById.items():
            obj.global_model = None
        self.save(use_original)
