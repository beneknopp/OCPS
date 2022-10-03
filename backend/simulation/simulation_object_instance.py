from object_model_generation.object_instance import ObjectInstance


class ScheduledActivity:
    activity: str
    involvedObjects: list  # of SimulationObjectInstance
    time: int  # time of execution as maximal time of involvedObjects

    def __init__(self):
        pass


class SimulationObjectInstance:
    oid: str
    otype: str
    time: int
    tokens: list  # of Token
    directObjectModel: list # of SimulationObjectInstance
    active: bool # next activity predicted
    nextActivity: ScheduledActivity

    def __init__(self, obj_instance: ObjectInstance, tokens):
        self.oid = obj_instance.oid
        self.otype = obj_instance.otype
        self.time = obj_instance.time
        self.tokens = tokens
        self.active = False
        self.directObjectModel = []
