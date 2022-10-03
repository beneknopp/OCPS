from object_model_generation.object_instance import ObjectInstance
from ocpn_discovery.net_utils import Transition


class ScheduledActivity:
    transition: Transition
    paths: dict  # object id -> firing sequence
    time: int  # time of execution as maximal time of involvedObjects
    def __init__(self, transition, paths, time):
        self.transition = transition
        self.paths = paths
        self.time = time
        pass


class SimulationObjectInstance:
    objectInstance: ObjectInstance
    oid: str
    otype: str
    time: int
    tokens: list  # of Token
    directObjectModel: dict # of SimulationObjectInstance
    active: bool # next activity predicted
    nextActivity: ScheduledActivity

    def __init__(self, obj_instance: ObjectInstance, tokens):
        self.objectInstance = obj_instance
        self.oid = obj_instance.oid
        self.otype = obj_instance.otype
        self.time = obj_instance.time
        self.tokens = tokens
        self.active = False
        self.directObjectModel = dict()
