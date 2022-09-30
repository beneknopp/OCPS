import logging

from utils.support_distribution import SupportDistribution


class ObjectInstance:
    otypes: list
    execution_model_paths: dict
    support_distributions: dict

    @classmethod
    def set_(cls, otypes, execution_model_paths, global_schemata):
        cls.otypes = otypes
        cls.execution_model_paths = execution_model_paths
        cls.support_distributions = cls.__make_support_distributions(otypes, global_schemata)

    @classmethod
    def __make_support_distributions(cls, otypes, global_schemata):
        support_distributions = dict()
        for i, otype in enumerate(otypes):
            otype_schemata = global_schemata[otype]
            otype_support_distributions = dict()
            for j, any_otype in enumerate(otypes):
                if i == j:
                    continue
                otype_support_distributions[any_otype] = SupportDistribution(otype, any_otype, otypes, otype_schemata)
            support_distributions[otype] = otype_support_distributions
        return support_distributions

    def __init__(self, otype, oid):
        self.oid = oid
        self.otype = otype
        self.attributes = dict()
        self.direct_object_model = {any_otype: set() for any_otype in self.otypes}
        self.reverse_object_model = {any_otype: set() for any_otype in self.otypes}
        self.total_local_model = {any_otype: set() for any_otype in self.otypes}
        self.global_model = {any_otype: set() for any_otype in self.otypes}
        self.locally_closed_types = {any_otype: False for any_otype in self.otypes}
        self.locally_closed = False
        self.globally_closed = False
        self.support_distributions = self.support_distributions[otype]

    def get_schema(self):
        local_schema = [0] * len(self.otypes)
        for i, otype in enumerate(self.otypes):
            local_schema[i] += len(self.global_model[otype])
        return tuple(local_schema)

    @classmethod
    def merge(cls, parent, child):
        parent.direct_object_model[child.otype].add(child)
        parent.total_local_model[child.otype].add(child)
        child.reverse_object_model[parent.otype].add(parent)
        child.total_local_model[parent.otype].add(parent)
        cls.update_global_models(parent, child)

    @classmethod
    def update_support(cls, member):
        old_local_support = member.local_support
        old_global_support = member.global_support
        new_local_support = {
            schema: old_local_support[schema]
            for schema in member.local_support
            if all(
                list(schema)[i] >= len(member.total_local_model[otype])
                for (i, otype) in enumerate(cls.otypes)
            )
        }
        new_global_support = {
            schema: old_global_support[schema]
            for schema in member.global_support
            if all(
                list(schema)[i] >= len(member.global_model[otype])
                for (i, otype) in enumerate(cls.otypes)
            )
        }
        member.local_support = new_local_support
        member.global_support = new_global_support

    @classmethod
    def emit(cls, source, otype, oid, reverse=False):
        emitted_obj = cls(otype, oid)
        child = source if reverse else emitted_obj
        parent = source if not reverse else emitted_obj
        cls.merge(parent, child)
        return emitted_obj

    @classmethod
    def update_global_models(cls, obj1, obj2):
        ot1 = obj1.otype
        ot2 = obj2.otype
        ot1_side = []
        ot2_side = []
        for ot in [ot for ot in cls.otypes if ot != ot1 and ot != ot2]:
            if ot not in cls.execution_model_paths[ot1]:
                continue
            sp1 = cls.execution_model_paths[ot1][ot]
            sp2 = cls.execution_model_paths[ot2][ot]
            if len(sp1) < len(sp2):
                ot1_side.append(ot)
            elif len(sp1) > len(sp2):
                ot2_side.append(ot)
            else:
                raise ValueError(
                    "The paths from " + ot1 + " and from " + ot2 + " to " + ot + " in the object type graph" + \
                    " have the same length.")
        obj1_global_model = {
            ot: obj1.global_model[ot]
            for ot in ot1_side
        }
        obj1_global_model[ot1] = {obj1}
        obj2_global_model = {
            ot: obj2.global_model[ot]
            for ot in ot2_side
        }
        obj2_global_model[ot2] = {obj2}
        for ex_ot1, ex_objs1 in obj1_global_model.items():
            for ex_ot2, ex_objs2 in obj2_global_model.items():
                for ex_obj1 in ex_objs1:
                    for ex_obj2 in ex_objs2:
                        ex_obj1.global_model[ex_ot2].add(ex_obj2)
                        ex_obj2.global_model[ex_ot1].add(ex_obj1)

    def close_type(self, otype):
        self.locally_closed_types[otype] = True

    def close_local_model(self):
        self.locally_closed = True

    def set_timestamp(self, time):
        self.time = time
