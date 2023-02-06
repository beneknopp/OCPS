import logging

from utils.support_distribution import SupportDistribution


class ObjectInstance:
    otypes: list
    executionModelPaths: dict
    executionModelDepth: int
    executionModelEvaluationDepth: int
    supportDistributions: dict

    @classmethod
    def set_(cls, otypes, execution_model_paths, execution_model_depth, execution_model_evaluation_depth, schema_distributions):
        cls.otypes = otypes
        cls.executionModelPaths = execution_model_paths
        cls.executionModelDepth = execution_model_depth
        cls.executionModelEvaluationDepth = execution_model_evaluation_depth
        cls.supportDistributions = cls.__make_support_distributions(otypes, schema_distributions)

    @classmethod
    def __make_support_distributions(cls, otypes, schema_distributions):
        support_distributions = dict()
        for otype in otypes:
            otype_schemata = schema_distributions[otype]
            otype_support_distributions = dict()
            for paths_with_distributions in otype_schemata.values():
                for path, distribution in paths_with_distributions.items():
                    otype_support_distributions[path] = SupportDistribution(path, distribution)
            support_distributions[otype] = otype_support_distributions
        return support_distributions
    # support: sample() ->

    def __init__(self, otype, oid):
        self.oid = oid
        self.otype = otype
        self.attributes = dict()
        self.direct_object_model = {any_otype: set() for any_otype in self.otypes}
        self.reverse_object_model = {any_otype: set() for any_otype in self.otypes}
        self.total_local_model = {any_otype: set() for any_otype in self.otypes}
        self.locally_closed_types = {any_otype: False for any_otype in self.otypes}
        self.locally_closed = False
        self.globally_closed = False
        self.supportDistributions = self.supportDistributions[otype]
        self.__initialize_global_model()

    def __initialize_global_model(self):
        self.global_model = {
            depth: {
                path: []
                for path in paths
            }
            for depth, paths in self.executionModelPaths[self.otype].items()
        }
        self.global_model[0][tuple([self.otype])] = [self]

    def get_schema(self):
        local_schema = [0] * len(self.otypes)
        for i, otype in enumerate(self.otypes):
            local_schema[i] += len(self.global_model[otype])
        return tuple(local_schema)

    @classmethod
    def merge(cls, parent, child, merge_map):
        parent.direct_object_model[child.otype].add(child)
        parent.total_local_model[child.otype].add(child)
        child.reverse_object_model[parent.otype].add(parent)
        child.total_local_model[parent.otype].add(parent)
        cls.update_global_models(merge_map)

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
    def update_global_models2(cls, obj1, obj2):
        ot1 = obj1.otype
        ot2 = obj2.otype
        for d in range(cls.executionModelDepth):
            for path in cls.executionModelPaths[ot2][d]:
                path_from_ot1 = tuple([ot1] + list(path))
                objs = obj2.global_model[d][path]
                obj1.global_model[d+1][path_from_ot1] = list(set(objs + obj1.global_model[d+1][path_from_ot1]))
            for path in cls.executionModelPaths[ot1][d]:
                path_from_ot2 = tuple([ot2] + list(path))
                objs = obj1.global_model[d][path]
                obj2.global_model[d+1][path_from_ot2] = list(set(objs + obj2.global_model[d+1][path_from_ot2]))

    @classmethod
    def update_global_models(cls, merge_map: dict):
        obj: ObjectInstance
        for obj, paths in merge_map.items():
            for path, any_objs in paths.items():
                depth = len(path) - 1
                obj.global_model[depth][path] += any_objs

    def close_type(self, otype):
        self.locally_closed_types[otype] = True

    def close_local_model(self):
        self.locally_closed = True

    def set_timestamp(self, time):
        self.time = time
