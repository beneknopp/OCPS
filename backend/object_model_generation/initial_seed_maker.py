import random

from object_model_generation.object_instance import ObjectInstance


class InitialSeedMaker:

    @classmethod
    def initialize_unconnected_objs(cls, leading_type_process_executions, oid,
        buffer, base_otype, nof_base_objects, open_objects, total_objects):

        base_objs_from_training_set = leading_type_process_executions[base_otype]
        if len(base_objs_from_training_set) < nof_base_objects:
            seed_process_execs = base_objs_from_training_set
        else:
            seed_keys = random.sample(list(base_objs_from_training_set.keys()), nof_base_objects)
            seed_process_execs = {seed_key : base_objs_from_training_set[seed_key]
                                  for seed_key in seed_keys}
        total_per_type = {
            base_otype: seed_process_execs.keys()
        }
        for otype in leading_type_process_executions.keys():
            if otype == base_otype:
                continue
            total_per_type[otype] = set()
            for seed_obj in seed_process_execs:
                total_per_type[otype].update(seed_process_execs[seed_obj][otype])

        for otype, objs in total_per_type.items():
            nof_objects = len(objs)
            for i in range(nof_objects):
                cls.create_obj(buffer, otype, oid, open_objects, total_objects)
        #random.shuffle(buffer)

    @classmethod
    def create_obj(cls, buffer, otype, oid, open_objects, total_objects):
        obj = ObjectInstance(otype, oid.get_and_inc())
        open_objects[otype].append(obj)
        total_objects[otype].append(obj)
        buffer.append(obj)
        oid.inc()
