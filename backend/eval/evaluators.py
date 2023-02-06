from ocpa.algo.conformance.precision_and_fitness.variants import replay_context
import ocpa.algo.conformance.precision_and_fitness.utils as utils

def ocpn_to_ocel(ocel, ocpn, contexts=None, bindings=None):
    object_types = ocel.object_types
    if contexts == None or bindings == None:
        contexts, bindings = utils.calculate_contexts_and_bindings(ocel)
    en_l = replay_context.enabled_log_activities(ocel.log, contexts)
    en_m = replay_context.enabled_model_activities_multiprocessing(contexts, bindings, ocpn, object_types)
    precision, skipped_events, fitness = replay_context.calculate_precision_and_fitness(ocel.log, contexts, en_l,
                                                                                        en_m)
    return precision, fitness

def ocel_to_ocel(original_ocel, simulated_ocel, contexts=None, bindings=None):
    en_A = replay_context.enabled_log_activities(original_ocel.log, contexts)
    en_B = replay_context.enabled_log_activities(simulated_ocel.log, contexts)
    precision, skipped_events, fitness = replay_context.calculate_precision_and_fitness(original_ocel.log, contexts, en_A,
                                                                                        en_B)
    return precision, fitness