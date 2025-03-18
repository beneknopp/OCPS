import os

import pm4py


def load_ocel(ocel_path):
    try:
        ocel = pm4py.read_ocel2(ocel_path)
    except:
        ocel = pm4py.read_ocel(ocel_path)
    return ocel

def load_postprocessed_input_ocel(session_path):
    postprocessed_ocel_path = os.path.join(session_path, "postprocessed_input.sqlite")
    return load_ocel(postprocessed_ocel_path)