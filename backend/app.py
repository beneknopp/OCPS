import json
import logging
import os
import pickle
import numpy as np
import pm4py
from flask import Flask, flash, request
from flask_cors import cross_origin

from dtos.response import Response
from input_ocel_processing.ocel_file_format import OcelFileFormat
from input_ocel_processing.postprocessor import InputOCELPostprocessor
from input_ocel_processing.preprocessor import InputOCELPreprocessor
from input_ocel_processing.process_config import ProcessConfig
from object_model_generation.object_model_generator import ObjectModelGenerator
from object_model_generation.object_model_parameters import ObjectModelParameters
from object_model_generation.training_model_preprocessor import TrainingModelPreprocessor
from ocpn_discovery.ocpn_discoverer import OCPN_Discoverer
from simulation.initializer import SimulationInitializer
from simulation.simulator import Simulator
from utils.request_params_parser import RequestParamsParser

RUNTIME_RESOURCE_FOLDER = os.path.abspath('runtime_resources')
ALLOWED_EXTENSIONS = {'jsonocel', 'xml'}

app = Flask(__name__)
app.config['RUNTIME_RESOURCE_FOLDER'] = RUNTIME_RESOURCE_FOLDER


@app.route('/')
@cross_origin()
def hello_world():  # put application's code here
    return {"ping": "Hello World!"}


@app.route('/upload-ocel', methods=['GET', 'POST'])
@cross_origin()
def upload_ocel():
    # try:
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return Response.get(True)
    if file and allowed_file(file.filename):
        clear_state()
        session_key, session_path = make_session()
        xml = OcelFileFormat.XML
        jsonocel = OcelFileFormat.JSONOCEL
        file_format = xml if file.filename.endswith('xml') else jsonocel
        file_name = "input.jsonocel" if file_format == jsonocel else "input.xml"
        ocel_preprocessor = InputOCELPreprocessor(session_path, file_name, file)
        ocel_preprocessor.preprocess()
        ocel_preprocessor.write_state()
        return {
            "sessionKey": session_key,
            "otypes": ocel_preprocessor.get_otypes(),
            "acts": ocel_preprocessor.get_acts(),
            "activity_allowed_types": ocel_preprocessor.get_activity_allowed_otypes(),
            "activity_leading_type_candidates": ocel_preprocessor.get_activity_leading_otype_candidates(),
            "activity_leading_type_groups": ocel_preprocessor.get_activity_leading_type_groups()
        }
    return Response.get(False)


@app.route('/ocel-config', methods=['GET', 'POST'])
@cross_origin()
def ocel_config():
    if not request.method == 'POST':
        return Response.get(True)
    session_path = get_session_path(request)
    config_bytes = request.files["ocelInfo"].read()
    config_dto = json.loads(config_bytes)
    process_config = ProcessConfig(config_dto, session_path)
    process_config.save()
    postprocessor = InputOCELPostprocessor(session_path, process_config)
    postprocessor.postprocess()
    return Response.get(True)


@app.route('/generate-object-model', methods=['GET', 'POST'])
@cross_origin()
def generate_object_model():
    if not request.method == 'POST':
        return True
    session_path = get_session_path(request)
    file_path = os.path.join(session_path, "postprocessed_input.jsonocel")
    ocel = pm4py.read_ocel(file_path)
    ProcessConfig.update_non_emitting_types(session_path, request.form['nonEmittingTypes'])
    object_model_parameters = ObjectModelParameters(request.form)
    logging.info("Preprocessing Training Data...")
    training_model_preprocessor = TrainingModelPreprocessor(session_path, ocel, object_model_parameters)
    training_model_preprocessor.build()
    training_model_preprocessor.save()
    object_model_generator = ObjectModelGenerator(session_path, ocel, object_model_parameters,
                                                  training_model_preprocessor)
    object_model_generator.generate()
    object_model_generator.save(session_path)
    return object_model_generator.get_response()


@app.route('/discover-ocpn', methods=['GET', 'POST'])
@cross_origin()
def discover_ocpn():
    if not request.method == 'POST':
        return True
    session_path = get_session_path(request)
    form = request.form
    ocpn_discoverer = OCPN_Discoverer(session_path)
    activity_selected_types = RequestParamsParser.parse_activity_selected_types(form)
    ocpn_discoverer.discover(activity_selected_types)
    ocpn_discoverer.save()
    ocpn_dto = ocpn_discoverer.export()
    return ocpn_dto


@app.route('/simulation-state', methods=['GET'])
@cross_origin()
def simulation_state():
    return Response.get(True)


@app.route('/object-model-stats', methods=['GET'])
@cross_origin()
def object_model_stats():
    session_path = get_session_path(request)
    args = request.args
    otype = args["otype"]
    process_config = ProcessConfig.load(session_path)
    resp = dict()
    for any_otype in process_config.otypes:
        log_based = pickle.load(open(os.path.join(
            session_path, otype + "_to_" + any_otype + "_schema_dist.pkl"), "rb"))
        simulated = pickle.load(open(os.path.join(
            session_path, otype + "_to_" + any_otype + "_schema_dist_simulated.pkl"), "rb"))
        total_min = min(min(log_based["x_axis"]), min(simulated["x_axis"]))
        total_max = max(max(log_based["x_axis"]), max(simulated["x_axis"]))
        x_axis = [str(i) for i in range(total_min, total_max + 1)]
        log_based = [0] * (min(log_based["x_axis"]) - total_min) \
                    + log_based["log_based"] \
                    + [0] * (total_max - max(log_based["x_axis"]))
        simulated = [0] * (min(simulated["x_axis"]) - total_min) \
                    + simulated["simulated"] \
                    + [0] * (total_max - max(simulated["x_axis"]))
        resp[any_otype] = {
            "x_axis": x_axis,
            "log_based": log_based,
            "simulated": simulated
        }
    return Response.get(resp)


@app.route('/arrival-times', methods=['GET'])
@cross_origin()
def arrival_stats():
    session_path = get_session_path(request)
    args = request.args
    otype = args["otype"]
    process_config = ProcessConfig.load(session_path)
    resp = dict()
    log_based = pickle.load(open(os.path.join(
        session_path, "arrival_times_" + otype + "_log_based.pkl"), "rb"))
    simulated = pickle.load(open(os.path.join(
        session_path, "arrival_times_" + otype + "_simulated.pkl"), "rb"))
    for any_otype in process_config.otypes:
        if not log_based[any_otype] or not simulated[any_otype]:
            resp[any_otype] = {
                "x_axis": [0],
                "log_based": [0],
                "simulated": [0]
            }
            continue
        total_min = min(min(log_based[any_otype]), min(simulated[any_otype]))
        total_max = max(max(log_based[any_otype]), max(simulated[any_otype]))
        bins = np.linspace(total_min, total_max, 10)
        bins = [round(i) for i in bins]
        abs_min = min([abs(x) for x in bins])
        if abs_min != 0:
            abs_min_arg = [x for x in bins if abs(x) == abs_min][0]
            bins = [x - abs_min_arg for x in bins]
            if abs_min_arg > 0:
                bins.append(bins[-1] + abs_min_arg)
            else:
                bins = [bins[0] + abs_min_arg] + bins
        log_based_digitized = np.digitize(log_based[any_otype], bins)
        simulated_digitized = np.digitize(simulated[any_otype], bins)
        log_based_freqs = [len(log_based_digitized[log_based_digitized == i]) for i in range(1, len(bins) + 1)]
        simulated_freqs = [len(simulated_digitized[simulated_digitized == i]) for i in range(1, len(bins) + 1)]
        log_based_rel = [freq / sum(log_based_freqs) for freq in log_based_freqs]
        simulated_rel = [freq / sum(simulated_freqs) for freq in simulated_freqs]
        x_axis = [str(i) for i in bins]
        resp[any_otype] = {
            "x_axis": x_axis,
            "log_based": log_based_rel,
            "simulated": simulated_rel
        }
    return Response.get(resp)


@app.route('/initialize-simulation', methods=['GET'])
@cross_origin()
def initialize_simulation():
    args = request.args
    session_key = args["sessionKey"]
    session_path = os.path.join(app.config['RUNTIME_RESOURCE_FOLDER'], session_key)
    simulation_initializer = SimulationInitializer(session_path)
    simulation_initializer.load()
    simulation_initializer.initialize()
    simulation_initializer.save()
    del simulation_initializer
    simulator = Simulator(session_path)
    simulator.initialize()
    simulator.schedule_next_activity()
    state = simulator.export_current_state()
    simulator.save()
    return Response.get(state)


@app.route('/simulate', methods=['GET'])
@cross_origin()
def simulate():
    args = request.args
    steps = int(args['steps'])
    session_path = get_session_path(request)
    simulator = Simulator.load(session_path)
    simulator.run_steps(steps)
    state = simulator.export_current_state()
    simulator.save()
    return Response.get(state)


def make_session():
    with open("running_session_key") as rf:
        session_key = str(int(rf.read()) + 1)
    with open("running_session_key", "w") as wf:
        wf.write(session_key)
    session_path = os.path.join(
        app.config['RUNTIME_RESOURCE_FOLDER'], session_key)
    try:
        os.mkdir(session_path)
    except FileNotFoundError:
        os.mkdir(app.config['RUNTIME_RESOURCE_FOLDER'])
        os.mkdir(session_path)
    logging.basicConfig(filename=os.path.join(session_path, "ocps_session.log"),
                        encoding='utf-8', level=logging.DEBUG)
    return session_key, session_path


def get_session_path(request):
    args = request.args
    session_key = args['sessionKey']
    session_path = os.path.join(app.config['RUNTIME_RESOURCE_FOLDER'], session_key)
    return session_path


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# TODO: delete existing resources for new OCEL to work on
def clear_state():
    pass


if __name__ == '__main__':
    app.run()
