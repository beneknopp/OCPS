import json
import logging
import os
import pickle

import numpy as np
import pm4py
from flask import Flask, flash, request, send_from_directory
from flask_cors import cross_origin

from dtos.response import Response
from eval.simulation_evaluator import SimulationEvaluator
from ocel_processing.load_ocel import load_ocel, load_postprocessed_input_ocel
from ocel_processing.ocel_file_format import OcelFileFormat
from ocel_processing.postprocessor import InputOCELPostprocessor
from ocel_processing.preprocessor import InputOCELPreprocessor
from ocel_processing.process_config import ProcessConfig
from object_model_generation.generator_parametrization import GeneratorParametrization
from object_model_generation.object_model_generator import ObjectModelGenerator
from object_model_generation.object_model_parameters import ObjectModelParameters
from object_model_generation.training_model_preprocessor import TrainingModelPreprocessor
from ocpn_discovery.ocpn_discoverer import OCPN_Discoverer
from simulation.initializer import SimulationInitializer
from simulation.simulator import Simulator
from utils.request_params_parser import RequestParamsParser

RUNTIME_RESOURCE_FOLDER = os.path.abspath('runtime_resources')
ALLOWED_EXTENSIONS = {'jsonocel', 'xml', 'sqlite'}
DEFAULT_OCEL_FILE_NAME = "order-management-small"
DEFAULT_FILE_FORMAT = "sqlite"
DEFAULT_OCEL_PATH = os.path.join(os.path.abspath("logs"), DEFAULT_OCEL_FILE_NAME + "." + DEFAULT_FILE_FORMAT)

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
        sqlite = OcelFileFormat.SQLITE
        file_format = xml if file.filename.endswith('xml') else jsonocel if file.filename.endswith(".jsonocel") else sqlite
        file_name = "input.jsonocel" if file_format == jsonocel else "input.xml" if file_format == xml else "input.sqlite"
        ocel_preprocessor = InputOCELPreprocessor(session_path)
        file_path = ocel_preprocessor.store_file(file_format, file_name, file)
        ocel_preprocessor.load_ocel(file_path)
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

@app.route('/load-default-ocel', methods=['GET', 'POST'])
@cross_origin()
def load_default_ocel():
    clear_state()
    session_key, session_path = make_session()
    ocel_preprocessor = InputOCELPreprocessor(session_path)
    ocel_preprocessor.store_file_format(DEFAULT_FILE_FORMAT)
    ocel_preprocessor.load_ocel(DEFAULT_OCEL_PATH)
    ocel_preprocessor.preprocess()
    ocel_preprocessor.write_state()
    ProcessConfig.update_raw_ocel_path(
        session_path,
        DEFAULT_OCEL_PATH
    )
    return {
        "sessionKey": session_key,
        "otypes": ocel_preprocessor.get_otypes(),
        "acts": ocel_preprocessor.get_acts(),
        "activity_allowed_types": ocel_preprocessor.get_activity_allowed_otypes(),
        "activity_leading_type_candidates": ocel_preprocessor.get_activity_leading_otype_candidates(),
        "activity_leading_type_groups": ocel_preprocessor.get_activity_leading_type_groups()
    }

@app.route('/ocel-config', methods=['GET', 'POST'])
@cross_origin()
def ocel_config():
    if not request.method == 'POST':
        return Response.get(True)
    session_path = get_session_path(request)
    config_bytes = request.files["ocelInfo"].read()
    config_dto = json.loads(config_bytes)
    process_config = ProcessConfig.load(session_path)
    process_config.init_config(config_dto)
    postprocessor = InputOCELPostprocessor(session_path, process_config)
    postprocessor.postprocess()
    clock_offset = postprocessor.get_clock_offset()
    process_config.clockOffset = clock_offset
    process_config.save()
    return Response.get(True)

@app.route('/object-model-names', methods=['GET'])
@cross_origin()
def objectModelNames():
    session_path = get_session_path(request)
    oms_path = os.path.join(session_path, "objects")
    names = list(os.walk(oms_path))[0][1]
    return Response.get(names)

@app.route('/initialize-object-generator', methods=['POST'])
@cross_origin()
def initialize_object_generator():
    session_path = get_session_path(request)
    ocel = load_postprocessed_input_ocel(session_path)
    ProcessConfig.update_non_emitting_types(session_path, request.form['nonEmittingTypes'])
    object_model_parameters = ObjectModelParameters(request.form)
    logging.info("Preprocessing Training Data...")
    training_model_preprocessor = TrainingModelPreprocessor(session_path, ocel, object_model_parameters)
    training_model_preprocessor.build()
    training_model_preprocessor.save()
    return Response.get(True)

@app.route('/generator-parameters', methods=['GET'])
@cross_origin()
def get_parameters():
    session_path = get_session_path(request)
    start_logging(session_path)
    args = request  .args
    otype = args["otype"]
    parameter_type = args["parameterType"]
    training_model_preprocessor: TrainingModelPreprocessor = TrainingModelPreprocessor.load(session_path)
    generator_parametrization: GeneratorParametrization = training_model_preprocessor.generatorParametrization
    parameter_export = generator_parametrization.export_parameters(otype, parameter_type)
    return Response.get(parameter_export)

@app.route('/select-for-training', methods=['GET'])
@cross_origin()
def select_for_training():
    session_path = get_session_path(request)
    start_logging(session_path)
    args = request.args
    otype = args["otype"]
    parameter_type = args["parameterType"]
    attribute = args["attribute"]
    selected = True if args["selected"] == "True" else False
    training_model_preprocessor: TrainingModelPreprocessor = TrainingModelPreprocessor.load(session_path)
    generator_parametrization: GeneratorParametrization = training_model_preprocessor.generatorParametrization
    generator_parametrization.select_for_training(otype, parameter_type, attribute, selected)
    training_model_preprocessor.save()
    parameter_export = generator_parametrization.export_parameters(otype, parameter_type, attribute)
    return Response.get(parameter_export)


@app.route('/mark-as-batch-arrival', methods=['GET'])
@cross_origin()
def mark_as_batch_arrival():
    session_path = get_session_path(request)
    start_logging(session_path)
    args = request.args
    otype = args["otype"]
    attribute = args["attribute"]
    selected = True if args["selected"] == "True" else False
    training_model_preprocessor: TrainingModelPreprocessor = TrainingModelPreprocessor.load(session_path)
    generator_parametrization: GeneratorParametrization = training_model_preprocessor.generatorParametrization
    generator_parametrization.mark_as_batch_arrival(otype, attribute, selected)
    training_model_preprocessor.save()
    return Response.get(True)

@app.route('/switch-model', methods=['GET'])
@cross_origin()
def switch_model():
    session_path = get_session_path(request)
    start_logging(session_path)
    args = request.args
    otype = args["otype"]
    parameter_type = args["parameterType"]
    attribute = args["attribute"]
    fitting_model = args["fittingModel"]
    training_model_preprocessor: TrainingModelPreprocessor = TrainingModelPreprocessor.load(session_path)
    generator_parametrization: GeneratorParametrization = training_model_preprocessor.generatorParametrization
    generator_parametrization.switch_fitting_model(otype, parameter_type, attribute, fitting_model)
    training_model_preprocessor.save()
    parameter_export = generator_parametrization.export_parameters(otype, parameter_type, attribute)
    return Response.get(parameter_export)

@app.route('/change-parameters', methods=['POST'])
@cross_origin()
def change_parameters():
    session_path = get_session_path(request)
    start_logging(session_path)
    args = request.form
    otype = args["otype"]
    parameter_type = args["parameterType"]
    attribute = args["attribute"]
    parameters = args["parameters"]
    training_model_preprocessor: TrainingModelPreprocessor = TrainingModelPreprocessor.load(session_path)
    generator_parametrization: GeneratorParametrization = training_model_preprocessor.generatorParametrization
    generator_parametrization.change_parameters(otype, parameter_type, attribute, parameters)
    training_model_preprocessor.save()
    parameter_export = generator_parametrization.export_parameters(otype, parameter_type, attribute)
    return Response.get(parameter_export)

@app.route('/generate-objects', methods=['GET', 'POST'])
@cross_origin()
def generate_object_model():
    if not request.method == 'POST':
        return True
    session_path = get_session_path(request)
    start_logging(session_path)
    ocel = load_postprocessed_input_ocel(session_path)
    ProcessConfig.update_non_emitting_types(session_path, request.form['nonEmittingTypes'])
    object_model_parameters = ObjectModelParameters(request.form)
    logging.info("Preprocessing Training Data...")
    training_model_preprocessor = TrainingModelPreprocessor.load(session_path)
    object_model_generator = ObjectModelGenerator(session_path, ocel, object_model_parameters, training_model_preprocessor)
    object_model_generator.generate()
    object_model_generator.make_model_and_stats()
    object_model_generator.save()
    response = object_model_generator.get_response()
    return response

@app.route('/name-objects', methods=['GET'])
@cross_origin()
def nameObjects():
    args = request.args
    session_path = get_session_path(request)
    name = args["name"]
    ObjectModelGenerator.name(session_path, name)
    return Response.get({})

@app.route('/discover-ocpn', methods=['GET', 'POST'])
@cross_origin()
def discover_ocpn():
    if not request.method == 'POST':
        return True
    session_path = get_session_path(request)
    start_logging(session_path)
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

@app.route('/arrival-times', methods=['GET'])
@cross_origin()
def arrival_stats():
    session_path = get_session_path(request)
    start_logging(session_path)
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
    use_original_marking = args["useOriginalMarking"] == "true"
    object_model_name = args["objectModelName"] if not use_original_marking else ""
    session_path = os.path.join(app.config['RUNTIME_RESOURCE_FOLDER'], session_key)
    ProcessConfig.update_use_original_marking(session_path, use_original_marking)
    start_logging(session_path)
    simulation_initializer = SimulationInitializer(session_path, use_original_marking, object_model_name)
    simulation_initializer.load_net_and_objects()
    simulation_initializer.initialize()
    simulation_initializer.save()
    del simulation_initializer
    simulator = Simulator(session_path, use_original_marking, object_model_name)
    simulator.initialize()
    simulator.schedule_next_activity()
    state = simulator.export_current_state()
    simulator.save()
    return Response.get(state)

@app.route('/simulate', methods=['GET'])
@cross_origin()
def simulate():
    args = request.args
    use_original_marking = args["useOriginalMarking"] == "true"
    object_model_name = args["objectModelName"] if not use_original_marking else ""
    steps = int(args['steps'])
    session_path = get_session_path(request)
    start_logging(session_path)
    simulator: Simulator = Simulator.load(session_path, object_model_name)
    simulator.run_steps(steps)
    state = simulator.export_current_state()
    simulator.save()
    return Response.get(state)

@app.route('/available-simulated-models', methods=['GET'])
@cross_origin()
def available_simulated_models():
    session_path = get_session_path(request)
    oms_path = os.path.join(session_path, "simulated_logs")
    names = list(os.walk(oms_path))[0][1]
    return Response.get(names)

@app.route('/update-evaluation-selected-object-models', methods=['POST'])
@cross_origin()
def update_evaluation_selected_object_models():
    session_path = get_session_path(request)
    form = request.form
    selected_object_models_str = form["selectedObjectModels"]
    selected_object_models = json.loads(selected_object_models_str)
    path = os.path.join(session_path, "evaluation")
    path = os.path.join(path, "selected_object_models.pkl")
    with open(path, "wb") as write_file:
        pickle.dump(selected_object_models, write_file)
    simulation_evaluator = SimulationEvaluator(session_path, selected_object_models)
    simulation_evaluator.evaluate()
    simulation_evaluator.save()
    return Response.get(selected_object_models)

@app.route('/evaluate', methods=['GET'])
@cross_origin()
def evaluate():
    args = request.args
    stats_type = args["statsType"]
    otype = args["otype"]
    session_path = get_session_path(request)
    simulation_evaluator: SimulationEvaluator = SimulationEvaluator.load(session_path)
    response = simulation_evaluator.get(otype, stats_type)
    return Response.get(response)

@app.route('/ocel-export', methods=['GET'])
@cross_origin()
def exportOCEL():
    args = request.args
    session_key = args["sessionKey"]
    download_path = os.path.join(app.config['RUNTIME_RESOURCE_FOLDER'], session_key)
    return send_from_directory(
        directory=download_path,
        path = "/",
        filename="simulated_ocel.jsonocel"
    )

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
    start_logging(session_path)
    os.mkdir(os.path.join(session_path, "objects"))
    os.mkdir(os.path.join(session_path, "simulated_logs"))
    os.mkdir(os.path.join(session_path, "evaluation"))
    process_config = ProcessConfig(session_path)
    process_config.save()
    return session_key, session_path

def start_logging(session_path):
    logging.basicConfig(filename=os.path.join(session_path, "ocps_session.log"),
                        encoding='utf-8', level=logging.DEBUG)

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
    app.debug = True
    app.run(use_reloader=False)
