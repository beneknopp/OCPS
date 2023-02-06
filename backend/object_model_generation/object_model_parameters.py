from utils.request_params_parser import RequestParamsParser


class ObjectModelParameters:
    otypes: []
    seedType: str
    nonEmittingTypes: []
    numberOfObjects: int
    activitySelectedTypes: {}
    activityLeadingTypes: {}
    executionModelsDepth: int
    executionModelEvaluationDepth: int

    def __init__(self, omp_dto):
        self.seedType = omp_dto['seedType']
        otypes_str = omp_dto['otypes']
        self.executionModelDepth = int(omp_dto["executionModelDepth"])
        self.executionModelEvaluationDepth = int(omp_dto["executionModelEvaluationDepth"])
        self.otypes = otypes_str.split(",") if len(otypes_str) > 0 else []
        non_emitting_types_str = omp_dto['nonEmittingTypes']
        self.nonEmittingTypes = non_emitting_types_str.split(",") if len(non_emitting_types_str) > 0 else []
        self.numberOfObjects = int(omp_dto['numberOfObjects'])
        activity_leading_types, activity_selected_types = RequestParamsParser \
            .parse_activity_leading_type_and_selected_types(omp_dto)
        self.activityLeadingTypes = activity_leading_types
        self.activitySelectedTypes = activity_selected_types
