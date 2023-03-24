import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { AppService } from '../app.service';
import { DOMService } from '../dom.service';
import { AttributeParametrizationResponse } from './generator-info';
import { ObjectModelGenerationResponse, ObjectModelInfo, ObjectModelStats } from './object-model-info';

@Component({
  selector: 'app-object-model-generator',
  templateUrl: './object-model-generator.component.html',
  styleUrls: ['./object-model-generator.component.css', '../app.component.css']
})
export class ObjectModelGeneratorComponent implements OnInit {

  objectModelInfo: ObjectModelInfo = new ObjectModelInfo()
  configValid = true
  initialized = false
  reloadStats = false
  responseValid = false;
  selectedSeedType = undefined
  nonEmittingTypes: string[] = []
  sessionKey: string | undefined;
  numberOfObjects: number = 0;
  selectedObjectType: string | undefined
  selectedParameterType: string | undefined
  cachedSelectedObjectType: string | undefined
  cachedselectedParameterType: string | undefined
  statsTypes = ["Cardinalities", "Object Attributes", "Timing Information"]
  trainingSelectionMap: { [attribute: string]: boolean } = {}
  trainingModelMap: { [attribute: string]: string } = {}
  touchedParameters: { [attribute: string]: boolean } = {}
  attributeModelCandidates: { [attribute: string]: string[] } = {}
  parameterMap: { [attribute: string]: string } = {}
  load: boolean = true

  public barChartData: { [otype: string]: { data: number[], label: 'Log-Based' | 'Modeled' | 'Simulated' }[] } = {
    'orders': [
      { data: [0.0, 0.2, 0.4, 0.3, 0.1, 0.0, 0.0], label: 'Log-Based' },
      { data: [0.0, 0.15, 0.25, 0.25, 0.2, 0.15, 0], label: 'Simulated' }
    ]
  };
  public mbarChartLabels: { [otype: string]: string[] } = { 'orders': ['0', '1', '2', '3', '4', '5', '6'] };
  public barChartType: string = 'bar';
  public barChartOptions: any = {
    scaleShowVerticalLines: false,
    responsive: true
  };
  public barChartColors: Array<any> = [
    {
      backgroundColor: 'rgba(105,159,177,0.2)',
      borderColor: 'rgba(105,159,177,1)',
      pointBackgroundColor: 'rgba(105,159,177,1)',
      pointBorderColor: '#fafafa',
      pointHoverBackgroundColor: '#fafafa',
      pointHoverBorderColor: 'rgba(105,159,177)'
    },
    {
      backgroundColor: 'rgba(77,20,96,0.3)',
      borderColor: 'rgba(77,20,96,1)',
      pointBackgroundColor: 'rgba(77,20,96,1)',
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: 'rgba(77,20,96,1)'
    }
  ];
  omgResponse: ObjectModelGenerationResponse | undefined;
  attributes: string[] = [];


  constructor(
    public domService: DOMService,
    private router: Router,
    private appService: AppService
  ) { }

  ngOnInit(): void {
    this.sessionKey = this.domService.getSessionKey()
    // TODO
    this.domService.objectModelInfo$.subscribe(object_model_info => {
      this.domService.otypeLocalModels$.subscribe(otype_local_models => {
        this.domService.activitySelectedTypes$.subscribe((activity_selected_types) => {
          this.domService.activityLeadingTypes$.subscribe(activity_leading_types => {
            this.mergeObjectModelInfo(object_model_info, otype_local_models, activity_selected_types, activity_leading_types)
          })
        })
      })
    })
  }

  mergeObjectModelInfo(
    object_model_info: ObjectModelInfo,
    otype_local_models: { [otype: string]: string[] },
    activity_selected_types: { [act: string]: string[] },
    activity_leading_types: { [act: string]: string | undefined }
  ) {
    //this.otypeLocalModels = otype_local_models
    let types: string[] = []
    Object.keys(activity_selected_types).forEach((act) => {
      types = types.concat(activity_selected_types[act])
    })
    types = [... new Set(types.flat())]
    let selected_seed_type = types.find(x => x == object_model_info.selectedSeedType) ?
      object_model_info.selectedSeedType : undefined
    let non_emitting_types = object_model_info.nonEmittingTypes.filter(net => types.find(ot => ot == net))
    let number_of_objects = selected_seed_type ? object_model_info.numberOfObjects : 0
    let execution_model_depth = object_model_info.executionModelDepth
    //! TODO
    let execution_model_evaluation_depth = object_model_info.executionModelEvaluationDepth
    let merged_object_model_info = new ObjectModelInfo(
      types,
      selected_seed_type,
      non_emitting_types,
      number_of_objects,
      execution_model_depth,
      execution_model_evaluation_depth,
      activity_selected_types,
      activity_leading_types
    )
    this.objectModelInfo = merged_object_model_info
  }

  ngOnDestroy() {
    this.domService.addObjectModelInfo(this.objectModelInfo)
  }

  onClickStart() {
    if (!(this.sessionKey && this.objectModelInfo.selectedSeedType)) {
      return
    }
    const formData = new FormData();
    formData.append("sessionKey", this.sessionKey);
    formData.append("seedType", this.objectModelInfo.selectedSeedType);
    formData.append("numberOfObjects", "" + this.objectModelInfo.numberOfObjects);
    formData.append("otypes", "" + this.objectModelInfo.otypes);
    formData.append("executionModelDepth", "" + this.objectModelInfo.executionModelDepth);
    // TODO
    formData.append("executionModelEvaluationDepth", "" + this.objectModelInfo.executionModelEvaluationDepth);
    formData.append("nonEmittingTypes", "" + this.objectModelInfo.nonEmittingTypes);
    Object.keys(this.objectModelInfo.activitySelectedTypes).forEach(act => {
      let leading_type = this.objectModelInfo.activityLeadingTypes[act]
      if (!leading_type) {
        return
      }
      let other_types = this.objectModelInfo.activitySelectedTypes[act].filter(x => x != leading_type)
      let act_types = [leading_type].concat(other_types)
      formData.append("act:" + act, "" + act_types)
    })

    const upload$ = this.appService.postObjectModelGeneration(this.sessionKey, formData).subscribe(
      (resp) => {
        let omgResponse = new ObjectModelGenerationResponse(resp)
        this.responseValid = true
        this.omgResponse = omgResponse
        if (!this.selectedObjectType) {
          this.selectedObjectType = Object.keys(omgResponse.stats)[0]
        }
        this.cachedSelectedObjectType = undefined
        this.cachedselectedParameterType = undefined
        this.reloadStats = true
        this.onChangeStatsInput()
      });
  }

  onChangeSelectedSeedType() { }

  onChangeNonEmittingTypes() { }

  getSelectedParameterType() {
    let parameter_type = this.selectedParameterType == "Cardinalities" ? "CARDINALITY" :
      this.selectedParameterType == "Timing Information" ? "TIMING" :
        this.selectedParameterType == "Object Attributes" ? "OBJECT_ATTRIBUTE" : null
    if (parameter_type == null) {
      throw ("Invalid value for selectedParameterType")
    }
    return parameter_type
  }

  getSelectedObjectType() {
    let object_type = this.selectedObjectType
    if (!object_type) {
      throw ("Invalid value for selectedObjectType")
    }
    return object_type
  }

  onChangeStatsInput() {
    let session_key = this.domService.getSessionKey()
    if (!session_key || !this.selectedObjectType || !this.selectedParameterType) {
      return
    }
    if (this.cachedSelectedObjectType == this.selectedObjectType
      && this.cachedselectedParameterType == this.selectedParameterType
      && !this.reloadStats) {
      // check if anything has changed because handler is called merely on click (not only on change)
      return
    }
    this.reloadStats = false
    this.cachedSelectedObjectType = this.selectedObjectType
    this.cachedselectedParameterType = this.selectedParameterType
    let parameter_type = this.getSelectedParameterType()
    let object_type = this.getSelectedObjectType()
    this.getParameters(session_key, object_type, parameter_type)
  }

  getParameters(session_key: string, object_type: string, parameter_type: string) {
    this.appService.getParameters(session_key, object_type, parameter_type)
      .subscribe((om_stats: {
        "err": any,
        "resp": {
          [attribute: string]: {
            //label: string // == attribute
            xAxis: string[]
            yAxes: ObjectModelStats,
            includeModeled: boolean,
            includeSimulated: boolean,
          }
        },
      }) => {
        let resp = om_stats["resp"]
        this.barChartData = {}
        this.attributes = []
        let so = [...Object.keys(resp)]
        so.sort((a, b) => a.length - b.length)
        so.forEach(attribute => {
          let params = resp[attribute]
          this.attributes = this.attributes.concat(attribute)
          this.touchedParameters[attribute] = false
          this.trainingSelectionMap[attribute] = params.includeModeled
          this.trainingModelMap[attribute] = "---"
          // TODO
          this.attributeModelCandidates[attribute] = ["Custom", "Normal", "Poisson"]
          let label_data: { data: number[], label: 'Log-Based' | 'Modeled' | 'Simulated' }[] = [
            { data: params.yAxes.LOG_BASED, label: 'Log-Based' },
          ]
          if (params.includeModeled) {
            label_data = label_data.concat([
              { data: params.yAxes.MODELED, label: 'Modeled' },
            ])
          }
          if (params.includeSimulated) {
            label_data = label_data.concat([
              { data: params.yAxes.SIMULATED, label: 'Simulated' },
            ])
          }
          this.barChartData[attribute] = label_data
          this.mbarChartLabels[attribute] = resp[attribute].xAxis
        })
      })
  }


  onClickInitialize() {
    if (!this.sessionKey){ // && this.objectModelInfo.selectedSeedType)) {
      return
    }
    const formData = new FormData();
    formData.append("sessionKey", this.sessionKey);
    //formData.append("seedType", this.objectModelInfo.selectedSeedType);
    formData.append("otypes", "" + this.objectModelInfo.otypes);
    formData.append("executionModelDepth", "" + this.objectModelInfo.executionModelDepth);
    // TODO
    formData.append("executionModelEvaluationDepth", "" + this.objectModelInfo.executionModelEvaluationDepth);
    formData.append("nonEmittingTypes", "" + this.objectModelInfo.nonEmittingTypes);
    Object.keys(this.objectModelInfo.activitySelectedTypes).forEach(act => {
      let leading_type = this.objectModelInfo.activityLeadingTypes[act]
      if (!leading_type) {
        return
      }
      let other_types = this.objectModelInfo.activitySelectedTypes[act].filter(x => x != leading_type)
      let act_types = [leading_type].concat(other_types)
      formData.append("act:" + act, "" + act_types)
    })

    const initialize$ = this.appService.initializeObjectGenerator(this.sessionKey, formData).subscribe(
      (resp) => {
        if (!this.selectedObjectType) {
          this.selectedObjectType = this.objectModelInfo.otypes[0]
        }
        if (!this.selectedParameterType) {
          this.selectedParameterType = this.statsTypes[0]
        }
        this.cachedSelectedObjectType = undefined
        this.cachedselectedParameterType = undefined
        this.onChangeStatsInput()
        this.initialized = true
      });
  }

  onClickSkip() {
    this.domService.setUseOriginalMarking(true)
    if (this.domService.netConfigValid) {
      this.router.navigate(['simulate_process'])
    } else {
      this.router.navigate(['discover_ocpn'])
    }
  }

  onClickConfirm() {
    this.domService.setObjectModelValid(true)
    this.domService.setUseOriginalMarking(false)
    if (this.domService.netConfigValid) {
      this.router.navigate(['simulate_process'])
    } else {
      this.router.navigate(['discover_ocpn'])
    }
  }

  onSelectForTraining(attribute: string) {
    if (!this.sessionKey) {
      return
    }
    if (this.trainingModelMap[attribute] == "---") {
      this.trainingModelMap[attribute] = this.attributeModelCandidates[attribute][0]
    }
    let parameter_type = this.getSelectedParameterType()
    let object_type = this.getSelectedObjectType()
    let selected = this.trainingSelectionMap[attribute]
    this.appService.selectForTraining(this.sessionKey, object_type, parameter_type, attribute, selected).subscribe((om_stats: {
      "err": any,
      "resp": AttributeParametrizationResponse,
    }) => {
      let parameters = om_stats["resp"]
      this.updateAttributeParametrization(attribute, parameters)
    })
  }

  onChangeFittingModel(attribute: string) {
    if (!this.sessionKey) {
      return
    }
    let parameter_type = this.getSelectedParameterType()
    let object_type = this.getSelectedObjectType()
    let fitting_model = this.trainingModelMap[attribute].toUpperCase()
    this.appService.switchModel(this.sessionKey, object_type, parameter_type, attribute, fitting_model).subscribe((om_stats: {
      "err": any,
      "resp": AttributeParametrizationResponse,
    }) => {
      let parameters = om_stats["resp"]
      this.updateAttributeParametrization(attribute, parameters)
    })
  }

  onClickApplyParameters(attribute: string) {
    if (!this.sessionKey) {
      return
    }
    let parameter_type = this.getSelectedParameterType()
    let object_type = this.getSelectedObjectType()
    let parameters = this.parameterMap[attribute]
    this.appService.changeParameters(this.sessionKey, parameter_type, object_type, attribute, parameters).subscribe((om_stats: {
      "err": any,
      "resp": AttributeParametrizationResponse,
    }) => {
      let parameters = om_stats["resp"]
      this.updateAttributeParametrization(attribute, parameters)
    })
  }

  updateAttributeParametrization(attribute: string, attribute_parameters: AttributeParametrizationResponse) {
    let include_modeled = attribute_parameters.includeModeled
    let include_simulated = attribute_parameters.includeSimulated
    this.trainingSelectionMap[attribute] = include_modeled
    this.touchedParameters[attribute] = false
    if (include_modeled) {
      this.parameterMap[attribute] = attribute_parameters.parameters
    }
    let y_axes = attribute_parameters.yAxes
    // TODO
    let label_data: { data: number[], label: 'Log-Based' | 'Modeled' | 'Simulated' }[] = [
      { data: y_axes.LOG_BASED, label: 'Log-Based' },
    ]
    if (include_modeled) {
      label_data = label_data.concat([
        { data: y_axes.MODELED, label: 'Modeled' },
      ])
    }
    if (include_simulated) {
      label_data = label_data.concat([
        { data: y_axes.SIMULATED, label: 'Simulated' },
      ])
    }
    this.barChartData[attribute] = label_data
    this.mbarChartLabels[attribute] = attribute_parameters.xAxis
  }

  onTouchParameters(attribute: string) {
    this.touchedParameters[attribute] = true
  }

  getNumberOfGeneratedObjects(otype: string) {
    if (!this.omgResponse) {
      return ""
    }
    if (!this.omgResponse.stats) {
      return ""
    }
    if (!(otype in this.omgResponse.stats)) {
      return ""
    }
    return this.omgResponse.stats[otype]["simulation_stats"]["number_of_objects"]
  }

  getOtherTypes(otype: string) {
    return this.objectModelInfo.otypes.filter(ot => ot != otype)
  }

  getPlotCaption(chart_label: string) {
    let selected_plot_type = this.selectedObjectType
    let selected_stats_type = this.selectedParameterType
    if (selected_stats_type == "Cardinalities") {
      return chart_label
    }
    return "Related " + chart_label + " arriving relative to " + selected_plot_type
  }

}
