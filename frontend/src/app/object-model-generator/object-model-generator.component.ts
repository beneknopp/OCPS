import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { AppService } from '../app.service';
import { DOMService } from '../dom.service';
import { ObjectModelGenerationResponse, ObjectModelInfo, ObjectModelStats } from './object-model-info';

@Component({
  selector: 'app-object-model-generator',
  templateUrl: './object-model-generator.component.html',
  styleUrls: ['./object-model-generator.component.css', '../app.component.css']
})
export class ObjectModelGeneratorComponent implements OnInit {

  objectModelInfo: ObjectModelInfo = new ObjectModelInfo()
  configValid = true
  responseValid = false;
  selectedSeedType = undefined
  nonEmittingTypes: string[] = []
  sessionKey: string | undefined;
  numberOfObjects: number = 0;
  selectedPlotType: string | undefined
  selectedStatsType: string | undefined
  cachedSelectedPlotType: string | undefined
  cachedSelectedStatsType: string | undefined
  statsTypes = ["Cardinalities", "Relative Arrival Times (in seconds)"]


  public barChartData: { [otype: string]: { data: number[], label: 'Log-Based' | 'Simulated' }[] } = {
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
    let merged_object_model_info = new ObjectModelInfo(
      types,
      selected_seed_type,
      non_emitting_types,
      number_of_objects,
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
        if (!this.selectedPlotType) {
          this.selectedPlotType = Object.keys(omgResponse.stats)[0]
        }
        if (!this.selectedStatsType) {
          this.selectedStatsType = this.statsTypes[0]
        }
        this.cachedSelectedPlotType = undefined
        this.cachedSelectedStatsType = undefined
        this.onChangeStatsInput()
      });
  }

  onChangeSelectedSeedType() { }

  onChangeNonEmittingTypes() { }

  onChangeStatsInput() {
    let session_key = this.domService.getSessionKey()
    if (!session_key || !this.selectedPlotType || !this.selectedStatsType) {
      return
    }
    if (this.cachedSelectedPlotType == this.selectedPlotType && this.cachedSelectedStatsType == this.selectedStatsType) {
      // check if anything has changed because handler is called merely on click (not only on change)
      return
    }
    this.cachedSelectedPlotType = this.selectedPlotType
    this.cachedSelectedStatsType = this.selectedStatsType
    const request$ = this.selectedStatsType == "Cardinalities" ?
      this.appService.getObjectModelStats(session_key, this.selectedPlotType) :
      this.appService.getArrivalTimesStats(session_key, this.selectedPlotType)
    request$.subscribe((om_stats: {
      "err": any,
      "resp": { [otype: string]: ObjectModelStats }
    }) => {
      let resp = om_stats["resp"]
      this.barChartData = {}
      Object.keys(resp).forEach(ot => {
        this.barChartData[ot] = [
          { data: resp[ot].log_based, label: 'Log-Based' },
          { data: resp[ot].simulated, label: 'Simulated' }
        ]
        this.mbarChartLabels[ot] = resp[ot].x_axis
      })
    })
  }

onClickConfirm() {
  this.domService.setObjectModelValid(true)
  if (this.domService.netConfigValid) {
    this.router.navigate(['simulate_process'])
  } else {
    this.router.navigate(['discover_ocpn'])
  }
}

getNumberOfGeneratedObjects(otype: string) {
  if (!this.omgResponse) {
    return ""
  }
  return this.omgResponse.stats[otype]["simulation_stats"]["number_of_objects"]
}

getOtherTypes(otype: string) {
  return this.objectModelInfo.otypes.filter(ot => ot != otype)
}

getPlotCaption(otype: string) {
  let selected_plot_type = this.selectedPlotType
  let selected_stats_type = this.selectedStatsType
  if (selected_stats_type == "Cardinalities") {
    return otype + " per " + selected_plot_type
  }
  return "Related " + otype + " arriving relative to " + selected_plot_type
}

}
