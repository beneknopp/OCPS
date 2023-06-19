import { Component, OnInit } from '@angular/core';
import { AppService } from '../app.service';
import { DOMService } from '../dom.service';

@Component({
  selector: 'app-evaluate',
  templateUrl: './evaluate.component.html',
  styleUrls: ['./evaluate.component.css', '../app.component.css']
})
export class EvaluateComponent implements OnInit {

  sessionKey: string | undefined
  originalCycleTimes: { [otype: string]: { "mean": number, "stdev": number } } = {}
  simulatedCycleTimes: { [otype: string]: { "mean": number, "stdev": number } } = {}
  earthMoversConformances: { [otype: string]: number } = {}
  availableSimulationsObjectModels: string[] = ["ORIGINAL"]
  selectedObjectModels: string[] = []

  otypes: string[] = []
  acts: string[] = []
  statsTypes: string[] = ["Earth Mover's Conformance", "Cycle Times", "Activity Delays"]
  statsTypesKeyMap: { [statsType: string]: string } = {
    "Earth Mover's Conformance": "earthmovers",
    "Cycle Times": "cycletimes",
    "Activity Delays": "actdelays"
  }

  selectedStatisticsType: string | undefined
  selectedObjectType: string | undefined
  showStats: boolean = false;
  statsHeader: string = "";
  statsAxesDescription: string = "";
  
  constructor(
    private appService: AppService,
    public domService: DOMService
  ) { }

  ngOnInit(): void {
    this.sessionKey = this.domService.getSessionKey()
    this.domService.ocelInfo$.subscribe(ocelInfo => {
      this.otypes = ocelInfo.otypes
      this.acts = ocelInfo.acts
      if (!this.sessionKey) {
        return
      }
      this.appService.getAvailableSimulationsObjectModels(this.sessionKey).subscribe((res: { "resp": string[], "err": any }) => {
        let availableModels = res["resp"]
        this.availableSimulationsObjectModels = availableModels.concat(["ORIGINAL"])
      })
    })
  }

  onConfirmModelSelection() {
    if (!this.sessionKey) { return }
    //return
    this.appService.updateEvaluationSelectedObjectModels(this.sessionKey, this.selectedObjectModels).subscribe((res) => {})
  }

  public attributes: string[] = []
  public barChartData: { [attribute: string]: { data: number[], label: string }[] } = {};
  public mbarChartLabels: { [attribute: string]: string[] } = {};
  public barChartType: string = 'bar';
  public barChartOptions: any = {
    scaleShowVerticalLines: true,
    responsive: true,
    axisX: {
      title: "Number of Simulation Steps",
      },
      axisY: { 
        title: "Precipitation (inches)"                   
      },
      theme: "light2"
  };
  public barChartColors: Array<any> = [
    {
      backgroundColor: 'rgba(145,19,177,0.2)',
      borderColor: 'rgba(105,159,177,1)',
      pointBackgroundColor: 'rgba(105,159,177,1)',
      pointBorderColor: '#fafafa',
      pointHoverBackgroundColor: '#fafafa',
      pointHoverBorderColor: 'rgba(105,159,177)'
    },
    {
      backgroundColor: 'rgba(77,200,96,0.3)',
      borderColor: 'rgba(77,20,96,1)',
      pointBackgroundColor: 'rgba(77,20,96,1)',
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: 'rgba(77,20,96,1)'
    }
  ];


  onChangeStatsInput() {
    let otype = this.selectedObjectType
    let stats_type = this.selectedStatisticsType 
    if (!this.sessionKey || !stats_type || !otype) { return }
    let stats_type_key = this.statsTypesKeyMap[stats_type]
    this.showStats = false
    this.statsHeader = ""
    this.appService.getEvaluation(this.sessionKey, stats_type_key, otype).subscribe(
      (res: {
        "err": any, "resp": any
      }) => {
        let resp = res["resp"]
        this.barChartData = resp["chartData"]
        this.mbarChartLabels = resp["axes"]
        this.attributes = Object.keys(this.barChartData)
        this.showStats = true
        this.statsHeader = this.getStatsTypesHeaders(otype, stats_type)
        this.statsAxesDescription = this.getStatsAxesDescription(otype, stats_type)
      })
  }

  getStatsTypesHeaders(otype: string | undefined, stats_type: string | undefined) {
    if (stats_type == "Earth Mover's Conformance") {
      return "Stochastic conformance of simulated logs against original data"
    }
    if (stats_type == "Activity Delays") {
      return "Delays (time until next activity) for '" + otype + "'"
    }
    if (stats_type == "Cycle Times") {
      return "Cycle times (throughput times / trace durations) for '" + otype + "'"
    }
    return "";
  };

  getStatsAxesDescription(otype: string| undefined, stats_type: string | undefined) {
    if (stats_type == "Earth Mover's Conformance") {
      return "X: Number of steps in simulation run; Y: Earth Mover's Conformance"
    }
    if (stats_type == "Activity Delays") {
      return "X: Number of steps in simulation run; Y: Delays (time)"
    }
    if (stats_type == "Cycle Times") {
      return "X: Number of steps in simulation run; Y: Cycle times"
    }
    return "";
  }

}
