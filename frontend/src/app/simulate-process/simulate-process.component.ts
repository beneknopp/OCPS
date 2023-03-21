import { Component, OnInit } from '@angular/core';
import { faPlay } from '@fortawesome/free-solid-svg-icons';
import * as shape from 'd3-shape';
import { AppService } from '../app.service';
import { DOMService } from '../dom.service';
import { Arc } from '../net-utils/arc';
import { OcpnGraphLink, OcpnGraphNode } from '../net-utils/ocpn-graph';
import { Place } from '../net-utils/place';
import { Transition } from '../net-utils/transition';
import { SimulationStateDto } from '../sim-utils/simulation-state';


@Component({
  selector: 'app-simulate-process',
  templateUrl: './simulate-process.component.html',
  styleUrls: ['./simulate-process.component.css', '../app.component.css']
})
export class SimulateProcessComponent implements OnInit {

  places: Place[] = []
  transitions: Transition[] = []
  arcs: Arc[] = []
  nodes: OcpnGraphNode[] = []
  links: OcpnGraphLink[] = []
  curve: any = shape.curveBasis
  numberOfSteps: number = 10
  otypes: string[] = []
  clock = "---"
  steps = "---"
  footer_info = "Waiting for Initialization"
  initialized = false
  simulationState: SimulationStateDto | undefined;
  placeTooltips: { [place_id: string]: string } = {};
  useOriginalMarking = true
  useGeneratedMarking = false

  startTransitionIcon = faPlay

  constructor(
    private appService: AppService,
    public domService: DOMService
  ) { }

  ngOnInit(): void {
    if (!this.domService.useOriginalMarking) {
      this.useOriginalMarking = false
      this.useGeneratedMarking = true
    }
    this.domService.activityLeadingTypes$.subscribe((activity_leading_types) => {
      this.domService.ocelInfo$.subscribe((ocelInfo) => {
        this.otypes = ocelInfo.otypes
        this.domService.ocpnInfo$.subscribe((ocpnInfo) => {
          this.places = ocpnInfo.places
          this.transitions = ocpnInfo.transitions
          this.arcs = ocpnInfo.arcs
          let ocpn_graph = this.domService.makeOcpnGraph(this.places, this.transitions, this.arcs,
            this.otypes, activity_leading_types)
          this.nodes = ocpn_graph[0]
          this.links = ocpn_graph[1]
        })
      })
    })
  }

  onClickInitialize() {
    let session_key = this.domService.getSessionKey()
    if (!session_key) {
      return
    }
    let useOriginalMarking = this.useOriginalMarking
    this.appService.initializeSimulation(session_key, useOriginalMarking).subscribe((resp) => {
      this.simulationState = new SimulationStateDto(resp.resp)
      this.makeTooltips()
      this.initialized = true
    })
  }

  onClickStartSimulation(steps: number) {
    let session_key = this.domService.getSessionKey()
    if (!session_key) {
      throw Error("Simulation started without valid session key")
    }
    this.appService.startSimulation(steps, session_key).subscribe((resp) => {
      this.simulationState = new SimulationStateDto(resp.resp)
      this.makeTooltips()
    })
  }

  concatObjectMarkingInfos(ot_map: any) {
    let info = ""
    let otypes = Object.keys(ot_map)
    if (otypes.length == 0) {
      return info
    }
    otypes.map(ot => {
      let num = ot_map[ot] + ""
      info += num + " " + ot + ", "
    })
    info = info.slice(0, -2)
    return info
  }

  getColor(otype: string) {
    let index = this.otypes.indexOf(otype)
    return this.domService.getColor(index)
  }

  getMarkerEndId(otype: string) {
    return otype + '-arrow'
  }

  getMarkerUrl(link: any) {
    let otype = link.otype
    let marker_end_id = this.getMarkerEndId(otype)
    return "url(#" + marker_end_id + ")"
  }

  isInitialTransition(node: any) {
    console.log(node)
    return true
  }

  onClickAbort() {
    this.simulationState = undefined
    this.footer_info = "Waiting for Initialization"
    this.initialized = false
  }

  onClickOCELExport() {
    let session_key = this.domService.getSessionKey()
    if (!session_key) {
      throw Error("OCEL Export started without valid session key")
    }
    this.appService.exportOCEL(session_key).subscribe(resp => {
      let fileName = resp.headers.get("content-disposition")
        ?.split(";")[1].split("=")[1];
      let blob : Blob = resp.body as Blob;
      let a = document.createElement('a');
      a.download = !!fileName ? fileName : "";
      a.href = window.URL.createObjectURL(blob)
      a.click()
    })
  }

  makeTooltips() {
    if (!this.simulationState) {
      return
    }
    this.placeTooltips = {}
    const activeTokens = this.simulationState.activeTokens
    const markingInfo = this.simulationState.markingInfo
    let activePerPlace: any = {}
    activeTokens.forEach(t => {
      let place_id = t.place_id
      if (!Object.keys(this.placeTooltips).find(p_id => p_id == place_id)) {
        let tooltip = "Tokens (Object ID - Timestamp):"
        this.placeTooltips[place_id] = tooltip
        activePerPlace[place_id] = 0
      }
      this.placeTooltips[place_id] += "\n" + t.oid + " @"
      this.placeTooltips[place_id] += t.time >= 0 ? "+" : ""
      this.placeTooltips[place_id] += t.time
      activePerPlace[place_id] += 1
    })
    for (let place_id in markingInfo) {
      if (!Object.keys(this.placeTooltips).find(p_id => p_id == place_id)) {
        this.placeTooltips[place_id] = "Total: " + markingInfo[place_id]
      } else {
        let more = markingInfo[place_id] - activePerPlace[place_id]
        this.placeTooltips[place_id] += more > 0 ? "\n ... and " + more + " more" : ""
      }
    }
  }

  getTransitionBorderColor(node: OcpnGraphNode) {
    if (!this.simulationState || node.type != "TRANSITION") {
      return ""
    }
    let transition_id = node.id
    let next_transition = this.simulationState.bindings[0][1]
    if (next_transition == node.id) {
      return "greenyellow"
    }
    let will_fire = this.simulationState.bindings.find(b => b[1] == transition_id)
    if (will_fire) {
      return "yellow"
    }
    return ""
  }

  getTransitionBorderWidth(node: OcpnGraphNode) {
    if (!this.simulationState || node.type != "TRANSITION") {
      return "0"
    }
    let transition_id = node.id
    let will_fire = this.simulationState.bindings.find(b => b[1] == transition_id)
    if (will_fire) {
      return "3"
    }
    return "0"
  }

  hasTokens(node: OcpnGraphNode) {
    return node.type == 'PLACE' && this.simulationState && this.simulationState.markingInfo[node.id] > 0
  }

  getNodeLabel(node: OcpnGraphNode) {
    if (node.transitionType) {
      if (node.transitionType == "ACTIVITY") {
        return node.label
      }
      return node.id 
    } 
    if (node.type == 'PLACE') {
      return node.label
      //let place = this.places.find(p => p.id == node.id)
      //if (place && place?.isInitial) {
      //   return place.otype
      //}
    }
    return ""
  }

  getNodeLabelX(node: OcpnGraphNode) {
    if (node.type == 'PLACE') {
      return -20
    }
    return 10
  }

  getNodeLabelY(node: OcpnGraphNode) {
    if (node.type == 'PLACE') {
      return 30
    }
    return 20
  }

  getTooltip(place: Place) {
    if (!Object.keys(this.placeTooltips).find(p_id => p_id == place.id)) {
      return ""
    }
    return this.placeTooltips[place.id]
  }

  getClock() {
    if (!this.simulationState) {
      return "---"
    }
    return "" + this.simulationState.clock
  }

  getSteps() {
    if (!this.simulationState) {
      return "---"
    }
    return "" + this.simulationState.steps
  }

  getFooter() {
    if (!this.simulationState) {
      return "Waiting for Initialization..."
    }
    const total = this.simulationState.totalObjects
    const initialized = this.simulationState.objectsInitialized
    const terminated = this.simulationState.objectsTerminated
    let footer = "Total number of objects: "
    Object.keys(total).forEach(otype => {
      footer += otype + ": " + total[otype] + "; "
    })
    footer += "Initialized: "
    Object.keys(initialized).forEach(otype => {
      footer += otype + ": " + initialized[otype] + "; "
    })
    footer += "Terminated: "
    Object.keys(terminated).forEach(otype => {
      footer += otype + ": " + terminated[otype] + "; "
    })
    footer = footer.slice(0, -2) + "."
    return footer
  }


}
