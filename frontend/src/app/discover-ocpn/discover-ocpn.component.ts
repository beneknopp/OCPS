import { Component, OnInit, OnDestroy } from '@angular/core';
import { AppService } from '../app.service';
import { DOMService } from '../dom.service';
import * as shape from 'd3-shape';
import { OcpnDto } from '../net-utils/ocpn';
import { Place, PlaceDto } from '../net-utils/place';
import { Transition, TransitionDto } from '../net-utils/transition';
import { Arc, ArcDto } from '../net-utils/arc';
import { OcpnInfo } from './ocpn-info';
import { OcpnGraphNode, OcpnGraphLink } from '../net-utils/ocpn-graph'
import { Router } from '@angular/router';


@Component({
  selector: 'app-discover-ocpn',
  templateUrl: './discover-ocpn.component.html',
  styleUrls: ['./discover-ocpn.component.css', '../app.component.css']
})
export class DiscoverOcpnComponent implements OnInit {

  activityLeadingTypes: { [act: string]: string } = {}
  activitySelectedTypes: { [act: string]: string[] } = {}
  sessionKey: string | undefined;
  
  nodes: OcpnGraphNode[] = []
  links: OcpnGraphLink[] = []

  flatNodes: { [otype:string]: OcpnGraphNode[]} = {}
  flatLinks: { [otype:string]: OcpnGraphLink[]} = {}

  otypes: string[] = []
  curve: any = shape.curveBasis
  places: Place[] = [];
  transitions: Transition[] = [];
  arcs: Arc[] = [];
  precision: number | undefined;
  fitness: number | undefined;
  ocpnDiscovered = false

  constructor(
    private appService: AppService,
    private router: Router,
    private domService: DOMService
  ) { }

  ngOnInit(): void {
    this.sessionKey = this.domService.getSessionKey()
    this.domService.activityLeadingTypes$.subscribe((activity_leading_types) => {
      this.activityLeadingTypes = activity_leading_types
      this.domService.activitySelectedTypes$.subscribe((activity_selected_types) => {
        this.activitySelectedTypes = activity_selected_types
        this.domService.ocelInfo$.subscribe((ocelInfo) => {
          this.otypes = ocelInfo.otypes
          this.domService.ocpnInfo$.subscribe((ocpnInfo) => {
            this.places = ocpnInfo.places
            this.transitions = ocpnInfo.transitions
            this.arcs = ocpnInfo.arcs
            this.precision = ocpnInfo.precision
            this.fitness = ocpnInfo.fitness
            let ocpn_graph = this.domService.makeOcpnGraph(ocpnInfo.places, ocpnInfo.transitions, ocpnInfo.arcs,
              this.otypes, this.activityLeadingTypes)
            this.nodes = ocpn_graph[0]
            this.links = ocpn_graph[1]
          })
        })
      })
    })
  }

  ngOnDestroy() {
    this.domService.addOcpnInfo(
      new OcpnInfo(this.places, this.transitions, this.arcs, this.precision, this.fitness)
    )
  }

  onClickStart() {
    if (!(this.sessionKey)) {
      return
    }
    const form_data = new FormData();
    form_data.append("sessionKey", this.sessionKey);
    Object.keys(this.activitySelectedTypes).forEach(act => {
      let types = this.activitySelectedTypes[act]
      form_data.append("act:" + act, "" + types)
    })
    const upload$ = this.appService.discoverOCPN(this.sessionKey, form_data).subscribe((resp: OcpnDto) => {
      let place_dtos: PlaceDto[] = resp.places
      let transition_dtos: TransitionDto[] = resp.transitions
      let arc_dtos: ArcDto[] = resp.arcs
      let places = place_dtos.map(place_dto => new Place(place_dto))
      let transitions = transition_dtos.map(transition_dto => new Transition(transition_dto))
      let arcs = arc_dtos.map(arc_dto => new Arc(arc_dto))
      this.places = places
      this.transitions = transitions
      this.arcs = arcs
      let ocpn_graph = this.domService.makeOcpnGraph(places, transitions, arcs, this.otypes, this.activityLeadingTypes)
      this.nodes = ocpn_graph[0]
      this.links = ocpn_graph[1]
      this.flatNodes = ocpn_graph[2]
      this.flatLinks = ocpn_graph[3]
      this.precision = resp.precision
      this.fitness = resp.fitness
      this.ocpnDiscovered = true
    });
  }

  getColor(otype: string) {
    let index = this.otypes.indexOf(otype)
    return this.domService.getColor(index)
  }

  getMarkerEndId(otype: string) {
    return otype + '-arrow'
  }

  getMarkerUrl(otype: string) {
    let marker_end_id = this.getMarkerEndId(otype)
    return "url(#" + marker_end_id + ")"
  }

  onConfirm() {
    this.domService.setOcpnConfigValid(true)
    if (this.domService.netConfigValid) {
      this.router.navigate(['simulate_process'])
    }
  }

  getFooter() {
    let precision = this.precision ? this.precision + "" : "- - - - -"
    let fitness = this.fitness ? this.fitness + "" : "- - - - -"
    return "Evaluation against Input Log - Object-Centric Precision: " + precision + "; Object-Centric Fitness: " + fitness
  }

  isLastOtype(otype: string) {
    return this.otypes.indexOf(otype) == this.otypes.length - 1
  }

}
