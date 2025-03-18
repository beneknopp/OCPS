import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, Subject } from 'rxjs';
import { filter, first } from 'rxjs/operators';
import { OcpnInfo } from './discover-ocpn/ocpn-info';
import { OCELInfo } from './log-upload/ocel-info';
import { Arc } from './net-utils/arc';
import { OcpnDto } from './net-utils/ocpn';
import { OcpnGraphLink, OcpnGraphNode } from './net-utils/ocpn-graph';
import { Place } from './net-utils/place';
import { Transition } from './net-utils/transition';
import { ObjectModelInfo } from './object-model-generator/object-model-info';

@Injectable({
  providedIn: 'root'
})
export class DOMService {

  private activitySelectedTypes: Subject<{ [act: string]: string[] }> = new BehaviorSubject<{ [act: string]: string[] }>({});
  private activityLeadingTypes: Subject<{ [act: string]: string }> =
    new BehaviorSubject<{ [act: string]: string }>({});
  private otypeLocalModels: Subject<{ [otype: string]: string[] }> = new BehaviorSubject<{ [otype: string]: string[] }>({})
  private ocelInfo: Subject<OCELInfo> = new BehaviorSubject<OCELInfo>(new OCELInfo())
  private objectModelInfo: Subject<ObjectModelInfo> = new BehaviorSubject<ObjectModelInfo>(new ObjectModelInfo())
  private ocpnInfo: Subject<OcpnInfo> = new BehaviorSubject<OcpnInfo>(new OcpnDto())
  private sessionKey: string | undefined;

  private object_color_set = ['#F9AA19', '#9FD1D0', '#F39891', '#C8BFE7', '#D0CECE', '#A2DEF4', '#0070C0']
  ocelConfigValid = false
  ocpnConfigValid = false
  objectModelValid = false
  useOriginalMarking = true
  netConfigValid = false;
  evaluationEnabled = false
  public step = 1;

  get activitySelectedTypes$() {
    let activitySelectedTypes: Observable<{ [act: string]: string[] }>
      = this.activitySelectedTypes.asObservable().pipe(first(), filter(otypes => !!otypes));
    return activitySelectedTypes
  }

  get activityLeadingTypes$() {
    let activityLeadingTypes: Observable<{ [act: string]: string }>
      = this.activityLeadingTypes.asObservable().pipe(first(), filter(leadingTypes => !!leadingTypes));
    return activityLeadingTypes
  }

  get ocelInfo$() {
    let ocelInfo: Observable<OCELInfo> = this.ocelInfo.asObservable().pipe(first(), filter(ocelInfo => !!ocelInfo))
    return ocelInfo
  }

  get otypeLocalModels$() {
    let otypeLocalModels: Observable<{ [otype: string]: string[] }> = this.otypeLocalModels.asObservable().pipe(first(), filter(otypeLocalModel => !!otypeLocalModel))
    return otypeLocalModels
  }

  get objectModelInfo$() {
    let objectModelInfo: Observable<ObjectModelInfo> = this.objectModelInfo.asObservable()
      .pipe(first(), filter(objectModelInfo => !!objectModelInfo))
    return objectModelInfo
  }

  get ocpnInfo$() {
    let ocpnInfo: Observable<OcpnInfo> = this.ocpnInfo.asObservable()
      .pipe(first(), filter(ocpnInfo => !!ocpnInfo))
    return ocpnInfo
  }

  addActivitySelectedTypes(data: { [act: string]: string[] }) {
    this.activitySelectedTypes.next(data);
  }

  addActivityLeadingTypes(data: { [act: string]: string }) {
    this.activityLeadingTypes.next(data);
  }

  addOtypeLocalModels(data: { [otype: string]: string[] }) {
    this.otypeLocalModels.next(data);
  }

  addOcelInfo(data: OCELInfo) {
    this.ocelInfo.next(data);
  }

  addObjectModelInfo(data: ObjectModelInfo) {
    this.objectModelInfo.next(data);
  }

  addOcpnInfo(data: OcpnInfo) {
    this.ocpnInfo.next(data)
  }

  getSessionKey() {
    //return "430"
    return this.sessionKey
  }

  setSessionKey(sessionKey: string) {
    this.sessionKey = sessionKey
  }

  getOgraphGenParameterTypeColor(parameter_type: string){
    // TODO
    let parameter_types = ["Log-Based", "Modeled", "Simulated"]
    return this.object_color_set[parameter_types.indexOf(parameter_type)]
  }

  getOtypeColor(otype: string, otypes: string[]) {
    return this.object_color_set[otypes.indexOf(otype) % this.object_color_set.length]
  }

  makeOcpnGraph(
    places: Place[],
    transitions: Transition[],
    arcs: Arc[],
    otypes: string[],
    activityLeadingTypes: { [act: string]: string }
  ): [OcpnGraphNode[], OcpnGraphLink[], { [otype: string]: OcpnGraphNode[] }, { [otype: string]: OcpnGraphLink[] }] {
    let nodes: OcpnGraphNode[] = []
    let links: OcpnGraphLink[] = []
    let flatNodes: { [otype: string]: OcpnGraphNode[] } = {}
    let flatLinks: { [otype: string]: OcpnGraphLink[] } = {}
    let col_codes: { [otype: string]: string } = {}
    otypes.forEach(otype => {
      col_codes[otype] = this.getOtypeColor(otype, otypes)
      flatNodes[otype] = []
      flatLinks[otype] = []
    })
    places.forEach(place => {
      let otype = place.otype
      let node = {
        id: place.id,
        label: place.isInitial ? otype : place.id,
        type: 'PLACE',
        color: col_codes[otype],
        x: 20.0,
        y: 312.0
      }
      flatNodes[otype] = flatNodes[otype].concat(node)
      nodes = nodes.concat({...node})
    })
    transitions.forEach(transition => {
      let leading_type: string | undefined = ""
      let adjacentArcs = arcs.filter(arc => arc.source == transition.id || arc.target == transition.id)
      let adjacentNodes = adjacentArcs.map(arc => arc.source).concat(adjacentArcs.map(arc => arc.target))
      let adjacentPlaces = places.filter(place => adjacentNodes.find(node => node == place.id))
      let adjacentOtypes = adjacentPlaces.map(place => place.otype)
      let otypes = [...new Set(adjacentOtypes)]
      if (transition.transitionType == 'ACTIVITY') {
        leading_type = activityLeadingTypes[transition.id]
      } else {
        leading_type = otypes[0]
      }
      //let color = col_codes[leading_type]
      otypes.forEach(otype => {
        let color = col_codes[otype]
        let node = {
          id: transition.id,
          label: transition.label,
          type: 'TRANSITION',
          transitionType: transition.transitionType,
          color: color,
          x: 20.0,
          y: 312.0
        }
        flatNodes[otype] = flatNodes[otype].concat(node)
        if (otype == leading_type) {
          nodes = nodes.concat({...node})
        }
      })
    })
    arcs.forEach(arc => {
      let try_source_place = places.find(p => p.id == arc.source)
      let arc_place = try_source_place ? try_source_place : places.find(p => p.id == arc.target)
      let arc_otype = arc_place ? arc_place.otype : ""
      let link: OcpnGraphLink = {
        source: arc.source,
        target: arc.target,
        label: "",
        otype: arc_otype,
        isVariableArc: arc.isVariableArc,
        color: col_codes[arc_otype]
      }
      flatLinks[arc_otype] = flatLinks[arc_otype].concat(link)
      links = links.concat({...link})
    })
    return [nodes, links, flatNodes, flatLinks]
  }

  getColor(index: number) {
    return this.object_color_set[index % this.object_color_set.length]
  }

  setOcelConfigValid(valid: boolean) {
    this.ocelConfigValid = valid
  }

  setUseOriginalMarking(doUse: boolean) {
    this.useOriginalMarking = doUse
  }

  setObjectModelValid(valid: boolean) {
    this.objectModelValid = valid
  }

  setOcpnConfigValid(valid: boolean) {
    this.netConfigValid = valid
  }

  enableEvaluation() {
    this.evaluationEnabled = true
  }

}