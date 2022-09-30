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

  private object_color_set = ['#F9AA19', '#9FD1D0', '#F39891', '#C8BFE7' ,'#D0CECE', '#A2DEF4', '#0070C0']
  ocelConfigValid = false
  ocpnConfigValid = false
  objectModelValid = false
  netConfigValid = false;
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
    return this.sessionKey
  }

  setSessionKey(sessionKey: string) {
    this.sessionKey = sessionKey
  }

  getOtypeColor(otype: string, otypes: string[]){
    return this.object_color_set[otypes.indexOf(otype) % this.object_color_set.length]
  }

  makeOcpnGraph(
    places: Place[],
    transitions: Transition[],
    arcs: Arc[],
    otypes: string[],
    activityLeadingTypes: { [act: string]: string }
  ): [OcpnGraphNode[], OcpnGraphLink[]] {
    let nodes: OcpnGraphNode[] = []
    let links: OcpnGraphLink[] = []
    let col_codes: { [otype: string]: string } = {}
    otypes.forEach(otype => {
      col_codes[otype] = this.getOtypeColor(otype, otypes)
    })
    places.forEach(place => {
      nodes = nodes.concat({
        id: place.id,
        label: place.isInitial ? place.otype : place.id,
        type: 'PLACE',
        color: col_codes[place.otype],
        x: 20.0,
        y: 312.0
      })
    })
    transitions.forEach(transition => {
      let leading_type: string | undefined = ""
      if (transition.transitionType == 'ACTIVITY') {
        leading_type = activityLeadingTypes[transition.id]
      } else {
        let any_arc = arcs.find(arc => arc.source == transition.id || arc.target == transition.id)
        let any_place_id = any_arc?.source == transition.id ? any_arc.target : any_arc?.source
        let any_place = places.find(place => place.id == any_place_id)
        leading_type = any_place?.otype
      }
      let color = leading_type ? col_codes[leading_type] : '#000000'
      nodes = nodes.concat({
        id: transition.id,
        label: transition.label,
        type: 'TRANSITION',
        transitionType: transition.transitionType,
        color: color,
        x: 20.0,
        y: 312.0
      })
    })
    arcs.forEach(arc => {
      let try_source_place = places.find(p => p.id == arc.source)
      let arc_place = try_source_place ? try_source_place : places.find(p => p.id == arc.target)
      let arc_otype = arc_place ? arc_place.otype : ""
      links = links.concat({
        source: arc.source,
        target: arc.target,
        label: "",
        otype: arc_otype,
        isVariableArc: arc.isVariableArc,
        color: col_codes[arc_otype]
      })
    })
    return [nodes, links]
  }

  getColor(index: number) {
    return this.object_color_set[index %  this.object_color_set.length]
  }

  setOcelConfigValid(valid: boolean) {
    this.ocelConfigValid = valid
  }


  setObjectModelValid(valid: boolean) {
    this.objectModelValid = valid
    if (!valid) {
      this.netConfigValid = false
    }
    if (this.ocpnConfigValid && this.objectModelValid) {
      this.netConfigValid = true
    }
  }

  setOcpnConfigValid(valid: boolean) {
    this.ocpnConfigValid = valid
    if (!valid) {
      this.netConfigValid = false
    }
    if (this.ocpnConfigValid && this.objectModelValid) {
      this.netConfigValid = true
    }
  }


}