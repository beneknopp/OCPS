import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { NgSelectConfig } from '@ng-select/ng-select';
import { cloneDeep, isEqual } from 'lodash';
import { AppService } from '../app.service';
import { DOMService } from '../dom.service';
import { OCELInfo, OCELInfoDTO } from './ocel-info';

@Component({
  selector: 'app-log-upload',
  templateUrl: './log-upload.component.html',
  styleUrls: ['./log-upload.component.css', '../app.component.css']
})
export class LogUploadComponent implements OnInit {

  ocelInitialized: boolean = true;
  graphBoxStyle = 'border-color: #ED1C24'
  type_nodes: { id: string, label: string, color: string }[] = []
  type_links: { source: string, target: string, label: string }[] = []
  ocelInfo: OCELInfo = new OCELInfo()
  chachedOcelInfo: OCELInfo | undefined
  otypes: string[] = []
  acts: string[] = []
  activity_allowed_types: { [act: string]: string[] } = {}
  activity_selected_types: { [act: string]: string[] } = {}
  activity_leading_type_groups: { [act: string]: string[] } = {}
  activity_leading_type_candidates_filtered: { [act: string]: string[] } = {}
  activity_leading_type_selections: { [act: string]: string | undefined } = {}
  otype_local_models: { [otype: string]: string[] } = {}
  configValid = false;
  filename : string | undefined
  leadingGroupsInfo: string | undefined

  constructor(
    private appService: AppService,
    private domService: DOMService,
    private router: Router,
    private selectConfig: NgSelectConfig
  ) {
  }

  ngOnInit(): void {
    this.domService.ocelInfo$.subscribe((ocelInfo) => {
      this.ocelInfo = ocelInfo
      this.onUpdateOCELInfo()
    })
  }

  ngOnChanges() {
    this.checkConfigurationValidity()
  }

  ngOnDestroy() {
    let activity_selected_types: {
      [act: string]: string[];
    } = {}
    Object.keys(this.ocelInfo.activity_selected_types).forEach((act) => {
      if (this.ocelInfo.activity_selected_types[act].length > 0) {
        activity_selected_types[act] = this.ocelInfo.activity_selected_types[act]
      }
    })
    let otype_local_models: { [otype: string]: string[] } = {}
    this.ocelInfo.otypes.forEach(otype => {
      otype_local_models[otype] = []
    })
    this.type_links.forEach(link => {
      let source_otype = this.type_nodes.find(node => node.id == link.source)?.label
      let target_otype = this.type_nodes.find(node => node.id == link.target)?.label
      if (!source_otype || !target_otype) { throw Error };
      otype_local_models[source_otype] = otype_local_models[source_otype].concat([target_otype])
      otype_local_models[target_otype] = otype_local_models[target_otype].concat([source_otype])
    })
    Object.keys(this.ocelInfo.otypes).forEach(otype => {
      otype_local_models[otype] = [...new Set(otype_local_models[otype])]
    })
    this.domService.addOtypeLocalModels(otype_local_models)
    this.domService.addActivitySelectedTypes(activity_selected_types)
    this.domService.addActivityLeadingTypes(this.ocelInfo.activity_leading_type_selections)
    this.domService.addOcelInfo(this.ocelInfo)
  }

  onFileSelected(event: any) {
    const file: File = event.target.files[0];
    if (file) {
      this.configValid = false
      this.domService.ocelConfigValid = false
      this.domService.netConfigValid = false
      this.ocelInfo = new OCELInfo()
      const formData = new FormData();
      formData.append("file", file);
      const upload$ = this.appService.postOCEL(formData).subscribe((resp: OCELInfoDTO) => {
        let sessionKey = resp.sessionKey
        this.domService.setSessionKey(sessionKey)
        let ocelInfo = new OCELInfo(resp)
        this.ocelInfo = ocelInfo
        this.filename = file.name
        this.onUpdateOCELInfo()
      });
    }
  }

  checkConfigurationValidity() {
    let configValid = this.ocelInfo.acts.length > 0 &&
      this.checkObjectTypeGraphValidity() && this.checkLeadingTypeAssignments()
    this.configValid = configValid
    let configChanged = this.configChanged()
    if (configChanged) {
      this.domService.setOcelConfigValid(false)
      this.domService.setObjectModelValid(false)
      this.domService.setOcpnConfigValid(false)
    }
  }

  configChanged() {
    if (this.ocelInfo && !this.chachedOcelInfo) {
      this.chachedOcelInfo = cloneDeep(this.ocelInfo)
      return false
    }
    let changed = !isEqual(this.chachedOcelInfo, this.ocelInfo)
    if (changed) {
      this.chachedOcelInfo = cloneDeep(this.ocelInfo)
      return true
    }
    return false
  }

  checkObjectTypeGraphValidity() {
    // Check if graph has unique shortest paths 
    let nodes: { id: string, label: string, color: string }[] = [];
    this.type_nodes.forEach(val => nodes.push(Object.assign({}, val)));
    for (var id in nodes) {
      let links: { source: string, target: string, label: string }[] = [];
      this.type_links.forEach(val => links.push(Object.assign({}, val)));
      let shortest_paths: any = {}
      let buffer: { id: string, path: string[] }[] = [{ id: id, path: [] }]
      while (buffer.length > 0) {
        let current = buffer[0]
        let current_id = current.id
        let current_path = current.path
        console.log(current_path)
        buffer = buffer.slice(1)
        if (Object.keys(shortest_paths).find(x => x == current_id)) {
          if (shortest_paths[current_id].length == current_path.length) {
            this.graphBoxStyle = 'border-color: red'
            return false;
          }
          else if (shortest_paths[current_id].length < current_path.length) {
            continue
          }
        }
        shortest_paths[current_id] = current_path
        let neighbors = links.filter((link) => link.source == current_id || link.target == current_id)
          .map((link) => {
            if (link.source == current_id) {
              return { id: link.target, path: current_path.concat([current_id]) }
            }
            return { id: link.source, path: current_path.concat([current_id]) }
          })
        //links = links.filter( (link) => link.source != current_id && link.target != current_id)
        buffer = neighbors.concat(buffer)
      }
    }
    this.graphBoxStyle = 'border-color: green'
    return true;
  }

  checkLeadingTypeAssignments() {
    const leadtype_selections = this.ocelInfo.activity_leading_type_selections
    const type_selections = this.ocelInfo.activity_selected_types
    const acts = this.ocelInfo.acts
    for (var index = 0; index < this.ocelInfo.acts.length; index++) {
      let act = this.ocelInfo.acts[index]
      if (type_selections[act].length > 0 &&
        !this.ocelInfo.otypes.includes(
          leadtype_selections[act]
        )
      ) {
        return false
      }
      let leadtype = leadtype_selections[act]
      // check if other activities of that leading type include same object types
      for (var j_index = 0; j_index < this.ocelInfo.acts.length; j_index++) {
        let j_act = this.ocelInfo.acts[j_index]
        if (Object.keys(leadtype_selections).includes(j_act)
          && leadtype_selections[j_act] == leadtype ) {
          let type_selections_j_act = type_selections[j_act].sort()
          let type_selections_act = type_selections[act].sort()
          if( type_selections_j_act.join(',') !== type_selections_act.join(',')) {
            return false
          }
        } 
      }
    }
    return true
  }

  onChangeSelectedOtypes() {
    Object.keys(this.ocelInfo.activity_selected_types).map((act) => {
      let selected_otypes = Object.assign([], this.ocelInfo.activity_selected_types[act])
      let lead_candidates = Object.assign([], this.ocelInfo.activity_leading_type_candidates[act])
      let filtered_leads = lead_candidates.filter(otype =>
        selected_otypes.includes(otype)
      )
      this.ocelInfo.activity_leading_type_candidates_filtered[act] = filtered_leads
      let selected_lead = this.ocelInfo.activity_leading_type_selections[act]
      if (selected_lead && !this.ocelInfo.activity_leading_type_candidates_filtered[act].includes(selected_lead)) {
        this.ocelInfo.activity_leading_type_selections[act] = ""
      }
      this.ocelInfo.activity_leading_type_candidates[act] = lead_candidates
    })
    this.updateObjectTypeGraph()
    this.checkConfigurationValidity()
  }

  onUpdateOCELInfo() {
    this.type_nodes = []
    this.otypes = this.ocelInfo.otypes
    this.acts = this.ocelInfo.acts
    this.otypes.map((otype: string, index: number) => {
      this.type_nodes = this.type_nodes.concat({
        id: "" + index,
        label: otype,
        color: this.domService.getColor(index)
      })
    })
    this.updateObjectTypeGraph()
    this.setLeadingGroupsInfo()
    this.checkConfigurationValidity()
  }


  updateObjectTypeGraph() {
    debugger
    this.type_links = []
    Object.keys(this.ocelInfo.activity_selected_types).map((act) => {
      let leading_type = this.ocelInfo.activity_leading_type_selections[act]
      if (!leading_type) {
        return
      }
      let leading_type_node = this.type_nodes.find(x => x.label == leading_type)
      this.ocelInfo.activity_selected_types[act].map((selected_type) => {
        if (selected_type == leading_type) {
          return
        }
        let type_node = this.type_nodes.find(x => x.label == selected_type)
        if (this.type_links.find(l =>
          (l.source == leading_type_node?.id && l.target == type_node?.id)
        )) {
          return
        }
        this.type_links = this.type_links.concat({
          source: leading_type_node?.id ?? "Germany",
          target: type_node?.id ?? "Germany",
          label: ""
        })
      })
    })
  }

  setLeadingGroupsInfo() {
    if(Object.keys(this.ocelInfo.activity_leading_type_groups).length == 0){
      return
    }
    let leadingGroupsInfo = "Possible Leading Groups: \n"
    const groups = this.ocelInfo.activity_leading_type_groups
    Object.keys(groups).forEach( otype => {
      if (groups[otype].length == 0) {
        return
      }
      leadingGroupsInfo += otype + ": "
      groups[otype].forEach(group => {
        leadingGroupsInfo += "("
        group.forEach(act => {
          leadingGroupsInfo += act + ","
        })
        leadingGroupsInfo = leadingGroupsInfo.slice(0, -1)
        leadingGroupsInfo += "),"
      })
      leadingGroupsInfo = leadingGroupsInfo.slice(0, -1)
      leadingGroupsInfo += ";"
    })
    leadingGroupsInfo += "--- An object type X can only lead activities within at most one leading group."
      + " These are activities where instances of X occur with one consistent object model."
    this.leadingGroupsInfo = leadingGroupsInfo
  }

  actConfigValid(act: string) {
    return this.ocelInfo.activity_selected_types[act].length > 0
  }

  onConfirm() {
    let session_key = this.domService.getSessionKey()
    if(!session_key) {return}
    const formData = this.ocelInfo.asFormData()
    this.appService.postOCELConfig(session_key, formData).subscribe( _ => {
      this.domService.setOcelConfigValid(true)
      this.router.navigate(['object_model'])
    })
  }

}
