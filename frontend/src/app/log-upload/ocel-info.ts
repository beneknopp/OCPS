export interface OCELInfoDTO {
    otypes: string[];
    acts: string[]
    activity_allowed_types: {[act: string] : string[]}
    activity_leading_type_candidates: {[act: string] : string[]}
    activity_leading_type_groups: {[otype: string] : string[][]}
    non_emitting_types: string[]
    sessionKey: string
}

export class OCELInfo {
    
    otypes: string[]
    acts: string[]
    activity_allowed_types: {[act: string] : string[]}
    activity_leading_type_candidates: {[act: string] : string[]}
    activity_leading_type_groups: {[otype: string] : string[][]}
    
    activity_selected_types : { [act: string]: string[] } = {}
    activity_leading_type_candidates_filtered : { [act: string]: string[] } = {}
    activity_leading_type_selections : { [act: string]: string } = {}
    non_emitting_types: string[] = []

    constructor(
        ocelInfoDto?: OCELInfoDTO
    ){
        this.otypes = ocelInfoDto?.otypes ?? []
        this.acts = ocelInfoDto?.acts ?? []
        this.activity_allowed_types = ocelInfoDto?.activity_allowed_types ?? {}
        this.activity_leading_type_groups = ocelInfoDto?.activity_leading_type_groups ?? {}
        this.activity_leading_type_candidates = ocelInfoDto?.activity_leading_type_candidates ?? {}
        this.initializeActivityObjectInfo()
    }

    private initializeActivityObjectInfo() {
        this.activity_allowed_types = Object.assign([], this.activity_allowed_types)
        this.activity_leading_type_groups = Object.assign([], this.activity_leading_type_groups)
        this.activity_leading_type_candidates = Object.assign([], this.activity_leading_type_candidates)
        this.activity_leading_type_candidates_filtered = Object.assign([], this.activity_leading_type_candidates)
        this.activity_leading_type_selections = {}
        Object.keys(this.activity_leading_type_candidates).map((act) => {
          this.activity_leading_type_selections[act] = this.activity_leading_type_candidates[act][0]
        })
        this.activity_selected_types = {}
        Object.keys(this.activity_allowed_types).map((act) => {
          this.activity_selected_types[act] = Object.assign([], this.activity_allowed_types[act])
        })
    }

    asFormData() {
        const formData = new FormData()
        const blob = new Blob([JSON.stringify(this, null, 2)], {
            type: "application/json",
          });
        formData.append("ocelInfo", blob)
        return formData
      }

}