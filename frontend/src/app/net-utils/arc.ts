
export class ArcDto {
    id: string
    source: string
    target: string
    isVariableArc: boolean
    constructor(id: string, source: string, target: string, is_variable_arc: boolean){
        this.id = id, this.source = source, this.target = target, this.isVariableArc = is_variable_arc
    }
}

export class Arc {

    id: string
    source: string
    target: string
    isVariableArc: boolean

    constructor(arc_dto: ArcDto) {
        this.id = arc_dto.id
        this.source = arc_dto.source
        this.target = arc_dto.target
        this.isVariableArc = arc_dto.isVariableArc
    }

}