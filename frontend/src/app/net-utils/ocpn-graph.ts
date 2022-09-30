export class OcpnGraphNode {
    id: string
    label: string
    type: string
    transitionType?: string
    color: string
    x: number
    y: number
    constructor(
        id: string,
        label: string,
        type: string,
        color: string,
        x: number,
        y: number,
        transitionType?: string
    ) {
        this.id = id
        this.label = label
        this.type = type
        this.color = color
        this.x = x
        this.y = y
        this.transitionType = transitionType
    }
}

export class OcpnGraphLink {
    source: string
    target: string
    label: string
    isVariableArc: boolean
    otype: string
    color: string 
    constructor(
        source: string,
        target: string,
        label: string,
        is_variable_arc: boolean,
        otype: string,
        color: string 
        ) {
        this.source = source
        this.target = target
        this.label = label
        this.isVariableArc = is_variable_arc
        this.otype = otype
        this.color = color
    }
}