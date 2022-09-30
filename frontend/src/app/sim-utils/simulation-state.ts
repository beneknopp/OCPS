import { Binding } from "./binding";
import { Token } from "./token";

export class SimulationStateDto {

    objectsInitialized: {[otype: string]: number}
    objectsTerminated: {[otype: string]: number}
    totalObjects: {[otype: string]: number}
    activeTokens: Token[]
    markingInfo: {[place_id: string]: number}
    bindings: [oid: number, transition: string][]
    clock: number
    steps: number

    constructor(dto: {
        "objectsInitialized": {[otype: string]: number},
        "objectsTerminated": {[otype: string]: number},
        "totalObjects": {[otype: string]: number},
        "activeTokens": Token[],
        "markingInfo": {[place_id: string]: number},
        "bindings": [oid: number, transition: string][],
        "clock": number,
        "steps": number
    }){
        this.objectsInitialized = dto.objectsInitialized
        this.objectsTerminated = dto.objectsTerminated
        this.totalObjects = dto.totalObjects
        this.activeTokens = dto.activeTokens
        this.markingInfo = dto.markingInfo
        this.bindings = dto.bindings
        this.clock = dto.clock
        this.steps = dto.steps
    }
}

export class SimulationState {
    
    tokens: Token[]
    bindings: Binding[]

    constructor(
        tokens: Token[],
        bindings: Binding[]
    ){
        this.tokens = tokens
        this.bindings = bindings
    }
}