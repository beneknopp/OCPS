import { Arc } from "../net-utils/arc";
import { Place } from "../net-utils/place";
import { Transition } from "../net-utils/transition";

export class OcpnInfo {
    
    places: Place[]
    transitions: Transition[]
    arcs: Arc[]
    precision: number | undefined
    fitness: number | undefined

    constructor(
        places?: Place[],
        transitions?: Transition[],
        arcs?: Arc[],
        precision?: number | undefined,
        fitness?: number | undefined
    ) {
        this.places = places ? places : []
        this.transitions = transitions ? transitions : []
        this.arcs = arcs ? arcs : []
        this.precision = precision
        this.fitness = fitness
    }


}