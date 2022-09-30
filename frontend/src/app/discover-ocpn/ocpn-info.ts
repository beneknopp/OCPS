import { Arc } from "../net-utils/arc";
import { Place } from "../net-utils/place";
import { Transition } from "../net-utils/transition";

export class OcpnInfo {
    
    places: Place[]
    transitions: Transition[]
    arcs: Arc[]

    constructor(
        places?: Place[],
        transitions?: Transition[],
        arcs?: Arc[]
    ) {
        this.places = places ? places : []
        this.transitions = transitions ? transitions : []
        this.arcs = arcs ? arcs : []
    }


}