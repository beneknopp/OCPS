import { ArcDto } from "./arc";
import { PlaceDto } from "./place";
import { TransitionDto } from "./transition";

export class OcpnDto {
    
    places : PlaceDto[]
    transitions: TransitionDto[]
    arcs: ArcDto[]

    constructor(
        places? : PlaceDto[],
        transitions?: TransitionDto[],
        arcs?: ArcDto[]
    ){
        this.places = places ?? []
        this.transitions = transitions ?? []
        this.arcs = arcs ?? []
    }

}