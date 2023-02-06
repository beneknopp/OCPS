import { ArcDto } from "./arc";
import { PlaceDto } from "./place";
import { TransitionDto } from "./transition";

export class OcpnDto {
    
    places : PlaceDto[]
    transitions: TransitionDto[]
    arcs: ArcDto[]
    precision: number | undefined
    fitness: number | undefined

    constructor(
        places? : PlaceDto[],
        transitions?: TransitionDto[],
        arcs?: ArcDto[],
        precision? : number | undefined,
        fitness?: number | undefined
    ){
        this.places = places ?? []
        this.transitions = transitions ?? []
        this.arcs = arcs ?? []
        this.precision = precision ?? undefined
        this.fitness = fitness ?? undefined
    }

}
