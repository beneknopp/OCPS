import { PetriNetNode } from "./petri-net-node";

export class PlaceDto {
    id: string
    otype: string
    isInitial: boolean
    isFinal: boolean
    constructor(id: string, otype: string, is_initial: boolean, is_final: boolean){
        this.id = id, this.otype = otype, this.isInitial = is_initial, this.isFinal = is_final
    }
}

export class Place extends PetriNetNode {

    otype: string
    isInitial: boolean
    isFinal: boolean

    constructor(place_dto: PlaceDto) {
        super(place_dto.id)
        this.otype = place_dto.otype
        this.isInitial = place_dto.isInitial
        this.isFinal = place_dto.isFinal
    }

}