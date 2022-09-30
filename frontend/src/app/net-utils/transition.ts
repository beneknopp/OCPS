import { PetriNetNode } from "./petri-net-node";

export class TransitionDto {
    id: string
    label: string
    transitionType: string
    constructor(id: string, label: string, transitionType: string){
        this.id = id, this.label = label,this.transitionType = transitionType
    }
}

export class Transition extends PetriNetNode {

    transitionType: string
    label: string

    constructor(transition_dto: TransitionDto) {
        super(transition_dto.id)
        this.transitionType = transition_dto.transitionType
        this.label = transition_dto.label
    }

}