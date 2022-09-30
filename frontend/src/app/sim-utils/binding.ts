import { Transition } from "../net-utils/transition"
import { Token } from "./token"

export class BindingDto {

    transition_id: string
    token_id: string
    object_id: string

    constructor(
        transition_id: string,
        token_id: string,
        object_id: string        
    ) {
        this.transition_id = transition_id
        this.token_id = token_id
        this.object_id = object_id
    }

}

export class Binding {

    transition: Transition
    token: Token
    object_id: string

    constructor(
        transition: Transition,
        token: Token,
        object_id: string        
    ) {
        this.transition = transition
        this.token = token
        this.object_id = object_id
    }

}