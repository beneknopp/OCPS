import { Place } from "../net-utils/place"

export class TokenDto {
    
    token_id: string
    object_id: string
    time: number
    otype: string
    place_id: string

    constructor(
        token_id: string,
        object_id: string,
        time: number,
        otype: string,
        place_id: string
    ) {
        this.object_id = object_id
        this.token_id = token_id
        this.time = time
        this.otype = otype
        this.place_id = place_id
    }

}

export class Token {

    oid: string
    time: number
    //otype: string
    place_id: string

    constructor(
        oid: string,
        time: number,
        //otype: string,
        place_id: string
    ) {
        //this.token_id = token_id
        this.oid = oid
        this.time = time
        //this.otype = otype
        this.place_id = place_id
    }

}