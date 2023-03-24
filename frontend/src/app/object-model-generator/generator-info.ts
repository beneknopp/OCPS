import { ObjectModelStats } from "./object-model-info"

export class AttributeParametrizationResponse{
    xAxis: string[]
    yAxes: ObjectModelStats
    includeModeled: boolean
    includeSimulated: boolean
    parameters: string
    constructor(
        xAxis: string[],
        yAxes: ObjectModelStats,
        includeModeled: boolean,
        includeSimulated: boolean,
        parameters: string
    ) {
      this.xAxis = xAxis
      this.yAxes = yAxes
      this.includeModeled = includeModeled
      this.includeSimulated = includeSimulated
      this.parameters = parameters
    }
}
