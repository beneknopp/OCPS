export class ObjectModelStats {
  LOG_BASED: number[]
  MODELED: number[]
  SIMULATED: number[]
  constructor(
    LOG_BASED: number[],
    MODELED: number[],
    SIMULATED: number[],
  ) {
    this.LOG_BASED = LOG_BASED
    this.MODELED = MODELED
    this.SIMULATED = SIMULATED
  }
}

export class ObjectModelInfo {

  otypes: string[]
  selectedSeedType: string | undefined = undefined
  nonEmittingTypes: string[] = []
  numberOfObjects: number = 0
  name: string
  executionModelDepth: number = 2
  executionModelEvaluationDepth: number = 2
  activitySelectedTypes: { [act: string]: string[] }
  activityLeadingTypes: { [act: string]: string | undefined }

  constructor(
    otypes: string[] = [],
    selectedSeedType: string | undefined = undefined,
    nonEmittingTypes: string[] = [],
    numberOfObjects: number = 0,
    name: string = "",
    executionModelDepth: number = 2,
    executionModelEvaluationDepth: number = 2,
    activitySelectedTypes: { [act: string]: string[] } = {},
    activityLeadingTypes: { [act: string]: string | undefined } = {}
  ) {
    this.otypes = otypes
    this.selectedSeedType = selectedSeedType
    this.nonEmittingTypes = nonEmittingTypes
    this.numberOfObjects = numberOfObjects
    this.name = name
    this.executionModelDepth = executionModelDepth
    this.executionModelEvaluationDepth = executionModelEvaluationDepth
    this.activitySelectedTypes = activitySelectedTypes
    this.activityLeadingTypes = activityLeadingTypes
  }

}

export class ObjectModelGenerationResponse {
  stats: { [otype: string]: { "number_of_objects": number } }

  constructor(stats: { [otype: string]: { "number_of_objects": number } }) {
    this.stats = stats
  }
}