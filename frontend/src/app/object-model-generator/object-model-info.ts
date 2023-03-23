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
  executionModelDepth: number = 1
  executionModelEvaluationDepth : number = 1
  activitySelectedTypes: { [act: string]: string[] }
  activityLeadingTypes: { [act: string]: string | undefined }

  constructor(
    otypes: string[] = [],
    selectedSeedType: string | undefined = undefined,
    nonEmittingTypes: string[] = [],
    numberOfObjects: number = 0,
    executionModelDepth: number = 1,
    executionModelEvaluationDepth: number = 1,
    activitySelectedTypes: { [act: string]: string[] } = {},
    activityLeadingTypes: { [act: string]: string | undefined } = {}
  ) {
    this.otypes = otypes
    this.selectedSeedType = selectedSeedType
    this.nonEmittingTypes = nonEmittingTypes
    this.numberOfObjects = numberOfObjects
    this.executionModelDepth = executionModelDepth
    this.executionModelEvaluationDepth = executionModelEvaluationDepth
    this.activitySelectedTypes = activitySelectedTypes
    this.activityLeadingTypes = activityLeadingTypes
  }

}

export class ObjectModelGenerationResponse {
  stats: {
    [otype: string]: {
      "original_stats": {
        "mean": number,
        "stdev": number
      },
      "simulation_stats": {
        "mean": number,
        "stdev": number,
        "number_of_objects": number,
        "relations": {
          [otype: string]: {
            "mean": number,
            "stdev": number
          }
        }
      },
    }
  }
  constructor(stats: {
    [otype: string]: {
      "original_stats": {
        "mean": number,
        "stdev": number
      },
      "simulation_stats": {
        "mean": number,
        "stdev": number,
        "number_of_objects": number,
        "relations": {
          [otype: string]: {
            "mean": number,
            "stdev": number
          }
        }
      }
    }
  }) {
    this.stats = stats
  }
}