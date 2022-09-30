export class ObjectModelStats {
  log_based: number[];
  simulated: number[];
  x_axis: string[];
  constructor(
    log_based: number[],
    simulated: number[],
    x_axis: string[]
  ) {
    this.log_based = log_based
    this.simulated = simulated
    this.x_axis = x_axis
  }
}

export class ObjectModelInfo {

  otypes: string[];
  selectedSeedType: string | undefined = undefined
  nonEmittingTypes: string[] = []
  numberOfObjects: number = 0;
  activitySelectedTypes: { [act: string]: string[] }
  activityLeadingTypes: { [act: string]: string | undefined }

  constructor(
    otypes: string[] = [],
    selectedSeedType: string | undefined = undefined,
    nonEmittingTypes: string[] = [],
    numberOfObjects: number = 0,
    activitySelectedTypes: { [act: string]: string[] } = {},
    activityLeadingTypes: { [act: string]: string | undefined } = {},
  ) {
    this.otypes = otypes
    this.selectedSeedType = selectedSeedType
    this.nonEmittingTypes = nonEmittingTypes
    this.numberOfObjects = numberOfObjects
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