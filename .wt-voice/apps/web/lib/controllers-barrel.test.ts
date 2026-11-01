import { describe, it, expect } from 'vitest'

import * as barrel from './controllers'
import { modelController } from './model-controller'
import { trainingJobsController, trainingController } from './training-controller'
import { sessionController } from './session-controller'
import { datasetController } from './dataset-controller'
import { agentsController } from './agents-controller'
import { soulsController } from './souls-controller'
import { multimodalController } from './multimodal-controller'
import { errorsController } from './errors-controller'
import { experimentsController } from './experiments-controller'
import { workflowController } from './workflow-controller'
import { filesController } from './files-controller'
import { voiceController } from './voice-controller'
import { registryController } from './registry-controller'

describe('controllers barrel', () => {
  it('re-exports every controller with identity (same object as source module)', () => {
    expect(barrel.modelController).toBe(modelController)
    expect(barrel.trainingJobsController).toBe(trainingJobsController)
    expect(barrel.trainingController).toBe(trainingController)
    expect(barrel.sessionController).toBe(sessionController)
    expect(barrel.datasetController).toBe(datasetController)
    expect(barrel.agentsController).toBe(agentsController)
    expect(barrel.soulsController).toBe(soulsController)
    expect(barrel.multimodalController).toBe(multimodalController)
    expect(barrel.errorsController).toBe(errorsController)
    expect(barrel.experimentsController).toBe(experimentsController)
    expect(barrel.workflowController).toBe(workflowController)
    expect(barrel.filesController).toBe(filesController)
    expect(barrel.voiceController).toBe(voiceController)
    expect(barrel.registryController).toBe(registryController)
  })

  it('exposes all named controllers as objects with methods', () => {
    const names = [
      'modelController', 'trainingJobsController', 'trainingController',
      'sessionController', 'datasetController', 'agentsController',
      'soulsController', 'multimodalController', 'errorsController',
      'experimentsController', 'workflowController', 'filesController',
      'voiceController', 'registryController',
    ]
    for (const name of names) {
      expect(typeof (barrel as Record<string, unknown>)[name]).toBe('object')
    }
  })

  it('has no duplicate controller references', () => {
    const names = Object.keys(barrel).filter(k => typeof (barrel as any)[k] === 'object' && (barrel as any)[k] !== null)
    const unique = new Set(names)
    expect(unique.size).toBe(names.length)
  })
})
