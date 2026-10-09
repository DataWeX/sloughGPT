import { describe, it, expect, vi } from 'vitest'

vi.mock('./training-controller', () => ({
  trainingJobsController: {
    list: vi.fn(async () => [{ id: 'j1' }]),
    getTurboStatus: vi.fn(async () => ({ status: 'idle' })),
    startTurboTrain: vi.fn(async () => ({ status: 'started' })),
    listCheckpoints: vi.fn(async () => [{ name: 'c1' }]),
  },
}))
vi.mock('./self-train-controller', () => ({
  selfTrainController: {
    getStatus: vi.fn(async () => ({ phase: 'idle' })),
    start: vi.fn(async () => ({ jobId: 's1' })),
  },
}))
vi.mock('./lora-eval-controller', () => ({
  loraEvalController: {
    runEval: vi.fn(async () => ({ status: 'ok' })),
    getHistory: vi.fn(async () => []),
    aggregate: vi.fn(async () => ({ status: 'ok' })),
  },
}))
vi.mock('./settings-controller', () => ({
  settingsController: {
    getAutoTrainSettingsStatus: vi.fn(async () => ({ enabled: true })),
    updateAutoTrainSettingsConfig: vi.fn(async () => ({ enabled: false })),
  },
}))

import { trainingJobsController } from './training-controller'
import { selfTrainController } from './self-train-controller'
import { loraEvalController } from './lora-eval-controller'
import { settingsController } from './settings-controller'
import { trainingFacade } from './training-facade'

describe('trainingFacade', () => {
  it('delegates jobs.list to trainingJobsController', async () => {
    await expect(trainingFacade.jobs.list()).resolves.toEqual([{ id: 'j1' }])
    expect(vi.mocked(trainingJobsController.list)).toHaveBeenCalled()
  })

  it('delegates turbo.status to trainingJobsController', async () => {
    await expect(trainingFacade.turbo.status()).resolves.toEqual({ status: 'idle' })
    expect(vi.mocked(trainingJobsController.getTurboStatus)).toHaveBeenCalled()
  })

  it('delegates turbo.start with params', async () => {
    const params = { dataset_id: 'tiny' }
    await trainingFacade.turbo.start(params as never)
    expect(vi.mocked(trainingJobsController.startTurboTrain)).toHaveBeenCalledWith(params)
  })

  it('delegates checkpoints.list', async () => {
    await expect(trainingFacade.checkpoints.list()).resolves.toEqual([{ name: 'c1' }])
    expect(vi.mocked(trainingJobsController.listCheckpoints)).toHaveBeenCalled()
  })

  it('delegates automation to settings + self-train controllers', async () => {
    await expect(trainingFacade.automation.autoTrainStatus()).resolves.toEqual({ enabled: true })
    await expect(trainingFacade.automation.startSelfTrain()).resolves.toEqual({ jobId: 's1' })
    expect(vi.mocked(settingsController.getAutoTrainSettingsStatus)).toHaveBeenCalled()
    expect(vi.mocked(selfTrainController.start)).toHaveBeenCalled()
  })

  it('delegates evaluation to lora-eval controller', async () => {
    await expect(trainingFacade.evaluation.run('a.npz')).resolves.toEqual({ status: 'ok' })
    await expect(trainingFacade.evaluation.history()).resolves.toEqual([])
    expect(vi.mocked(loraEvalController.runEval)).toHaveBeenCalledWith('a.npz')
  })
})
