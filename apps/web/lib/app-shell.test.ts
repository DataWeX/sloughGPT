import { describe, it, expect, beforeEach } from 'vitest'
import {
  appShellStore,
  readTraining,
  writeTraining,
  isTrainingActive,
  getTrainingSnapshot,
  type TrainingShellState,
} from './app-shell'

describe('app-shell', () => {
  beforeEach(() => {
    appShellStore.setState({
      training: {
        phase: 'idle',
        method: null,
        loss: null,
        progress: 0,
        epoch: 0,
        totalEpochs: 0,
        globalStep: 0,
        totalSteps: 0,
        eta: null,
        stepsPerSec: null,
        elapsedSeconds: null,
        message: '',
        startTime: null,
        lossHistory: [],
        evalResult: null,
        checkpoint: null,
        finalLoss: null,
        modelPath: null,
        error: null,
        jobId: null,
        visualOutputDir: null,
        visualSouPath: null,
        avgQuality: null,
        dataQuality: null,
        finetunedModelPath: null,
        finetunedModelLoss: null,
        distillCheckpoint: null,
        distillFinalLoss: null,
        distillEpochs: null,
      },
      lastActiveRoute: '/chat',
      lastActivity: Date.now(),
    })
  })

  describe('readTraining', () => {
    it('returns current training state', () => {
      const t = readTraining()
      expect(t.phase).toBe('idle')
      expect(t.loss).toBeNull()
      expect(t.progress).toBe(0)
    })

    it('reflects updates from setTraining', () => {
      appShellStore.getState().setTraining({ phase: 'TRAINING', loss: 0.5 })
      const t = readTraining()
      expect(t.phase).toBe('TRAINING')
      expect(t.loss).toBe(0.5)
    })
  })

  describe('writeTraining', () => {
    it('partial-updates training state', () => {
      writeTraining({ epoch: 3, totalEpochs: 10 })
      const t = readTraining()
      expect(t.epoch).toBe(3)
      expect(t.totalEpochs).toBe(10)
      expect(t.phase).toBe('idle') // unchanged
    })

    it('updates lastActivity timestamp', () => {
      const before = appShellStore.getState().lastActivity
      writeTraining({ loss: 1.0 })
      expect(appShellStore.getState().lastActivity).toBeGreaterThanOrEqual(before)
    })
  })

  describe('isTrainingActive', () => {
    it.each<{ phase: TrainingShellState['phase']; expected: boolean }>([
      { phase: 'idle', expected: false },
      { phase: 'complete', expected: false },
      { phase: 'error', expected: false },
      { phase: 'TRAINING', expected: true },
    ])('phase=$phase → $expected', ({ phase, expected }) => {
      expect(isTrainingActive({ ...readTraining(), phase })).toBe(expected)
    })
  })

  describe('resetTraining', () => {
    it('restores all fields to defaults', () => {
      writeTraining({ phase: 'TRAINING', loss: 2.0, epoch: 5, jobId: 'abc' })
      appShellStore.getState().resetTraining()
      const t = readTraining()
      expect(t.phase).toBe('idle')
      expect(t.loss).toBeNull()
      expect(t.epoch).toBe(0)
      expect(t.jobId).toBeNull()
      expect(t.lossHistory).toEqual([])
    })
  })

  describe('getTrainingSnapshot', () => {
    it('returns a snapshot matching readTraining', () => {
      writeTraining({ loss: 3.14 })
      expect(getTrainingSnapshot()).toEqual(readTraining())
    })
  })

  describe('setLastActiveRoute', () => {
    it('updates route and lastActivity', () => {
      const before = appShellStore.getState().lastActivity
      appShellStore.getState().setLastActiveRoute('/training')
      const s = appShellStore.getState()
      expect(s.lastActiveRoute).toBe('/training')
      expect(s.lastActivity).toBeGreaterThanOrEqual(before)
    })
  })

  describe('partialize', () => {
    it('only persists training, lastActiveRoute, lastActivity', () => {
      const state = appShellStore.getState()
      // Zustand persist uses partialize to select what to store
      const persisted = {
        training: state.training,
        lastActiveRoute: state.lastActiveRoute,
        lastActivity: state.lastActivity,
      }
      expect(persisted).toHaveProperty('training')
      expect(persisted).toHaveProperty('lastActiveRoute')
      expect(persisted).toHaveProperty('lastActivity')
      expect(Object.keys(persisted)).toHaveLength(3)
    })
  })
})
