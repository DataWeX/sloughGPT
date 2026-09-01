/**
 * App Shell — persistent state layer that survives page refreshes.
 *
 * The UI is a thin view over this store. All critical state (training,
 * model, sessions) lives here in localStorage-backed vanilla Zustand.
 * On mount, the shell reconciles with the server.
 */
'use client'

import { createStore } from 'zustand/vanilla'
import { persist } from 'zustand/middleware'
import { trackEvent } from '@/lib/dev-log'

export interface TrainingShellState {
  phase: 'idle' | 'TRAINING' | 'complete' | 'error'
  method: 'slonet' | 'hf' | 'turbo' | null
  loss: number | null
  progress: number
  epoch: number
  totalEpochs: number
  globalStep: number
  totalSteps: number
  eta: number | null
  stepsPerSec: number | null
  elapsedSeconds: number | null
  message: string
  startTime: number | null
  lossHistory: { step: number; loss: number; isEval?: boolean }[]
  evalResult: string | null
  checkpoint: string | null
  finalLoss: number | null
  modelPath: string | null
  error: string | null
  jobId: string | null
  visualOutputDir: string | null
  visualSouPath: string | null
  avgQuality: number | null
  dataQuality: { avg_quality: number; repetition_rate: number; diversity: number; language_quality: number } | null
  finetunedModelPath: string | null
  finetunedModelLoss: number | null
  distillCheckpoint: string | null
  distillFinalLoss: number | null
  distillEpochs: number | null
}

export interface AppShellState {
  training: TrainingShellState
  lastActiveRoute: string
  lastActivity: number

  // Training actions
  setTraining: (partial: Partial<TrainingShellState>) => void
  resetTraining: () => void
  setLastActiveRoute: (route: string) => void
}

const DEFAULT_TRAINING: TrainingShellState = {
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
  avgQuality: null,
  dataQuality: null,
  visualSouPath: null,
  finetunedModelPath: null,
  finetunedModelLoss: null,
  distillCheckpoint: null,
  distillFinalLoss: null,
  distillEpochs: null,
}

export const appShellStore = createStore<AppShellState>()(
  persist(
    (set) => ({
      training: { ...DEFAULT_TRAINING },
      lastActiveRoute: '/chat',
      lastActivity: Date.now(),

      setTraining: (partial) =>
        set((state) => {
          if (partial.phase !== undefined && partial.phase !== state.training.phase) {
            trackEvent('training_phase_changed', { from: state.training.phase, to: partial.phase, method: partial.method ?? state.training.method })
          }
          return {
            training: { ...state.training, ...partial },
            lastActivity: Date.now(),
          }
        }),

      resetTraining: () =>
        set((state) => {
          trackEvent('training_reset', { phase: state.training.phase })
          return {
            training: { ...DEFAULT_TRAINING },
            lastActivity: Date.now(),
          }
        }),

      setLastActiveRoute: (route) =>
        set((state) => {
          trackEvent('route_changed', { from: state.lastActiveRoute, to: route })
          return { lastActiveRoute: route, lastActivity: Date.now() }
        }),
    }),
    {
      name: 'app-shell',
      partialize: (state) => ({
        training: state.training,
        lastActiveRoute: state.lastActiveRoute,
        lastActivity: state.lastActivity,
      }),
    }
  )
)

export type TrainingToastFn = (msg: string, type?: 'success' | 'error' | 'info') => void

/**
 * Read current training state from the shell store.
 * Use this outside React (in callbacks, intervals, etc.) to avoid stale closures.
 */
export function readTraining(): TrainingShellState {
  return appShellStore.getState().training
}

/**
 * Write a partial update to the shell training state.
 * Use this outside React (in callbacks, intervals, etc.) to avoid stale closures.
 */
export function writeTraining(partial: Partial<TrainingShellState>) {
  appShellStore.getState().setTraining(partial)
}

/**
 * Check if training is active (not idle/complete/error).
 * Used by the shell to decide whether to reconcile with server.
 */
export function isTrainingActive(training: TrainingShellState): boolean {
  return training.phase !== 'idle' && training.phase !== 'complete' && training.phase !== 'error'
}

/**
 * Get a snapshot of shell training state for the UI.
 */
export function getTrainingSnapshot(): TrainingShellState {
  return appShellStore.getState().training
}
