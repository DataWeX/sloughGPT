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

export interface TrainingShellState {
  phase: 'idle' | 'TRAINING' | 'complete' | 'error'
  method: 'slnet' | 'hf' | 'turbo' | null
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
  visualSouPath: null,
}

export const appShellStore = createStore<AppShellState>()(
  persist(
    (set) => ({
      training: { ...DEFAULT_TRAINING },
      lastActiveRoute: '/chat',
      lastActivity: Date.now(),

      setTraining: (partial) =>
        set((state) => ({
          training: { ...state.training, ...partial },
          lastActivity: Date.now(),
        })),

      resetTraining: () =>
        set((state) => ({
          training: { ...DEFAULT_TRAINING },
          lastActivity: Date.now(),
        })),

      setLastActiveRoute: (route) =>
        set({ lastActiveRoute: route, lastActivity: Date.now() }),
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
