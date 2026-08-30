// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { makeForm, makeDatasets, makeCheckpoints, ds } from './__test-helper'

vi.mock('@/components/training/LossChart', () => ({
  LossChart: () => <div data-testid="loss-chart" />,
}))

vi.mock('@/components/training/TrainingStatus', () => ({
  TrainingErrorBanner: ({ error }: any) => <div data-testid="error-banner">{error}</div>,
}))

import { TrainingPipeline } from './TrainingPipeline'
import type { TrainingFormState } from '@/hooks/useTrainingForm'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'
import type { UseTrainingSessionReturn } from '@/hooks/useTrainingSession'
import type { UseTrainingCheckpointsReturn } from '@/hooks/useTrainingCheckpoints'

const form: TrainingFormState = makeForm({
  trainingEpochs: 5,
  trainingLR: 1e-3,
  trainingBatchSize: 64,
  useLoRA: true,
  visualStage1Epochs: 5,
  visualStage2Epochs: 5,
  nativeLayers: 4,
})

const datasets: UseTrainingDatasetsReturn = makeDatasets({
  datasets: [ds({ name: 'shakespeare' })],
  selectedDataset: '1',
})

const session: UseTrainingSessionReturn = {
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
  finetunedModelPath: null,
  finetunedModelLoss: null,
  distillCheckpoint: null,
  distillFinalLoss: null,
  distillEpochs: null,
  avgQuality: null,
  dataQuality: null,
  turboPhase: 'idle',
  turboResult: null,
  turboError: null,
  turboProgress: 0,
  turboGlobalStep: 0,
  turboTotalSteps: 0,
  turboEta: null,
  turboStepsPerSec: null,
  turboElapsedSeconds: null,
  turboLoss: null,
  visualOutputDir: null,
  visualSouPath: null,
  setPhase: vi.fn(),
  setLoss: vi.fn(),
  setProgress: vi.fn(),
  setEpoch: vi.fn(),
  setTotalEpochs: vi.fn(),
  setGlobalStep: vi.fn(),
  setTotalSteps: vi.fn(),
  setEta: vi.fn(),
  setStepsPerSec: vi.fn(),
  setElapsedSeconds: vi.fn(),
  setMessage: vi.fn(),
  setLossHistory: vi.fn(),
  setEvalResult: vi.fn(),

  trainingRunning: false,
  resetTraining: vi.fn(),
  stopTraining: vi.fn(),
  pauseTraining: vi.fn(),
  resumeTraining: vi.fn(),
  paused: false,
  startFineTune: vi.fn(),
  startVisualTraining: vi.fn(),
  startTurboTrain: vi.fn(),
  stopTurboTrain: vi.fn(),
  turboRunning: false,
}

const checkpoints: UseTrainingCheckpointsReturn = makeCheckpoints()

describe('TrainingPipeline', () => {
  afterEach(cleanup)

  it('renders step indicator with 4 steps', () => {
    render(<TrainingPipeline form={form} datasets={datasets} session={session} checkpoints={checkpoints} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.getByText('Data')).toBeDefined()
    expect(screen.getByText('Configure')).toBeDefined()
    expect(screen.getByText('Train')).toBeDefined()
    expect(screen.getByText('Results')).toBeDefined()
  })

  it('starts on data step', () => {
    render(<TrainingPipeline form={form} datasets={datasets} session={session} checkpoints={checkpoints} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.getByText(/Pick your data/)).toBeDefined()
  })

  it('shows training in progress when session is running', () => {
    const runningSession = { ...session, trainingRunning: true }
    render(<TrainingPipeline form={form} datasets={datasets} session={runningSession} checkpoints={checkpoints} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.getByText('Training in progress')).toBeDefined()
  })

  it('shows error banner on training error', () => {
    const errorSession = { ...session, trainingRunning: true, phase: 'error' as const, message: 'OOM error' }
    render(<TrainingPipeline form={form} datasets={datasets} session={errorSession} checkpoints={checkpoints} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.getByTestId('error-banner')).toBeDefined()
    expect(screen.getByText('OOM error')).toBeDefined()
  })

  it('shows loss chart during training with data', () => {
    const trainSession = {
      ...session,
      trainingRunning: true,
      lossHistory: [{ step: 1, loss: 0.5 }, { step: 2, loss: 0.3 }],
    }
    render(<TrainingPipeline form={form} datasets={datasets} session={trainSession} checkpoints={checkpoints} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.getByText('Training in progress')).toBeDefined()
  })

  it('shows epoch info during training', () => {
    const trainSession = { ...session, trainingRunning: true, epoch: 3, totalEpochs: 10 }
    render(<TrainingPipeline form={form} datasets={datasets} session={trainSession} checkpoints={checkpoints} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.getByText('Epoch 3/10')).toBeDefined()
  })

  it('shows step, speed, ETA and elapsed stats during training', () => {
    const trainSession = {
      ...session,
      trainingRunning: true,
      phase: 'TRAINING' as const,
      globalStep: 80,
      totalSteps: 500,
      stepsPerSec: 4.25,
      eta: 98,
      elapsedSeconds: 20,
    }
    render(<TrainingPipeline form={form} datasets={datasets} session={trainSession} checkpoints={checkpoints} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.getByText('Step 80/500')).toBeDefined()
    expect(screen.getByText('4.3 steps/s')).toBeDefined()
    expect(screen.getByText('ETA 1m 38s')).toBeDefined()
    expect(screen.getByText('Elapsed 20s')).toBeDefined()
  })

  it('does not show stats when training is complete', () => {
    const completeSession = {
      ...session,
      trainingRunning: true,
      phase: 'complete' as const,
      globalStep: 500,
      totalSteps: 500,
      eta: 0,
    }
    render(<TrainingPipeline form={form} datasets={datasets} session={completeSession} checkpoints={checkpoints} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.queryByText(/Step /)).toBeNull()
  })

  it('shows complete message and Test button', () => {
    const completeSession = { ...session, trainingRunning: true, phase: 'complete' as const }
    render(<TrainingPipeline form={form} datasets={datasets} session={completeSession} checkpoints={checkpoints} onTest={vi.fn()} addToast={vi.fn()} />)
    expect(screen.getByText('Training complete')).toBeDefined()
    expect(screen.getByText('Test model')).toBeDefined()
  })
})
