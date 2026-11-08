// @vitest-environment jsdom
/**
 * Plain-language guard — Build Order #5 (card 20261101_140445_build-order-5),
 * UX_FLOWS Feature #12 "Train My AI" + "Training: 3-Click Flow".
 *
 * The spec's "What they should NOT see" lists are invariants: no "Epoch"
 * (plain language: "Training round"), no "LoRA" (hidden behind Advanced per
 * the translation table). Loss numbers stay on purpose — the 3-Click Flow
 * explicitly allows checkpoint cards "with loss and quality scores"
 * (ruling: user 2026-11-08).
 *
 * Assertions are SCOPED (specific strings / specific surface), never a
 * whole-page banned-word scan: FoldSection renders collapsed children
 * in-DOM, so a naive textContent scan false-positives on legitimately
 * hidden advanced settings.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'
import { makeForm, makeDatasets, makeCheckpoints, ds } from './__test-helper'

vi.mock('@/components/training/LossChart', () => ({
  LossChart: () => <div data-testid="loss-chart" />,
}))

vi.mock('@/components/training/TrainingStatus', () => ({
  TrainingErrorBanner: ({ error }: any) => <div data-testid="error-banner">{error}</div>,
}))

const { mockLoadAdapter } = vi.hoisted(() => ({ mockLoadAdapter: vi.fn() }))
vi.mock('@/lib/controllers', async (importOriginal) => {
  const mod = await importOriginal<Record<string, unknown>>()
  return {
    ...mod,
    trainingJobsController: {
      ...(mod.trainingJobsController as Record<string, unknown>),
      loadAdapter: mockLoadAdapter,
    },
  }
})

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
  startSSETraining: vi.fn(),
  closeStream: vi.fn(),
  startStandardPoll: vi.fn(),
  turboRunning: false,
}

const checkpoints: UseTrainingCheckpointsReturn = makeCheckpoints()


describe('training plain-language guard (Build Order #5, UX_FLOWS Feature 12)', () => {
  afterEach(cleanup)

  const renderPipeline = (session: UseTrainingSessionReturn, addToast = vi.fn()) =>
    render(
      <TrainingPipeline
        form={form}
        datasets={datasets}
        session={session}
        checkpoints={checkpoints}
        onTest={vi.fn()}
        addToast={addToast}
        step="data"
        onStepChange={vi.fn()}
        completedSteps={new Set()}
        onStepComplete={vi.fn()}
      />,
    )

  it('running state says "Training round", never "Epoch" (header + stat tile)', () => {
    const running: UseTrainingSessionReturn = {
      ...session,
      trainingRunning: true,
      epoch: 3,
      totalEpochs: 10,
    }
    const { container } = renderPipeline(running)
    expect(screen.getByText('Training round 3/10')).toBeDefined()
    expect(screen.getByText('Training round')).toBeDefined() // stat-tile label
    expect(container.textContent).not.toMatch(/epoch/i)
  })

  it('adapter-load surface is LoRA-free and toasts plainly', async () => {
    mockLoadAdapter.mockClear()
    const done: UseTrainingSessionReturn = {
      ...session,
      trainingRunning: true,
      phase: 'complete' as const,
      method: 'hf' as UseTrainingSessionReturn['method'],
      checkpoint: 'adapter-final.npz',
    }
    const addToast = vi.fn()
    const { container } = renderPipeline(done, addToast)
    expect(container.textContent).not.toMatch(/lora/i)
    const btn = screen.getByRole('button', { name: /load adapter into model/i })
    fireEvent.click(btn)
    await vi.waitFor(() => expect(mockLoadAdapter).toHaveBeenCalledWith('adapter-final.npz', false))
    await vi.waitFor(() => expect(addToast).toHaveBeenCalled())
    for (const [msg] of addToast.mock.calls) {
      expect(String(msg)).not.toMatch(/lora/i)
    }
  })
})
