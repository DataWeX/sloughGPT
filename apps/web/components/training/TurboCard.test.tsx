// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'
import { makeDatasets } from './__test-helper'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children }: any) => <div>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled }: any) => (
    <button onClick={onClick} disabled={disabled}>{children}</button>
  ),
  Input: (props: any) => <input {...props} />,
  Label: ({ children }: any) => <label>{children}</label>,
  Progress: ({ value }: any) => <div data-testid="progress">{value}</div>,
}))

vi.mock('@/components/training/DatasetSelector', () => ({
  DatasetSelector: (props: any) => <div data-testid="dataset-selector">{props.value || 'none'}</div>,
}))

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: { loadCheckpoint: vi.fn(async () => ({ status: 'loaded' })) },
}))

import { trainingJobsController } from '@/lib/training-controller'
import { TurboCard, TURBO_DEFAULTS } from './TurboCard'
import type { UseTrainingDatasetsReturn } from '@/hooks/useTrainingDatasets'
import type { UseTrainingSessionReturn } from '@/hooks/useTrainingSession'

function makeSession(overrides: Record<string, any> = {}): UseTrainingSessionReturn {
  return {
    turboPhase: 'idle',
    turboProgress: 0,
    turboGlobalStep: 0,
    turboTotalSteps: 0,
    turboEta: null,
    turboStepsPerSec: null,
    turboElapsedSeconds: null,
    turboLoss: null,
    turboResult: null,
    turboError: null,
    startTurboTrain: vi.fn(),
    stopTurboTrain: vi.fn(),
    ...overrides,
  } as unknown as UseTrainingSessionReturn
}

describe('TurboCard', () => {
  afterEach(cleanup)

  it('renders dataset selector and config inputs in idle phase', () => {
    const datasets = makeDatasets({ selectedDataset: '1' })
    render(<TurboCard datasets={datasets as unknown as UseTrainingDatasetsReturn} session={makeSession()} addToast={vi.fn()} />)
    expect(screen.getByTestId('dataset-selector')).toBeDefined()
    expect(screen.getByText('Epochs')).toBeDefined()
    expect(screen.getByText('Embed')).toBeDefined()
    expect(screen.getByText('Start turbo train')).toBeDefined()
  })

  it('disables start when no dataset is selected', () => {
    const datasets = makeDatasets({ selectedDataset: '' })
    const addToast = vi.fn()
    render(<TurboCard datasets={datasets as unknown as UseTrainingDatasetsReturn} session={makeSession()} addToast={addToast} />)
    const start = screen.getByText('Start turbo train') as HTMLButtonElement
    expect(start.disabled).toBe(true)
    expect(addToast).not.toHaveBeenCalled()
  })

  it('starts turbo training with selected dataset and defaults', () => {
    const datasets = makeDatasets({ selectedDataset: '1' })
    const session = makeSession()
    render(<TurboCard datasets={datasets as unknown as UseTrainingDatasetsReturn} session={session} addToast={vi.fn()} />)
    fireEvent.click(screen.getByText('Start turbo train'))
    expect(session.startTurboTrain).toHaveBeenCalledWith('1', TURBO_DEFAULTS, expect.any(Function), undefined)
  })

  it('renders live progress and stop during training', () => {
    const session = makeSession({
      turboPhase: 'training',
      turboProgress: 42,
      turboGlobalStep: 42,
      turboTotalSteps: 100,
      turboStepsPerSec: 2.5,
      turboEta: 30,
      turboElapsedSeconds: 10,
      turboLoss: 1.5,
    })
    render(<TurboCard datasets={makeDatasets() as unknown as UseTrainingDatasetsReturn} session={session} addToast={vi.fn()} />)
    expect(screen.getByTestId('progress')).toBeDefined()
    expect(screen.getByText(/Step 42\/100/)).toBeDefined()
    expect(screen.getByText(/2.5 steps\/s/)).toBeDefined()
    expect(screen.getByText(/ETA 30s/)).toBeDefined()
    expect(screen.getByText(/Elapsed 10s/)).toBeDefined()
    expect(screen.getByText(/Loss 1.5000/)).toBeDefined()
    fireEvent.click(screen.getByText('Stop'))
    expect(session.stopTurboTrain).toHaveBeenCalled()
  })

  it('shows completion result with train-another action', () => {
    const session = makeSession({
      turboPhase: 'complete',
      turboResult: { status: 'complete', final_loss: 0.9, total_steps: 100, model_path: '/models/x.soul' },
    })
    render(<TurboCard datasets={makeDatasets() as unknown as UseTrainingDatasetsReturn} session={session} addToast={vi.fn()} />)
    expect(screen.getByText('Turbo training complete!')).toBeDefined()
    expect(screen.getByText(/Final loss: 0.9000/)).toBeDefined()
    fireEvent.click(screen.getByText('Train another'))
    expect(session.stopTurboTrain).toHaveBeenCalled()
  })

  it('loads the trained model for chat from the result model path', async () => {
    const addToast = vi.fn()
    const session = makeSession({
      turboPhase: 'complete',
      turboResult: { status: 'complete', final_loss: 0.9, total_steps: 100, model_path: 'models/turbo-trained/turbo_42.soul' },
    })
    render(<TurboCard datasets={makeDatasets() as unknown as UseTrainingDatasetsReturn} session={session} addToast={addToast} />)
    fireEvent.click(screen.getByText('Load for chat'))
    await vi.waitFor(() => {
      expect(trainingJobsController.loadCheckpoint).toHaveBeenCalledWith('turbo_42.soul')
    })
    await vi.waitFor(() => {
      expect(addToast).toHaveBeenCalledWith('Loaded trained version: turbo_42.soul', 'success')
    })
  })

  it('reports a failure toast when loading the model fails', async () => {
    vi.mocked(trainingJobsController.loadCheckpoint).mockRejectedValueOnce(new Error('boom'))
    const addToast = vi.fn()
    const session = makeSession({
      turboPhase: 'complete',
      turboResult: { status: 'complete', final_loss: 0.5, total_steps: 50, model_path: 'models/turbo-trained/bad.soul' },
    })
    render(<TurboCard datasets={makeDatasets() as unknown as UseTrainingDatasetsReturn} session={session} addToast={addToast} />)
    fireEvent.click(screen.getByText('Load for chat'))
    await vi.waitFor(() => {
      expect(addToast).toHaveBeenCalledWith('Failed to load trained version', 'error')
    })
  })

  it('shows error state with dismiss action', () => {
    const session = makeSession({ turboPhase: 'error', turboError: 'oom' })
    render(<TurboCard datasets={makeDatasets() as unknown as UseTrainingDatasetsReturn} session={session} addToast={vi.fn()} />)
    expect(screen.getByText('Turbo training failed')).toBeDefined()
    expect(screen.getByText('oom')).toBeDefined()
    fireEvent.click(screen.getByText('Dismiss'))
    expect(session.stopTurboTrain).toHaveBeenCalled()
  })
})
