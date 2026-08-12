import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    Button: ({ children, onClick, disabled, className }: any) => <button onClick={onClick} disabled={disabled} className={className}>{children}</button>,
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    StatCard: ({ label, value }: any) => <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span></div>,
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    FoldSection: ({ heading, children }: any) => <details open><summary>{heading}</summary><div>{children}</div></details>,
  }
})

const mockSearchParams = vi.hoisted(() => ({ dataset: null as string | null }))
const mockSession = vi.hoisted(() => ({ trainingRunning: false }))
const mockDatasets = vi.hoisted(() => ({
  datasets: [] as any[], selectedDataset: '', loadingDatasets: false, importModalOpen: false, datasetPreview: null,
  setSelectedDataset: vi.fn(), setImportModalOpen: vi.fn(), setDatasetPreview: vi.fn(), fetchDatasets: vi.fn(),
}))
const mockCheckpoints = vi.hoisted(() => ({
  checkpoints: [] as any[], loadingCheckpoints: false, loadingJobs: false, activeCheckpoint: null, builds: [], loadingBuilds: false, jobs: [],
  setActiveCheckpoint: vi.fn(), setCheckpoints: vi.fn(), fetchCheckpoints: vi.fn(), fetchBuilds: vi.fn(), fetchJobs: vi.fn(),
  handleLoadCheckpoint: vi.fn(), handleDeleteCheckpoint: vi.fn(),
}))
const mockForm = vi.hoisted(() => ({
  canStart: true, inputMode: 'dataset', allJobs: [] as any[], startTraining: vi.fn(),
}))
const mockTest = vi.hoisted(() => ({
  testDialogOpen: false, testPrompt: '', testResult: null, testLoading: false,
  setTestDialogOpen: vi.fn(), setTestPrompt: vi.fn(), handleTestModel: vi.fn(), clearTest: vi.fn(),
}))

const { mockExportMetrics, mockModelStatus, mockPreview, mockAddToast } = vi.hoisted(() => ({
  mockExportMetrics: vi.fn(), mockModelStatus: vi.fn(), mockPreview: vi.fn(), mockAddToast: vi.fn(),
}))

vi.mock('next/navigation', () => ({ useSearchParams: () => ({ get: (k: string) => (mockSearchParams as Record<string, string | null>)[k] ?? null }) }))
vi.mock('@/lib/controllers', () => ({
  datasetController: { preview: mockPreview },
  modelController: { status: mockModelStatus },
}))
vi.mock('@/lib/training-controller', () => ({ trainingJobsController: { exportMetrics: mockExportMetrics } }))
vi.mock('@/lib/souls-controller', () => ({ soulsController: { loadCheckpoint: vi.fn() } }))
vi.mock('@/lib/toast-store', () => ({ useToastStore: (sel: any) => sel({ addToast: mockAddToast }) }))
vi.mock('@/components/AppRouteHeader', () => ({
  AppRouteHeader: ({ left, right }: any) => <div data-testid="app-route-header">{left}{right}</div>,
  AppRouteHeaderLead: ({ title, subtitle }: any) => <div data-testid="app-route-header-lead">{title}<span>{subtitle}</span></div>,
}))
vi.mock('@/hooks/useTrainingSession', () => ({ useTrainingSession: () => mockSession }))
vi.mock('@/hooks/useTrainingDatasets', () => ({ useTrainingDatasets: () => mockDatasets }))
vi.mock('@/hooks/useTrainingCheckpoints', () => ({ useTrainingCheckpoints: () => mockCheckpoints }))
vi.mock('@/hooks/useTestDialog', () => ({ useTestDialog: () => mockTest }))
vi.mock('@/hooks/useTrainingForm', () => ({ useTrainingForm: () => mockForm }))
vi.mock('@/hooks/useLiveStatus', () => ({ useApiReady: () => true }))
vi.mock('@/components/training/TrainingSummaryCard', () => ({ TrainingSummaryCard: () => null }))
vi.mock('@/components/training/TrainingHealthCard', () => ({ TrainingHealthCard: () => null }))
vi.mock('@/components/OutputCard', () => ({ OutputCard: () => null }))
vi.mock('@/components/training/TestModelDialog', () => ({ TestModelDialog: () => null }))
vi.mock('@/components/training/TrainingPipeline', () => ({
  TrainingPipeline: ({ form, datasets, session, checkpoints, onTest }: any) => (
    <div data-testid="training-pipeline">
      <span>pipeline:{checkpoints.checkpoints.length}</span>
      <span>jobs:{form.allJobs.length}</span>
      {session.trainingRunning && <span>running</span>}
    </div>
  ),
}))

import Page from './page'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })
beforeEach(() => {
  vi.clearAllMocks()
  mockSearchParams.dataset = null
  mockSession.trainingRunning = false
  mockDatasets.datasets = []
  mockDatasets.selectedDataset = ''
  mockCheckpoints.checkpoints = []
  mockCheckpoints.loadingCheckpoints = false
  mockCheckpoints.loadingJobs = false
  mockForm.allJobs = []
  mockForm.canStart = true
  mockForm.inputMode = 'dataset'
  mockModelStatus.mockResolvedValue({ model_type: 'gpt2' })
  mockExportMetrics.mockResolvedValue(new Blob())
  vi.stubGlobal('URL', { createObjectURL: vi.fn(() => 'blob:fake'), revokeObjectURL: vi.fn() })
})

describe('TrainingPage', () => {
  it('renders header and zero-count stats on initial load', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Teach me')).toBeTruthy() })
    expect(screen.getByText('Teach your agent from your data')).toBeTruthy()
    expect(screen.getByTestId('stat-Training runs').textContent).toContain('0')
    expect(screen.getByTestId('stat-Running').textContent).toContain('0')
    expect(screen.getByTestId('stat-Completed').textContent).toContain('0')
    expect(screen.getByTestId('stat-Saved versions').textContent).toContain('0')
  })

  it('shows stat counts derived from jobs and checkpoints', async () => {
    mockForm.allJobs = [
      { id: 'a', status: 'running' },
      { id: 'b', status: 'completed' },
      { id: 'c', status: 'completed' },
    ]
    mockCheckpoints.checkpoints = [{ name: 'cp-1' }, { name: 'cp-2' }]
    render(<Page />)
    await waitFor(() => { expect(screen.getByTestId('stat-Training runs').textContent).toContain('3') })
    expect(screen.getByTestId('stat-Running').textContent).toContain('1')
    expect(screen.getByTestId('stat-Completed').textContent).toContain('2')
    expect(screen.getByTestId('stat-Saved versions').textContent).toContain('2')
  })

  it('fetches datasets, checkpoints, and jobs on mount', async () => {
    render(<Page />)
    await waitFor(() => { expect(mockDatasets.fetchDatasets).toHaveBeenCalled() })
    expect(mockCheckpoints.fetchCheckpoints).toHaveBeenCalled()
    expect(mockCheckpoints.fetchJobs).toHaveBeenCalled()
  })

  it('previews the selected dataset in dataset mode', async () => {
    mockDatasets.selectedDataset = 'ds-1'
    mockForm.inputMode = 'dataset'
    mockPreview.mockResolvedValue({ id: 'ds-1', samples: 3 })
    render(<Page />)
    await waitFor(() => { expect(mockPreview).toHaveBeenCalledWith('ds-1', 3) })
    expect(mockDatasets.setDatasetPreview).toHaveBeenCalledWith({ id: 'ds-1', samples: 3 })
  })

  it('auto-selects the dataset from the dataset search param', async () => {
    mockSearchParams.dataset = 'ds-9'
    mockDatasets.datasets = [{ id: 'ds-1' }]
    render(<Page />)
    await waitFor(() => { expect(mockDatasets.setSelectedDataset).toHaveBeenCalledWith('ds-9') })
    expect(mockDatasets.setSelectedDataset).not.toHaveBeenCalledWith('ds-1')
  })

  it('auto-selects the first dataset when no search param is present', async () => {
    mockDatasets.datasets = [{ id: 'ds-1' }, { id: 'ds-2' }]
    render(<Page />)
    await waitFor(() => { expect(mockDatasets.setSelectedDataset).toHaveBeenCalledWith('ds-1') })
  })

  it('exports metrics on button click and toasts success', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Export metrics')).toBeTruthy() })
    await act(async () => { screen.getByText('Export metrics').click() })
    await waitFor(() => { expect(mockExportMetrics).toHaveBeenCalled() })
    expect(mockAddToast).toHaveBeenCalledWith('Metrics exported', 'success')
    expect(URL.createObjectURL).toHaveBeenCalledWith(expect.any(Blob))
  })

  it('toasts an error when exporting metrics fails', async () => {
    mockExportMetrics.mockRejectedValue(new Error('boom'))
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Export metrics')).toBeTruthy() })
    await act(async () => { screen.getByText('Export metrics').click() })
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Failed to export metrics', 'error') })
  })

  it('refreshes jobs and checkpoints when Refresh is clicked', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Refresh')).toBeTruthy() })
    await act(async () => { screen.getByText('Refresh').click() })
    await waitFor(() => { expect(mockCheckpoints.fetchJobs).toHaveBeenCalled() })
    expect(mockCheckpoints.fetchCheckpoints).toHaveBeenCalled()
  })

  it('starts training via Ctrl+Enter keyboard shortcut and toasts', async () => {
    render(<Page />)
    await waitFor(() => { expect(mockDatasets.fetchDatasets).toHaveBeenCalled() })
    await act(async () => {
      fireEvent.keyDown(window, { key: 'Enter', ctrlKey: true })
    })
    expect(mockForm.startTraining).toHaveBeenCalled()
    expect(mockAddToast).toHaveBeenCalledWith('Training started', 'success')
  })

  it('does not start training via Ctrl+Enter when canStart is false', async () => {
    mockForm.canStart = false
    render(<Page />)
    await waitFor(() => { expect(mockDatasets.fetchDatasets).toHaveBeenCalled() })
    await act(async () => {
      fireEvent.keyDown(window, { key: 'Enter', ctrlKey: true })
    })
    expect(mockForm.startTraining).not.toHaveBeenCalled()
  })
})
