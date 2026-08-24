import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...args: any[]) => args.join(' '),
    Button: ({ children, onClick, disabled, variant, size, className }: any) => (
      <button onClick={onClick} disabled={disabled} data-variant={variant} className={className}>{children}</button>
    ),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    Input: ({ value, onChange, type, id, className, placeholder, inputMode }: any) => (
      <input value={value} onChange={onChange} type={type} id={id} className={className} placeholder={placeholder} inputMode={inputMode} />
    ),
    Label: ({ children, htmlFor, variant }: any) => <label htmlFor={htmlFor} data-variant={variant}>{children}</label>,
    Progress: ({ value, max }: any) => <div role="progressbar" aria-valuenow={value} aria-valuemax={max} />,
    Skeleton: () => <div data-testid="skeleton" />,
    StatCard: ({ label, value }: any) => <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span></div>,
    KpiGrid: ({ children, columns }: any) => <div data-columns={columns}>{children}</div>,
    Badge: ({ label, variant, size, className }: any) => <span data-variant={variant} className={className}>{label || children}</span>,
    Slider: ({ value, onValueChange, min, max, step }: any) => (
      <input type="range" value={value?.[0]} min={min} max={max} step={step}
        onChange={e => onValueChange?.([Number(e.target.value)])} />
    ),
    SettingsRow: ({ title, children, control }: any) => <div><span>{title}</span>{control}</div>,
    IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
    IconTrash: () => <span data-testid="icon-trash">trash</span>,
    FoldSection: ({ heading, children }: any) => <details open><summary>{heading}</summary><div>{children}</div></details>,
  }
})

const mocks = vi.hoisted(() => ({
  addToast: vi.fn(),
  session: {
    trainingRunning: false, paused: false, loss: null, progress: 0, globalStep: 0, totalSteps: 0,
    stepsPerSec: null, epoch: 0, totalEpochs: 0, eta: 0, elapsedSeconds: 0, message: '',
    lossHistory: [] as any[],
    startSSETraining: vi.fn(), stopTraining: vi.fn(), pauseTraining: vi.fn(), resumeTraining: vi.fn(),
  },
  datasets: {
    datasets: [] as any[], selectedDataset: '', loadingDatasets: false, importModalOpen: false, datasetPreview: null,
    setSelectedDataset: vi.fn(), setImportModalOpen: vi.fn(), setDatasetPreview: vi.fn(), fetchDatasets: vi.fn(),
  },
  checkpoints: {
    checkpoints: [] as any[], loading: false, fetchCheckpoints: vi.fn(),
  },
  startAutoTrain: vi.fn(),
  stopAutoTrain: vi.fn(),
  getTrainingLog: vi.fn(),
  loadCheckpoint: vi.fn(),
  deleteCheckpoint: vi.fn(),
  listDatasets: vi.fn(),
  listCheckpoints: vi.fn(),
}))

vi.mock('next/navigation', () => ({ useParams: () => ({}), useRouter: () => ({ push: vi.fn() }) }))
vi.mock('@/lib/toast-store', () => ({ useToastStore: (sel: any) => sel({ addToast: mocks.addToast }) }))
vi.mock('@/hooks/useLiveStatus', () => ({ useApiReady: () => true }))
vi.mock('@/hooks/useTrainingSession', () => ({ useTrainingSession: () => mocks.session }))
vi.mock('@/hooks/useTrainingDatasets', () => ({ useTrainingDatasets: () => mocks.datasets }))
vi.mock('@/hooks/useTrainingCheckpoints', () => ({ useTrainingCheckpoints: () => mocks.checkpoints }))
vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    startAutoTrain: mocks.startAutoTrain,
    stopAutoTrain: mocks.stopAutoTrain,
    getTrainingLog: mocks.getTrainingLog,
    loadCheckpoint: mocks.loadCheckpoint,
    deleteCheckpoint: mocks.deleteCheckpoint,
    listDatasets: mocks.listDatasets,
    listCheckpoints: mocks.listCheckpoints,
  },
}))
vi.mock('@/lib/controllers', () => ({
  datasetController: { list: mocks.listDatasets },
}))
vi.mock('@/components/training/LossChart', () => ({
  LossChart: ({ data }: any) => <div data-testid="loss-chart">LossChart</div>,
}))
vi.mock('@/components/training/DatasetSelector', () => ({
  DatasetSelector: ({ value, onChange }: any) => (
    <select value={value} onChange={e => onChange(e.target.value)}><option>ds-1</option></select>
  ),
}))
vi.mock('@/components/training/formatDuration', () => ({
  formatDuration: (n: number) => `${n}s`,
}))
vi.mock('next/dynamic', () => ({ default: () => () => <div data-testid="dynamic" /> }))

import AutoTrainPage from './page'

beforeEach(() => {
  vi.clearAllMocks()
  mocks.getTrainingLog.mockResolvedValue(['line 1', 'line 2'])
  mocks.loadCheckpoint.mockResolvedValue({})
  mocks.deleteCheckpoint.mockResolvedValue({})
  mocks.session.trainingRunning = false
  mocks.session.paused = false
  mocks.session.loss = null
  mocks.session.progress = 0
  mocks.session.globalStep = 0
  mocks.session.totalSteps = 0
  mocks.session.stepsPerSec = null
  mocks.session.epoch = 0
  mocks.session.totalEpochs = 0
  mocks.session.eta = 0
  mocks.session.elapsedSeconds = 0
  mocks.session.message = ''
  mocks.session.lossHistory = []
})

afterEach(() => cleanup())

describe('AutoTrainPage', () => {
  it('renders page title', async () => {
    render(<AutoTrainPage />)
    expect(screen.getByText('Auto-train')).toBeTruthy()
  })

  it('shows stat cards', async () => {
    render(<AutoTrainPage />)
    expect(screen.getByText('Checkpoints')).toBeTruthy()
    expect(screen.getByText('Trained')).toBeTruthy()
    expect(screen.getByText('Status')).toBeTruthy()
    expect(screen.getByText('Current loss')).toBeTruthy()
  })

  it('shows idle status when not training', async () => {
    render(<AutoTrainPage />)
    expect(screen.getByText('Idle')).toBeTruthy()
  })

  it('shows training status when running', async () => {
    mocks.session.trainingRunning = true
    mocks.session.progress = 50
    mocks.session.globalStep = 100
    mocks.session.totalSteps = 200
    mocks.session.epoch = 1
    mocks.session.totalEpochs = 5
    mocks.session.stepsPerSec = 10
    mocks.session.eta = 600
    mocks.session.elapsedSeconds = 300
    render(<AutoTrainPage />)
    expect(screen.getByText('Training')).toBeTruthy()
    expect(screen.getByText('Stop')).toBeTruthy()
  })

  it('shows input mode tabs', async () => {
    render(<AutoTrainPage />)
    expect(screen.getByText('Text input')).toBeTruthy()
    expect(screen.getByText('Dataset')).toBeTruthy()
    expect(screen.getByText('Resume checkpoint')).toBeTruthy()
  })

  it('shows text input mode by default', async () => {
    render(<AutoTrainPage />)
    expect(screen.getByText('Source text')).toBeTruthy()
  })

  it('switches to dataset mode', async () => {
    render(<AutoTrainPage />)
    fireEvent.click(screen.getByText('Dataset'))
    expect(screen.getByText('ds-1')).toBeTruthy()
  })

  it('shows training params', async () => {
    render(<AutoTrainPage />)
    expect(screen.getByText('Epochs')).toBeTruthy()
    expect(screen.getByText('LR')).toBeTruthy()
    expect(screen.getByText('Teacher')).toBeTruthy()
    expect(screen.getByText('Temperature')).toBeTruthy()
  })

  it('enables start button only when canStart', async () => {
    render(<AutoTrainPage />)
    const startBtn = screen.getByRole('button', { name: /Start auto-train/ })
    expect(startBtn).toBeDisabled()

    const textareas = screen.getAllByRole('textbox')
    fireEvent.change(textareas[0], { target: { value: 'hello' } })
    expect(screen.getByRole('button', { name: /Start auto-train/ })).not.toBeDisabled()
  })

  it('shows show logs button', async () => {
    render(<AutoTrainPage />)
    expect(screen.getByText('Show logs')).toBeTruthy()
  })

  it('fetches and shows logs', async () => {
    render(<AutoTrainPage />)
    fireEvent.click(screen.getByText('Show logs'))
    await waitFor(() => {
      expect(mocks.getTrainingLog).toHaveBeenCalled()
    })
  })

  it('shows error toast when logs fail', async () => {
    mocks.getTrainingLog.mockRejectedValue(new Error('boom'))
    render(<AutoTrainPage />)
    fireEvent.click(screen.getByText('Show logs'))
    await waitFor(() => {
      expect(mocks.addToast).toHaveBeenCalledWith('Could not fetch logs', 'error')
    })
  })

  it('stops training', async () => {
    mocks.stopAutoTrain.mockResolvedValue({})
    mocks.session.trainingRunning = true
    render(<AutoTrainPage />)
    fireEvent.click(screen.getByText('Stop'))
    await waitFor(() => {
      expect(mocks.stopAutoTrain).toHaveBeenCalled()
      expect(mocks.addToast).toHaveBeenCalledWith('Training stopped', 'success')
    })
  })

  it('shows error toast on stop failure', async () => {
    mocks.stopAutoTrain.mockRejectedValue(new Error('fail'))
    mocks.session.trainingRunning = true
    render(<AutoTrainPage />)
    fireEvent.click(screen.getByText('Stop'))
    await waitFor(() => {
      expect(mocks.addToast).toHaveBeenCalledWith('Could not stop training', 'error')
    })
  })

  it('loads checkpoint', async () => {
    mocks.checkpoints.checkpoints = [{ name: 'cp-1', loss: 0.5 }]
    render(<AutoTrainPage />)
    await waitFor(() => { expect(screen.getByText('Load')).toBeTruthy() })
    fireEvent.click(screen.getByText('Load'))
    await waitFor(() => {
      expect(mocks.loadCheckpoint).toHaveBeenCalledWith('cp-1')
      expect(mocks.addToast).toHaveBeenCalledWith('Loaded checkpoint: cp-1', 'success')
    })
  })

  it('deletes checkpoint', async () => {
    mocks.checkpoints.checkpoints = [{ name: 'cp-1', loss: 0.5 }]
    render(<AutoTrainPage />)
    await waitFor(() => { expect(screen.getByText('Delete')).toBeTruthy() })
    fireEvent.click(screen.getByText('Delete'))
    await waitFor(() => {
      expect(mocks.deleteCheckpoint).toHaveBeenCalledWith('cp-1')
      expect(mocks.addToast).toHaveBeenCalledWith('Deleted checkpoint: cp-1', 'success')
    })
  })
})
