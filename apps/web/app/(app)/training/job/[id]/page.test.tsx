import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

// ── strui mock ──
vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => { const C = () => <span data-testid={`icon-${name}`}>{name}</span>; C.displayName = `Icon${name}`; return C }
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    Button: ({ children, onClick, disabled, 'aria-label': ariaLabel, className }: any) => (
      <button onClick={onClick} disabled={disabled} aria-label={ariaLabel} className={className}>{children}</button>
    ),
    Skeleton: () => <div data-testid="skeleton" />,
    Badge: ({ children, variant }: any) => <span data-variant={variant}>{children}</span>,
    StatCard: ({ label, value }: any) => <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span></div>,
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    Breadcrumbs: ({ items }: any) => <nav aria-label="Breadcrumb">{items?.map((item: any, i: number) => <span key={i}>{item.label}</span>)}</nav>,
    IconTrash: iconMock('trash'), IconRefresh: iconMock('refresh'), IconDownload: iconMock('download'),
    AlertDialog: ({ open, onOpenChange, children }: any) => open ? <div data-testid="alert-dialog">{children}</div> : null,
    AlertDialogContent: ({ children }: any) => <div>{children}</div>,
    AlertDialogHeader: ({ children }: any) => <div>{children}</div>,
    AlertDialogTitle: ({ children }: any) => <div>{children}</div>,
    AlertDialogDescription: ({ children }: any) => <div>{children}</div>,
    AlertDialogFooter: ({ children }: any) => <div>{children}</div>,
    AlertDialogCancel: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    AlertDialogAction: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
  }
})

// ── controller & router mocks ──
const { mockGet, mockGetSummary, mockDelete, mockStop, mockDownloadTrainingJob, mockLoadModelPath, mockPush, mockAddToast, mockDownloadBlob, mockDownloadJson } = vi.hoisted(() => ({
  mockGet: vi.fn(), mockGetSummary: vi.fn(), mockDelete: vi.fn(), mockStop: vi.fn(),
  mockDownloadTrainingJob: vi.fn(), mockLoadModelPath: vi.fn(), mockPush: vi.fn(), mockAddToast: vi.fn(),
  mockDownloadBlob: vi.fn(), mockDownloadJson: vi.fn(),
}))

const stableRouter = { push: mockPush }
vi.mock('next/navigation', () => ({ useParams: () => ({ id: 'job-1' }), useRouter: () => stableRouter }))
vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    get: mockGet, getSummary: mockGetSummary, delete: mockDelete, stop: mockStop,
    downloadTrainingJob: mockDownloadTrainingJob,
  },
}))
vi.mock('@/lib/model-controller', () => ({ modelController: { loadModelPath: mockLoadModelPath } }))
vi.mock('@/lib/toast-store', () => ({ useToastStore: (sel: any) => sel({ addToast: mockAddToast }) }))
vi.mock('@/lib/download-utils', () => ({ downloadBlob: mockDownloadBlob, downloadJson: mockDownloadJson }))
vi.mock('next/dynamic', () => ({ default: () => () => <div data-testid="loss-chart" /> }))
vi.mock('@/components/AppRouteHeader', () => ({
  AppRouteHeader: ({ left, right }: any) => <div data-testid="app-route-header">{left}{right}</div>,
  AppRouteHeaderLead: ({ title }: any) => <div data-testid="app-route-header-lead">{title}</div>,
}))

import Page from './page'

const COMPLETED_JOB: any = {
  id: 'job-1',
  name: 'Shakespeare Fine-tune',
  status: 'completed',
  progress: 100,
  created_at: '2026-08-01T00:00:00Z',
  finished_at: '2026-08-01T00:05:00Z',
  model: 'gpt2',
  dataset: 'shakespeare',
  epochs: 5,
  current_epoch: 5,
  global_step: 420,
  loss: 1.2345,
  train_loss: 1.1,
  eval_loss: 1.5,
  checkpoint: 'cp-20260801',
  data_source: 'dataset',
  message: 'Training finished successfully',
  explanation: 'This version performed well on your data.',
  result: { final_reward: 0.98 },
  loss_history: [
    { step: 1, value: 3, type: 'train' },
    { step: 2, value: 2, type: 'train' },
    { step: 3, value: 1, type: 'train' },
  ],
  reward_history: [{ step: 1, value: 0.5 }, { step: 2, value: 0.75 }],
}

const RUNNING_JOB: any = {
  id: 'job-1',
  name: 'Active Training',
  status: 'running',
  progress: 40,
  created_at: '2026-08-01T00:00:00Z',
  model: 'gpt2',
  dataset: 'shakespeare',
  epochs: 5,
  current_epoch: 2,
  global_step: 100,
  loss_history: [{ step: 1, value: 2, type: 'train' }],
}

function waitForName() {
  return waitFor(() => { expect(screen.getAllByText('Shakespeare Fine-tune').length).toBeGreaterThan(0) })
}

afterEach(() => { cleanup(); vi.useRealTimers() })
beforeEach(() => {
  vi.clearAllMocks()
  mockGetSummary.mockResolvedValue({ job_id: 'job-1', summary: 'Training completed successfully', status: 'completed', model: 'gpt2', dataset: 'shakespeare' })
  mockGet.mockResolvedValue(COMPLETED_JOB)
  mockDownloadTrainingJob.mockResolvedValue(new Blob())
  mockLoadModelPath.mockResolvedValue({ status: 'ok' })
})

describe('TrainingJobDetailPage', () => {
  it('shows loading skeletons and fetches job + summary on mount', () => {
    mockGet.mockReturnValue(new Promise(() => {}))
    const { container } = render(<Page />)
    expect(container.querySelectorAll('[data-testid="skeleton"]').length).toBeGreaterThanOrEqual(2)
    expect(mockGet).toHaveBeenCalledWith('job-1')
    expect(mockGetSummary).toHaveBeenCalledWith('job-1')
  })

  it('displays job details after loading', async () => {
    render(<Page />)
    await waitForName()
    expect(screen.getByText('Completed')).toBeTruthy()
    expect(screen.getByText('Training completed successfully')).toBeTruthy()
    expect(screen.getByText('This version performed well on your data.')).toBeTruthy()
    expect(screen.getByText('Summary')).toBeTruthy()
    expect(screen.getByText('Job ID')).toBeTruthy()
    expect(screen.getByText('job-1')).toBeTruthy()
    expect(screen.getByText('cp-20260801')).toBeTruthy()
    expect(screen.getByText('Training finished successfully')).toBeTruthy()
  })

  it('shows model/dataset/epoch/steps KPIs', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('gpt2')).toBeTruthy() })
    expect(screen.getByTestId('stat-Model').textContent).toContain('gpt2')
    expect(screen.getByTestId('stat-Dataset').textContent).toContain('shakespeare')
    expect(screen.getByTestId('stat-Epochs').textContent).toContain('5 / 5')
    expect(screen.getByTestId('stat-Steps').textContent).toContain('420')
  })

  it('shows loss and reward KPIs with trend badge', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByTestId('stat-Final loss')).toBeTruthy() })
    expect(screen.getByTestId('stat-Final loss').textContent).toContain('1.2345')
    expect(screen.getByTestId('stat-Train loss').textContent).toContain('1.1000')
    expect(screen.getByTestId('stat-Validation loss').textContent).toContain('1.5000')
    expect(screen.getByTestId('stat-Final reward').textContent).toContain('0.9800')
    expect(screen.getByText('↓ 50%')).toBeTruthy()
    expect(screen.getByTestId('loss-chart')).toBeTruthy()
  })

  it('falls back to job id when job has no name', async () => {
    mockGet.mockResolvedValue({ ...COMPLETED_JOB, name: '' })
    render(<Page />)
    await waitFor(() => { expect(screen.getAllByText('job-1').length).toBeGreaterThan(0) })
  })

  it('shows Summary not available when summary is empty', async () => {
    mockGetSummary.mockResolvedValue({ job_id: 'job-1', summary: '', status: 'completed', model: 'gpt2', dataset: 'shakespeare' })
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Summary not available')).toBeTruthy() })
  })

  it('shows Job not found when get returns null', async () => {
    mockGet.mockResolvedValue(null)
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Job not found')).toBeTruthy() })
  })

  it('shows error toast when loading fails', async () => {
    mockGet.mockRejectedValue(new Error('boom'))
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Job not found')).toBeTruthy() })
    expect(mockAddToast).toHaveBeenCalledWith('Something went wrong loading the job', 'error')
  })

  it('renders running job with progress bar and Stop button', async () => {
    mockGet.mockResolvedValue(RUNNING_JOB)
    render(<Page />)
    await waitFor(() => { expect(screen.getAllByText('Active Training').length).toBeGreaterThan(0) })
    expect(screen.getByText('Running')).toBeTruthy()
    expect(screen.getByText('40%')).toBeTruthy()
    expect(screen.getByText(/ETA:/)).toBeTruthy()
    expect(screen.getByText('Stop')).toBeTruthy()
  })

  it('stops a running job', async () => {
    mockGet.mockResolvedValue(RUNNING_JOB)
    mockStop.mockResolvedValue({})
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Stop')).toBeTruthy() })
    await act(async () => { screen.getByText('Stop').click() })
    await waitFor(() => { expect(mockStop).toHaveBeenCalledWith('job-1') })
    expect(mockAddToast).toHaveBeenCalledWith('Training stopped', 'info')
    expect(mockGet).toHaveBeenCalledWith('job-1')
  })

  it('loads saved version for completed job', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Load saved version')).toBeTruthy() })
    await act(async () => { screen.getByText('Load saved version').click() })
    await waitFor(() => { expect(mockLoadModelPath).toHaveBeenCalledWith('cp-20260801') })
    expect(mockAddToast).toHaveBeenCalledWith('Loaded trained version: cp-20260801', 'success')
  })

  it('exports checkpoint on export click', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Export')).toBeTruthy() })
    await act(async () => { screen.getByText('Export').click() })
    await waitFor(() => { expect(mockDownloadTrainingJob).toHaveBeenCalledWith('job-1') })
    expect(mockDownloadBlob).toHaveBeenCalledWith(expect.any(Blob), 'job-1.checkpoint')
    expect(mockAddToast).toHaveBeenCalledWith('Checkpoint downloaded', 'success')
  })

  it('exports job details JSON', async () => {
    render(<Page />)
    await waitForName()
    await act(async () => { screen.getByLabelText('Export job details').click() })
    await waitFor(() => { expect(mockDownloadJson).toHaveBeenCalled() })
    expect(mockDownloadJson).toHaveBeenCalledWith(expect.any(Object), 'training-job-job-1.json')
    expect(mockAddToast).toHaveBeenCalledWith('Job details exported', 'success')
  })

  it('deletes job after confirmation', async () => {
    mockDelete.mockResolvedValue({})
    render(<Page />)
    await waitForName()
    await act(async () => { screen.getByLabelText('Delete job').click() })
    await waitFor(() => { expect(screen.getByTestId('alert-dialog')).toBeTruthy() })
    const dialog = screen.getByTestId('alert-dialog')
    const confirmBtn = dialog.querySelector('button:last-child') as HTMLElement
    await act(async () => { confirmBtn.click() })
    await waitFor(() => { expect(mockDelete).toHaveBeenCalledWith('job-1') })
    expect(mockAddToast).toHaveBeenCalledWith('Job deleted', 'info')
    expect(mockPush).toHaveBeenCalledWith('/training')
  })

  it('cancels delete without deleting', async () => {
    render(<Page />)
    await waitForName()
    await act(async () => { screen.getByLabelText('Delete job').click() })
    await waitFor(() => { expect(screen.getByTestId('alert-dialog')).toBeTruthy() })
    const dialog = screen.getByTestId('alert-dialog')
    const cancelBtn = dialog.querySelector('button:first-child') as HTMLElement
    await act(async () => { cancelBtn.click() })
    expect(mockDelete).not.toHaveBeenCalled()
  })

  it('navigates to chat via Try in chat', async () => {
    render(<Page />)
    await waitFor(() => { expect(screen.getByText('Try in chat')).toBeTruthy() })
    await act(async () => { screen.getByText('Try in chat').click() })
    await waitFor(() => { expect(mockLoadModelPath).toHaveBeenCalledWith('cp-20260801') })
    expect(mockPush).toHaveBeenCalledWith('/chat')
  })

  it('shows failed status badge', async () => {
    mockGet.mockResolvedValue({ ...COMPLETED_JOB, name: 'Broken Job', status: 'failed', checkpoint: undefined })
    render(<Page />)
    await waitFor(() => { expect(screen.getAllByText('Broken Job').length).toBeGreaterThan(0) })
    expect(screen.getByText('Failed')).toBeTruthy()
  })

  it('refetches job when refresh button clicked', async () => {
    render(<Page />)
    await waitForName()
    await act(async () => { screen.getByLabelText('Refresh job status').click() })
    await waitFor(() => { expect(mockGet).toHaveBeenCalledTimes(2) })
  })

  it('polls a running job on an interval', async () => {
    vi.useFakeTimers()
    mockGet.mockResolvedValue(RUNNING_JOB)
    render(<Page />)
    await act(async () => {})
    const initialCalls = mockGet.mock.calls.length
    expect(initialCalls).toBeGreaterThanOrEqual(1)
    await act(async () => { await vi.advanceTimersByTimeAsync(3000) })
    expect(mockGet.mock.calls.length).toBeGreaterThan(initialCalls)
  })
})
