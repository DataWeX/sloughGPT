/**
 * Unit tests for the useTrainingJob page-model hook extracted from the
 * training job detail page in the Phase 2 split.
 */
import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest'
import { renderHook, act, cleanup, waitFor } from '@testing-library/react'
import { useTrainingJob } from './useTrainingJob'
import type { TrainingJob } from '@/lib/training-controller'

const mockPush = vi.fn()
vi.mock('@/vite/next-compat/navigation', () => ({
  useParams: () => ({ id: 'job-1' }),
  useRouter: () => ({ push: mockPush }),
}))

const mockGet = vi.fn()
const mockGetSummary = vi.fn()
const mockDelete = vi.fn()
const mockRecover = vi.fn()
const mockStop = vi.fn()
const mockDownloadTrainingJob = vi.fn()
vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    get: (...args: unknown[]) => mockGet(...args),
    getSummary: (...args: unknown[]) => mockGetSummary(...args),
    delete: (...args: unknown[]) => mockDelete(...args),
    recover: (...args: unknown[]) => mockRecover(...args),
    stop: (...args: unknown[]) => mockStop(...args),
    downloadTrainingJob: (...args: unknown[]) => mockDownloadTrainingJob(...args),
  },
}))

const mockLoadModelPath = vi.fn()
vi.mock('@/lib/model-controller', () => ({
  modelController: { loadModelPath: (...args: unknown[]) => mockLoadModelPath(...args) },
}))

const mockAddToast = vi.fn()
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: typeof mockAddToast }) => unknown) =>
    selector({ addToast: mockAddToast }),
}))

const mockDownloadJson = vi.fn()
const mockDownloadBlob = vi.fn()
vi.mock('@/lib/download-utils', () => ({
  downloadJson: (...args: unknown[]) => mockDownloadJson(...args),
  downloadBlob: (...args: unknown[]) => mockDownloadBlob(...args),
}))

const JOB: TrainingJob = {
  id: 'job-1',
  name: 'My fine-tune',
  status: 'completed',
  progress: 100,
  created_at: '2026-10-09T00:00:00Z',
  model: 'gpt2',
  checkpoint: '/ckpt/best.pt',
}

async function renderLoaded() {
  const utils = renderHook(() => useTrainingJob())
  await waitFor(() => expect(utils.result.current.loading).toBe(false))
  expect(utils.result.current.job?.id).toBe('job-1')
  return utils
}

afterEach(() => {
  cleanup()
})

beforeEach(() => {
  vi.clearAllMocks()
  mockGet.mockResolvedValue(JOB)
  mockGetSummary.mockResolvedValue({ summary: 'A trained model.' })
  mockDelete.mockResolvedValue({})
  mockRecover.mockResolvedValue({ message: 'Resume started' })
  mockStop.mockResolvedValue({})
  mockDownloadTrainingJob.mockResolvedValue(new Blob(['x']))
  mockLoadModelPath.mockResolvedValue({})
  mockDownloadJson.mockReturnValue(undefined)
})

describe('useTrainingJob', () => {
  it('loads the job and its summary on mount', async () => {
    await renderLoaded()
    expect(mockGet).toHaveBeenCalledWith('job-1')
    expect(mockGetSummary).toHaveBeenCalledWith('job-1')
  })

  it('maps known statuses to badges and falls back for unknown ones', async () => {
    mockGet.mockResolvedValue({ ...JOB, status: 'failed' })
    const failed = renderHook(() => useTrainingJob())
    await waitFor(() => expect(failed.result.current.loading).toBe(false))
    expect(failed.result.current.badge).toEqual({ label: 'Failed', variant: 'destructive' })
    cleanup()

    mockGet.mockResolvedValue({ ...JOB, status: 'mystery' })
    const unknown = renderHook(() => useTrainingJob())
    await waitFor(() => expect(unknown.result.current.loading).toBe(false))
    expect(unknown.result.current.badge).toEqual({ label: 'mystery', variant: 'outline' })
  })

  it('deletes the job, closes the dialog, and navigates to /training', async () => {
    const { result } = await renderLoaded()
    act(() => {
      result.current.setShowDelete(true)
    })
    await act(async () => {
      await result.current.handleDelete()
    })
    expect(mockDelete).toHaveBeenCalledWith('job-1')
    expect(mockPush).toHaveBeenCalledWith('/training')
    expect(result.current.showDelete).toBe(false)
    expect(mockAddToast).toHaveBeenCalledWith('Job deleted', 'info')
  })

  it('exports job details as JSON', async () => {
    const { result } = await renderLoaded()
    act(() => {
      result.current.handleExport()
    })
    expect(mockDownloadJson).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'job-1', name: 'My fine-tune' }),
      'training-job-job-1.json',
    )
    expect(mockAddToast).toHaveBeenCalledWith('Job details exported', 'success')
  })

  it('routes resume to the recovery job when it differs', async () => {
    mockRecover.mockResolvedValue({ message: 'Resume started', recovery_job_id: 'job-9' })
    const { result } = await renderLoaded()
    await act(async () => {
      await result.current.handleResume()
    })
    expect(mockRecover).toHaveBeenCalledWith('job-1')
    expect(mockPush).toHaveBeenCalledWith('/training/job/job-9')
  })

  it('refetches in place when resume targets the same job', async () => {
    mockRecover.mockResolvedValue({ message: 'Resume started', recovery_job_id: 'job-1' })
    const { result } = await renderLoaded()
    const callsBefore = mockGet.mock.calls.length
    await act(async () => {
      await result.current.handleResume()
    })
    expect(mockPush).not.toHaveBeenCalled()
    expect(mockGet.mock.calls.length).toBeGreaterThan(callsBefore)
  })

  it('loads the checkpoint then routes to /chat for try-in-chat', async () => {
    const { result } = await renderLoaded()
    await act(async () => {
      await result.current.handleTryInChat()
    })
    expect(mockLoadModelPath).toHaveBeenCalledWith('/ckpt/best.pt')
    expect(mockPush).toHaveBeenCalledWith('/chat')
  })
})
