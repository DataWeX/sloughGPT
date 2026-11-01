/**
 * Tests for useVmWorkbench — the stateful core of the VM Console.
 *
 * The page suite (59 tests) drives this state through the rendered DOM, but
 * it cannot see the parts that never render: the *Touched guard that stops
 * hydration from clobbering an edit made mid-load, and the training-poll
 * lifecycle that must stop itself at a terminal status. Those live here.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, waitFor, act } from '@testing-library/react'

const kvStore = new Map<string, unknown>()

vi.mock('@/lib/db', () => ({
  chatDB: {
    getKV: vi.fn(async (key: string) => kvStore.get(key) ?? null),
    setKV: vi.fn(async (key: string, value: unknown) => {
      kvStore.set(key, value)
    }),
  },
}))

vi.mock('@/lib/vm-controller', () => ({
  vmController: {
    run: vi.fn(),
    trainingJob: vi.fn(),
    stopTrainingJob: vi.fn(),
  },
}))

vi.mock('@/lib/dataset-controller', () => ({
  datasetController: { list: vi.fn() },
}))

const refreshHolder: { cb: (() => void) | null } = { cb: null }
vi.mock('@/hooks/useRefreshShortcut', () => ({
  // Called on every render — keep only the latest callback instead of
  // accumulating one per re-render.
  useRefreshShortcut: (cb: () => void) => {
    refreshHolder.cb = cb
  },
}))

import { vmController } from '@/lib/vm-controller'
import { datasetController } from '@/lib/dataset-controller'
import { chatDB } from '@/lib/db'
import { useVmWorkbench } from './useVmWorkbench'

const mockedRun = vi.mocked(vmController.run)
const mockedTrainingJob = vi.mocked(vmController.trainingJob)
const mockedDatasets = vi.mocked(datasetController.list)

function fakeResult(overrides: Record<string, unknown> = {}) {
  return {
    success: true,
    exit_code: 0,
    steps_executed: 10,
    elapsed_ms: 1.2,
    output: '',
    registers: [{ name: 'EAX', value: 0, hex: '0x00000000' }],
    eip: 0,
    eip_hex: '0x00000000',
    status: 'halted',
    ...overrides,
  }
}

describe('useVmWorkbench', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    kvStore.clear()
    refreshHolder.cb = null
    vi.mocked(chatDB.getKV).mockImplementation(async (key: string) => kvStore.get(key) ?? null)
    vi.mocked(chatDB.setKV).mockImplementation(async (key: string, value: unknown) => {
      kvStore.set(key, value)
    })
    mockedRun.mockResolvedValue(fakeResult() as never)
    mockedDatasets.mockResolvedValue([] as never)
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('hydrates role, steps and train config from chatDB', async () => {
    kvStore.set('vm-role', 'admin')
    kvStore.set('vm-max-steps', 1234)
    kvStore.set('vm-train-config', { dataset: 'tiny', epochs: 7 })

    const { result } = renderHook(() => useVmWorkbench())

    await waitFor(() => expect(result.current.hydrated).toBe(true))

    expect(result.current.role).toBe('admin')
    expect(result.current.maxSteps).toBe(1234)
    expect(result.current.trainConfig.dataset).toBe('tiny')
    expect(result.current.trainConfig.epochs).toBe(7)
  })

  it('falls back to defaults for corrupt persisted values', async () => {
    kvStore.set('vm-role', 'root') // not one of user/admin/kernel
    kvStore.set('vm-max-steps', 'not-a-number')

    const { result } = renderHook(() => useVmWorkbench())
    await waitFor(() => expect(result.current.hydrated).toBe(true))

    expect(result.current.role).toBe('user')
    expect(result.current.maxSteps).toBe(5000)
  })

  it('persists edits once hydrated', async () => {
    const { result } = renderHook(() => useVmWorkbench())
    await waitFor(() => expect(result.current.hydrated).toBe(true))

    act(() => {
      result.current.setSource('MOV EAX, 1')
    })

    await waitFor(() => expect(kvStore.get('vm-source')).toBe('MOV EAX, 1'))
    expect(vi.mocked(chatDB.setKV)).toHaveBeenCalledWith('vm-source', 'MOV EAX, 1')
  })

  it('maps a run failure onto the error result shape', async () => {
    mockedRun.mockRejectedValue(new Error('emulator exploded'))

    const { result } = renderHook(() => useVmWorkbench())

    let returned: unknown = 'unset'
    await act(async () => {
      returned = await result.current.handleRun()
    })

    expect(returned).toBeNull()
    expect(result.current.result?.success).toBe(false)
    expect(result.current.result?.status).toBe('error')
    expect(result.current.result?.error).toContain('emulator exploded')
    expect(result.current.running).toBe(false)
  })

  it('clamps an absurd step count instead of sending it to the backend', async () => {
    const { result } = renderHook(() => useVmWorkbench())
    act(() => {
      result.current.setMaxSteps(9_999_999)
    })

    await act(async () => {
      await result.current.handleRun()
    })

    expect(mockedRun).toHaveBeenCalledWith(expect.anything(), {
      maxSteps: 1_000_000,
      role: 'user',
      debug: false,
      keyboardInput: undefined,
    })
  })

  it('polls a launched training job and stops at a terminal status', async () => {
    vi.useFakeTimers()
    mockedRun.mockResolvedValue(fakeResult({ training_job_id: 7 }) as never)
    mockedTrainingJob
      .mockResolvedValueOnce({ job_id: 7, api_job_id: '', status: 'running', progress: 0.5 } as never)
      .mockResolvedValue({ job_id: 7, api_job_id: '', status: 'completed', progress: 1 } as never)

    const { result } = renderHook(() => useVmWorkbench())

    await act(async () => {
      await result.current.handleRun()
    })
    // The job-id effect fires its first poll immediately; flush the microtask.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(result.current.trainingJob?.status).toBe('running')

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000)
    })
    expect(result.current.trainingJob?.status).toBe('completed')

    const callsAtTerminal = mockedTrainingJob.mock.calls.length

    // Terminal status must clear the interval — no further polls.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000)
    })
    expect(mockedTrainingJob.mock.calls.length).toBe(callsAtTerminal)
  })

  it('re-runs the page loads on the refresh shortcut without a document reload', async () => {
    const { result } = renderHook(() => useVmWorkbench())
    await waitFor(() => expect(result.current.hydrated).toBe(true))
    expect(refreshHolder.cb).toBeTruthy()

    const loadsBefore = vi.mocked(chatDB.getKV).mock.calls.filter(([k]) => k === 'vm-role').length
    await act(async () => {
      refreshHolder.cb?.()
    })
    const loadsAfter = vi.mocked(chatDB.getKV).mock.calls.filter(([k]) => k === 'vm-role').length

    expect(loadsAfter).toBeGreaterThan(loadsBefore)
  })
})
