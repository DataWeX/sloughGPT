/**
 * Unit tests for the useTrainingRuns page-model hook extracted from the
 * training runs page in the Phase 2 split.
 */
import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest'
import { renderHook, act, cleanup, waitFor } from '@testing-library/react'
import { useTrainingRuns } from './useTrainingRuns'

const mockFilterTrainingRuns = vi.fn()
const mockCompareTrainingRuns = vi.fn()
const mockBulkAddTag = vi.fn()

vi.mock('@/lib/settings-controller', () => ({
  settingsController: {
    filterTrainingRuns: (...args: unknown[]) => mockFilterTrainingRuns(...args),
    exportTrainingHistory: vi.fn(() => Promise.resolve({ content: '' })),
    deleteTrainingRun: vi.fn(() => Promise.resolve({})),
    addRunTag: vi.fn(() => Promise.resolve({})),
    removeRunTag: vi.fn(() => Promise.resolve({})),
    setRunNotes: vi.fn(() => Promise.resolve({})),
    exportTrainingRun: vi.fn(() => Promise.resolve({ content: '' })),
    compareTrainingRuns: (...args: unknown[]) => mockCompareTrainingRuns(...args),
    toggleBookmark: vi.fn(() => Promise.resolve({})),
    duplicateTrainingRun: vi.fn(() => Promise.resolve({})),
    bulkDeleteRuns: vi.fn(() => Promise.resolve({})),
    bulkAddTag: (...args: unknown[]) => mockBulkAddTag(...args),
    bulkBookmark: vi.fn(() => Promise.resolve({})),
  },
}))

const RUN_A = {
  run_id: 'run-1',
  model: 'gpt2',
  converged: true,
  quality_score: 0.8,
  final_loss: 0.5,
  tags: ['prod'],
  notes: 'baseline notes',
  timestamp: 1700000000,
}
const RUN_B = {
  run_id: 'run-2',
  model: 'llama',
  quality_score: 0.4,
  final_loss: 0.3,
  timestamp: 1700000100,
}

async function renderLoaded() {
  const utils = renderHook(() => useTrainingRuns())
  await waitFor(() => expect(utils.result.current.loading).toBe(false))
  expect(utils.result.current.runs).toHaveLength(2)
  return utils
}

afterEach(() => {
  cleanup()
})

beforeEach(() => {
  vi.clearAllMocks()
  mockFilterTrainingRuns.mockResolvedValue({ runs: [RUN_A, RUN_B] })
  mockCompareTrainingRuns.mockResolvedValue({
    differences: { final_loss: { run_a: 0.5, run_b: 0.6 } },
  })
  mockBulkAddTag.mockResolvedValue({})
})

describe('useTrainingRuns', () => {
  it('computes summary stats from the loaded runs', async () => {
    const { result } = await renderLoaded()
    expect(result.current.summary.total).toBe(2)
    expect(result.current.summary.converged).toBe(1)
    expect(result.current.summary.avgQuality).toBeCloseTo(0.6)
    expect(result.current.summary.avgLoss).toBeCloseTo(0.4)
  })

  it('filters across model, tags, notes, and run id', async () => {
    const { result } = await renderLoaded()
    act(() => {
      result.current.setSearchQuery('llama')
    })
    expect(result.current.filteredRuns.map((r) => r.run_id)).toEqual(['run-2'])
    act(() => {
      result.current.setSearchQuery('prod')
    })
    expect(result.current.filteredRuns.map((r) => r.run_id)).toEqual(['run-1'])
    act(() => {
      result.current.setSearchQuery('run-2')
    })
    expect(result.current.filteredRuns.map((r) => r.run_id)).toEqual(['run-2'])
    act(() => {
      result.current.setSearchQuery('zzz-no-match')
    })
    expect(result.current.filteredRuns).toHaveLength(0)
  })

  it('toggles individual selection and select-all', async () => {
    const { result } = await renderLoaded()
    act(() => {
      result.current.toggleSelect('run-1')
    })
    expect(result.current.selectedIds).toEqual(new Set(['run-1']))
    act(() => {
      result.current.toggleSelect('run-1')
    })
    expect(result.current.selectedIds.size).toBe(0)
    act(() => {
      result.current.toggleSelectAll()
    })
    expect(result.current.selectedIds.size).toBe(2)
    act(() => {
      result.current.toggleSelectAll()
    })
    expect(result.current.selectedIds.size).toBe(0)
  })

  it('selectRun switches the selection and resets notes editing state', async () => {
    const { result } = await renderLoaded()
    act(() => {
      result.current.setEditingNotes(true)
      result.current.setNotesValue('scratch')
    })
    act(() => {
      result.current.selectRun(RUN_A as never)
    })
    expect(result.current.selectedRun?.run_id).toBe('run-1')
    expect(result.current.editingNotes).toBe(false)
    expect(result.current.notesValue).toBe('baseline notes')
    act(() => {
      result.current.closeSelection()
    })
    expect(result.current.selectedRun).toBeNull()
  })

  it('compares runs and clears the result on retarget', async () => {
    const { result } = await renderLoaded()
    act(() => {
      result.current.selectRun(RUN_A as never)
    })
    act(() => {
      result.current.changeCompareTarget('run-2')
    })
    expect(result.current.compareRunId).toBe('run-2')
    await act(async () => {
      await result.current.handleCompare()
    })
    expect(mockCompareTrainingRuns).toHaveBeenCalledWith('run-1', 'run-2')
    expect(result.current.compareResult).not.toBeNull()
    act(() => {
      result.current.changeCompareTarget('')
    })
    expect(result.current.compareRunId).toBe('')
    expect(result.current.compareResult).toBeNull()
  })

  it('bulk tag is a no-op with an empty tag or no selection', async () => {
    const { result } = await renderLoaded()
    await act(async () => {
      await result.current.handleBulkTag()
    })
    expect(mockBulkAddTag).not.toHaveBeenCalled()
    act(() => {
      result.current.toggleSelect('run-1')
      result.current.setBulkTag('   ')
    })
    await act(async () => {
      await result.current.handleBulkTag()
    })
    expect(mockBulkAddTag).not.toHaveBeenCalled()
    act(() => {
      result.current.setBulkTag('team-a')
    })
    await act(async () => {
      await result.current.handleBulkTag()
    })
    expect(mockBulkAddTag).toHaveBeenCalledWith(['run-1'], 'team-a')
  })
})
