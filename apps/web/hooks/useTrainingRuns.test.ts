import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'

const { mockFilter, mockDelete, mockDuplicate, mockBookmark, mockAddTag, mockRemoveTag, mockNotes, mockBulkDelete, mockBulkTag, mockBulkBookmark } = vi.hoisted(() => ({
  mockFilter: vi.fn(),
  mockDelete: vi.fn(),
  mockDuplicate: vi.fn(),
  mockBookmark: vi.fn(),
  mockAddTag: vi.fn(),
  mockRemoveTag: vi.fn(),
  mockNotes: vi.fn(),
  mockBulkDelete: vi.fn(),
  mockBulkTag: vi.fn(),
  mockBulkBookmark: vi.fn(),
}))

vi.mock('@/lib/settings-controller', () => ({
  settingsController: {
    filterTrainingRuns: mockFilter,
    deleteTrainingRun: mockDelete,
    duplicateTrainingRun: mockDuplicate,
    toggleBookmark: mockBookmark,
    addRunTag: mockAddTag,
    removeRunTag: mockRemoveTag,
    setRunNotes: mockNotes,
    bulkDeleteRuns: mockBulkDelete,
    bulkAddTag: mockBulkTag,
    bulkBookmark: mockBulkBookmark,
  },
}))

import { useTrainingRuns } from './useTrainingRuns'

const RUN = { run_id: 'r1', timestamp: 1, model: 'gpt2', quality_score: 0.9, bookmarked: false }

describe('useTrainingRuns', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockFilter.mockResolvedValue({ runs: [RUN] })
    mockAddTag.mockImplementation(async (_id: string, tag: string) => ({ ...RUN, tags: [tag] }))
    mockRemoveTag.mockImplementation(async () => ({ ...RUN, tags: [] }))
    mockNotes.mockImplementation(async (_id: string, notes: string) => ({ ...RUN, notes }))
    mockBookmark.mockImplementation(async () => ({ ...RUN, bookmarked: true }))
    mockDuplicate.mockImplementation(async () => ({ ...RUN, run_id: 'r2' }))
  })

  it('loads runs on mount', async () => {
    const { result } = renderHook(() => useTrainingRuns())
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.runs).toEqual([RUN])
  })

  it('deleteRun removes the run', async () => {
    const { result } = renderHook(() => useTrainingRuns())
    await waitFor(() => expect(result.current.loading).toBe(false))
    await result.current.deleteRun('r1')
    await waitFor(() => expect(result.current.runs).toEqual([]))
    expect(mockDelete).toHaveBeenCalledWith('r1')
  })

  it('duplicateRun prepends the copy', async () => {
    const { result } = renderHook(() => useTrainingRuns())
    await waitFor(() => expect(result.current.loading).toBe(false))
    await result.current.duplicateRun('r1')
    await waitFor(() => expect(result.current.runs[0].run_id).toBe('r2'))
  })

  it('toggleBookmark updates the run in place', async () => {
    const { result } = renderHook(() => useTrainingRuns())
    await waitFor(() => expect(result.current.loading).toBe(false))
    await result.current.toggleBookmark('r1')
    await waitFor(() => expect(result.current.runs[0].bookmarked).toBe(true))
  })

  it('addTag/removeTag/saveNotes patch the run', async () => {
    const { result } = renderHook(() => useTrainingRuns())
    await waitFor(() => expect(result.current.loading).toBe(false))
    const tagged = await result.current.addTag('r1', 'good')
    expect(tagged?.tags).toEqual(['good'])
    await result.current.removeTag('r1', 'good')
    const noted = await result.current.saveNotes('r1', 'hello')
    expect(noted?.notes).toBe('hello')
  })

  it('bulk ops call controller and refresh', async () => {
    const { result } = renderHook(() => useTrainingRuns())
    await waitFor(() => expect(result.current.loading).toBe(false))
    await result.current.bulkDelete(['r1'])
    expect(mockBulkDelete).toHaveBeenCalledWith(['r1'])
    await result.current.bulkTag(['r1'], 't')
    expect(mockBulkTag).toHaveBeenCalledWith(['r1'], 't')
    await result.current.bulkBookmark(['r1'], true)
    expect(mockBulkBookmark).toHaveBeenCalledWith(['r1'], true)
  })
})
