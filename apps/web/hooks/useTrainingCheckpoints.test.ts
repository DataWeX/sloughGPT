import { renderHook, waitFor, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'

const { mockListCheckpoints, mockListBuilds, mockList, mockLoadCheckpoint, mockDeleteCheckpoint } = vi.hoisted(() => ({
  mockListCheckpoints: vi.fn(),
  mockListBuilds: vi.fn(),
  mockList: vi.fn(),
  mockLoadCheckpoint: vi.fn(),
  mockDeleteCheckpoint: vi.fn(),
}))

vi.mock('@/lib/controllers', () => ({
  trainingJobsController: {
    listCheckpoints: mockListCheckpoints,
    listBuilds: mockListBuilds,
    list: mockList,
    loadCheckpoint: mockLoadCheckpoint,
    deleteCheckpoint: mockDeleteCheckpoint,
  },
  modelController: { list: vi.fn() },
}))

import { useTrainingCheckpoints } from './useTrainingCheckpoints'

describe('useTrainingCheckpoints', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockListCheckpoints.mockResolvedValue([])
    mockListBuilds.mockResolvedValue([])
    mockList.mockResolvedValue([])
  })

  it('defaults loading to true', () => {
    const { result } = renderHook(() => useTrainingCheckpoints())
    expect(result.current.loadingCheckpoints).toBe(true)
    expect(result.current.loadingBuilds).toBe(true)
    expect(result.current.loadingJobs).toBe(true)
  })

  it('fetchCheckpoints sets checkpoints and loading to false', async () => {
    mockListCheckpoints.mockResolvedValue([{ name: 'cp1', soul: 'default' }])
    const { result } = renderHook(() => useTrainingCheckpoints())
    result.current.fetchCheckpoints()
    await waitFor(() => expect(result.current.loadingCheckpoints).toBe(false))
    expect(result.current.checkpoints).toEqual([{ name: 'cp1', soul: 'default' }])
  })

  it('fetchCheckpoints handles error gracefully', async () => {
    mockListCheckpoints.mockRejectedValue(new Error('fail'))
    const { result } = renderHook(() => useTrainingCheckpoints())
    result.current.fetchCheckpoints()
    await waitFor(() => expect(result.current.loadingCheckpoints).toBe(false))
    expect(result.current.checkpoints).toEqual([])
  })

  it('fetchBuilds sets builds', async () => {
    mockListBuilds.mockResolvedValue([{ id: 'b1' }])
    const { result } = renderHook(() => useTrainingCheckpoints())
    result.current.fetchBuilds()
    await waitFor(() => expect(result.current.loadingBuilds).toBe(false))
    expect(result.current.builds).toEqual([{ id: 'b1' }])
  })

  it('fetchJobs sets jobs', async () => {
    mockList.mockResolvedValue([{ id: 'j1' }])
    const { result } = renderHook(() => useTrainingCheckpoints())
    result.current.fetchJobs()
    await waitFor(() => expect(result.current.loadingJobs).toBe(false))
    expect(result.current.jobs).toEqual([{ id: 'j1' }])
  })

  it('handleLoadCheckpoint sets active and shows success toast', async () => {
    const addToast = vi.fn()
    mockLoadCheckpoint.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingCheckpoints())
    await result.current.handleLoadCheckpoint('cp1', addToast)
    expect(mockLoadCheckpoint).toHaveBeenCalledWith('cp1')
    await waitFor(() => expect(result.current.activeCheckpoint).toBe('cp1'))
    expect(addToast).toHaveBeenCalledWith('Loaded trained version: cp1', 'success')
  })

  it('handleLoadCheckpoint shows error toast on failure', async () => {
    const addToast = vi.fn()
    mockLoadCheckpoint.mockRejectedValue(new Error('fail'))
    const { result } = renderHook(() => useTrainingCheckpoints())
    await result.current.handleLoadCheckpoint('cp1', addToast)
    expect(addToast).toHaveBeenCalledWith('Failed to load trained version', 'error')
  })

  it('handleDeleteCheckpoint with confirm deletes and shows toast', async () => {
    window.confirm = vi.fn().mockReturnValue(true)
    const addToast = vi.fn()
    mockDeleteCheckpoint.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingCheckpoints())
    await waitFor(() => {
      act(() => result.current.setCheckpoints([{ name: 'cp1', soul: 'default' }, { name: 'cp2', soul: 'default' }]))
    })
    await result.current.handleDeleteCheckpoint('cp1', addToast)
    expect(mockDeleteCheckpoint).toHaveBeenCalledWith('cp1')
    await waitFor(() => expect(result.current.checkpoints).toEqual([{ name: 'cp2', soul: 'default' }]))
    expect(addToast).toHaveBeenCalledWith('Deleted cp1', 'success')
  })

  it('handleDeleteCheckpoint clears activeCheckpoint if deleted', async () => {
    window.confirm = vi.fn().mockReturnValue(true)
    const addToast = vi.fn()
    mockDeleteCheckpoint.mockResolvedValue(undefined)
    const { result } = renderHook(() => useTrainingCheckpoints())
    act(() => result.current.setActiveCheckpoint('cp1'))
    await act(async () => {
      await result.current.handleDeleteCheckpoint('cp1', addToast)
    })
    expect(result.current.activeCheckpoint).toBeNull()
  })

  it('handleDeleteCheckpoint skips if confirm false', async () => {
    window.confirm = vi.fn().mockReturnValue(false)
    const addToast = vi.fn()
    const { result } = renderHook(() => useTrainingCheckpoints())
    await result.current.handleDeleteCheckpoint('cp1', addToast)
    expect(mockDeleteCheckpoint).not.toHaveBeenCalled()
  })

  it('activeCheckpoint defaults to null', () => {
    const { result } = renderHook(() => useTrainingCheckpoints())
    expect(result.current.activeCheckpoint).toBeNull()
  })
})
