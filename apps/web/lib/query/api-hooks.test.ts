import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, waitFor, cleanup } from '@testing-library/react'
import { useQueryStore, fetchQuery, isStale } from './client'

const mocks = vi.hoisted(() => ({
  mockListModels: vi.fn(),
  mockLoadModel: vi.fn(),
  mockListSouls: vi.fn(),
  mockGetCurrentSoul: vi.fn(),
  mockListCheckpoints: vi.fn(),
  mockSwitchSoul: vi.fn(),
}))

vi.mock('@/lib/model-controller', () => ({
  modelController: { list: mocks.mockListModels, load: mocks.mockLoadModel },
}))

vi.mock('@/lib/souls-controller', () => ({
  soulsController: {
    list: mocks.mockListSouls,
    getCurrent: mocks.mockGetCurrentSoul,
    listCheckpoints: mocks.mockListCheckpoints,
    switch: mocks.mockSwitchSoul,
  },
}))

import {
  useModels,
  useLoadModel,
  useSouls,
  useCurrentSoul,
  useCheckpoints,
  useSwitchSoul,
} from './api-hooks'

async function primeFresh(key: string) {
  await fetchQuery(key, () => Promise.resolve('base'), { staleTime: 60000 })
}

describe('api-hooks', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useQueryStore.setState({ cache: {}, fetchingKeys: new Set() })
  })

  afterEach(cleanup)

  it('useModels lists models', async () => {
    mocks.mockListModels.mockResolvedValue([{ id: 'a' }])
    const { result } = renderHook(() => useModels())
    await waitFor(() => expect(result.current.data).toEqual([{ id: 'a' }]))
    expect(mocks.mockListModels).toHaveBeenCalledTimes(1)
  })

  it('useLoadModel loads a model and invalidates the models query', async () => {
    await primeFresh('models')
    expect(isStale('models')).toBe(false)
    mocks.mockLoadModel.mockResolvedValue({ status: 'ok' })
    const { result } = renderHook(() => useLoadModel())
    await act(async () => {
      await result.current.mutateAsync('gpt2')
    })
    expect(mocks.mockLoadModel).toHaveBeenCalledWith('gpt2')
    expect(isStale('models')).toBe(true)
  })

  it('useSouls lists souls', async () => {
    mocks.mockListSouls.mockResolvedValue([{ name: 'warm' }])
    const { result } = renderHook(() => useSouls())
    await waitFor(() => expect(result.current.data).toEqual([{ name: 'warm' }]))
    expect(mocks.mockListSouls).toHaveBeenCalledTimes(1)
  })

  it('useCurrentSoul fetches the current soul', async () => {
    mocks.mockGetCurrentSoul.mockResolvedValue({ name: 'warm' })
    const { result } = renderHook(() => useCurrentSoul())
    await waitFor(() => expect(result.current.data).toEqual({ name: 'warm' }))
    expect(mocks.mockGetCurrentSoul).toHaveBeenCalledTimes(1)
  })

  it('useCheckpoints lists checkpoints', async () => {
    mocks.mockListCheckpoints.mockResolvedValue(['cp1'])
    const { result } = renderHook(() => useCheckpoints())
    await waitFor(() => expect(result.current.data).toEqual(['cp1']))
    expect(mocks.mockListCheckpoints).toHaveBeenCalledTimes(1)
  })

  it('useSwitchSoul switches and invalidates souls, current-soul and checkpoints', async () => {
    await primeFresh('souls')
    await primeFresh('current-soul')
    await primeFresh('checkpoints')
    mocks.mockSwitchSoul.mockResolvedValue({ ok: true })
    const { result } = renderHook(() => useSwitchSoul())
    await act(async () => {
      await result.current.mutateAsync({ name: 'warm', checkpointName: 'cp1' })
    })
    expect(mocks.mockSwitchSoul).toHaveBeenCalledWith('warm', 'cp1')
    expect(isStale('souls')).toBe(true)
    expect(isStale('current-soul')).toBe(true)
    expect(isStale('checkpoints')).toBe(true)
  })
})
