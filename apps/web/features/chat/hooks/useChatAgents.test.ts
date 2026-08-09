/**
 */
import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest'
import { renderHook, act, cleanup } from '@testing-library/react'
import { useChatAgents } from './useChatAgents'

const mockList = vi.fn()
vi.mock('@/lib/agents-controller', () => ({
  agentsController: { list: (...args: unknown[]) => mockList(...args) },
}))

vi.mock('@/lib/agents', () => ({
  AGENTS: {
    general: { id: 'general', name: 'General', description: 'Default assistant', icon: 'brain' },
    researcher: { id: 'researcher', name: 'Researcher', description: 'Deep research', icon: 'search' },
  },
}))

const { mockChatDB, kvStore } = vi.hoisted(() => {
  const kvStore = new Map<string, unknown>()
  const mockChatDB = {
    getKV: vi.fn(async (key: string) => kvStore.get(key)),
    setKV: vi.fn(async (key: string, value: unknown) => { kvStore.set(key, value) }),
    deleteKV: vi.fn(async (key: string) => { kvStore.delete(key) }),
  }
  return { mockChatDB, kvStore }
})

vi.mock('@/lib/db', () => ({
  chatDB: mockChatDB,
}))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  kvStore.clear()
  mockChatDB.getKV.mockImplementation(async (key: string) => kvStore.get(key))
  mockChatDB.setKV.mockImplementation(async (key: string, value: unknown) => { kvStore.set(key, value) })
  mockChatDB.deleteKV.mockImplementation(async (key: string) => { kvStore.delete(key) })
})

describe('useChatAgents', () => {
  it('returns default state', () => {
    const { result } = renderHook(() => useChatAgents())
    expect(result.current.agents).toEqual([])
    expect(result.current.currentAgent).toBeNull()
    expect(result.current.knowledgeCtx).toEqual({ showing: false, count: 0, context: '' })
  })

  it('handleSelectAgent sets current agent and saves to chatDB', async () => {
    const { result } = renderHook(() => useChatAgents())
    await act(async () => { result.current.handleSelectAgent({ id: 'custom', name: 'Custom', description: 'test', icon: 'brain', instructions: '' }) })
    expect(result.current.currentAgent).toEqual({ id: 'custom', name: 'Custom', description: 'test', icon: 'brain', instructions: '' })
    expect(mockChatDB.setKV).toHaveBeenCalledWith('man_current_agent', 'custom')
  })

  it('handleToggleKnowledge flips knowledgeCtx.showing', () => {
    const { result } = renderHook(() => useChatAgents())
    act(() => result.current.handleToggleKnowledge())
    expect(result.current.knowledgeCtx.showing).toBe(true)
    act(() => result.current.handleToggleKnowledge())
    expect(result.current.knowledgeCtx.showing).toBe(false)
  })

  it('fetchInitialData loads agents from controller', async () => {
    mockList.mockResolvedValue([{ id: 'a1', name: 'A1', description: 'desc', icon: 'robot' }])
    const { result } = renderHook(() => useChatAgents())
    await act(async () => { await result.current.fetchInitialData() })
    expect(result.current.agents).toEqual([{ id: 'a1', name: 'A1', description: 'desc', icon: 'robot' }])
  })

  it('fetchInitialData falls back to local AGENTS on controller error', async () => {
    mockList.mockRejectedValue(new Error('fail'))
    const { result } = renderHook(() => useChatAgents())
    await act(async () => { await result.current.fetchInitialData() })
    expect(result.current.agents).toHaveLength(2)
    expect(result.current.agents[0].id).toBe('general')
  })

  it('fetchInitialData uses saved agent from chatDB', async () => {
    mockChatDB.getKV.mockResolvedValueOnce('researcher')
    mockList.mockRejectedValue(new Error('fail'))
    const { result } = renderHook(() => useChatAgents())
    await act(async () => { await result.current.fetchInitialData() })
    expect(result.current.currentAgent?.id).toBe('researcher')
  })

  it('fetchInitialData defaults to general agent when nothing saved', async () => {
    mockList.mockRejectedValue(new Error('fail'))
    const { result } = renderHook(() => useChatAgents())
    await act(async () => { await result.current.fetchInitialData() })
    expect(result.current.currentAgent?.id).toBe('general')
  })
})
