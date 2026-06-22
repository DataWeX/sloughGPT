/**
 * @vitest-environment jsdom
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

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  localStorage.clear()
})

describe('useChatAgents', () => {
  it('returns default state', () => {
    const { result } = renderHook(() => useChatAgents())
    expect(result.current.agents).toEqual([])
    expect(result.current.currentAgent).toBeNull()
    expect(result.current.knowledgeCtx).toEqual({ showing: false, count: 0, context: '' })
  })

  it('handleSelectAgent sets current agent and saves to localStorage', () => {
    const { result } = renderHook(() => useChatAgents())
    act(() => result.current.handleSelectAgent({ id: 'custom', name: 'Custom', description: 'test', icon: 'brain' }))
    expect(result.current.currentAgent).toEqual({ id: 'custom', name: 'Custom', description: 'test', icon: 'brain' })
    expect(localStorage.getItem('man_current_agent')).toBe('custom')
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

  it('fetchInitialData uses saved agent from localStorage', async () => {
    localStorage.setItem('man_current_agent', 'researcher')
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
