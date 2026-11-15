import { describe, it, expect, vi, beforeEach } from 'vitest'
import {
  useConversationScope,
  saveConversationScope,
  effectiveGeneration,
} from './conversation-scope-store'
import { sessionController } from './session-controller'

vi.mock('./session-controller', () => ({
  sessionController: { update: vi.fn(), list: vi.fn() },
}))

describe('conversation scope cache', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useConversationScope.setState({ scopes: {} })
  })

  it('seeds deltas from the session index and skips rows without one', () => {
    useConversationScope
      .getState()
      .seed([{ id: 'a', scope: { temperature: 0.2 } }, { id: 'b', scope: null }, { id: 'c' }])
    expect(useConversationScope.getState().scopes).toEqual({ a: { temperature: 0.2 } })
  })

  it('a local write survives a later seed — the PUT is newer than the index row', () => {
    useConversationScope.getState().setScope('a', { temperature: 0.9 })
    useConversationScope.getState().seed([{ id: 'a', scope: { temperature: 0.2 } }])
    expect(useConversationScope.getState().scopes.a.temperature).toBe(0.9)
  })

  it('saveConversationScope merges the patch over the cached delta and PUTs the whole delta', async () => {
    useConversationScope.getState().setScope('a', { temperature: 0.4, use_rag: false })
    vi.mocked(sessionController.update).mockResolvedValue(undefined)

    await saveConversationScope('a', { max_tokens: 64 })

    // The endpoint replaces the stored delta wholesale — the send must carry
    // every field the conversation already owned, not just the new one.
    expect(sessionController.update).toHaveBeenCalledWith('a', {
      scope: { temperature: 0.4, use_rag: false, max_tokens: 64 },
    })
    expect(useConversationScope.getState().scopes.a).toEqual({
      temperature: 0.4,
      use_rag: false,
      max_tokens: 64,
    })
  })

  it('a failed PUT leaves the cache claiming nothing the server did not take', async () => {
    useConversationScope.getState().setScope('a', { temperature: 0.4 })
    vi.mocked(sessionController.update).mockRejectedValue(new Error('boom'))

    await expect(saveConversationScope('a', { temperature: 1.5 })).rejects.toThrow('boom')
    expect(useConversationScope.getState().scopes.a.temperature).toBe(0.4)
  })

  it('effectiveGeneration resolves per field: delta wins, absent field falls back', () => {
    useConversationScope.getState().setScope('a', { temperature: 0.1 })

    expect(effectiveGeneration('a', { temperature: 0.7, maxTokens: 256 })).toEqual({
      temperature: 0.1,
      maxTokens: 256,
    })
    expect(effectiveGeneration('unknown-session', { temperature: 0.7, maxTokens: 256 })).toEqual({
      temperature: 0.7,
      maxTokens: 256,
    })
  })
})
