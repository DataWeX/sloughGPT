import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./auth', () => ({
  useAuthStore: {
    getState: () => ({ token: null as string | null }),
  },
}))

vi.mock('./config', () => ({
  PUBLIC_API_URL: 'http://127.0.0.1:9',
}))

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import { feedbackConversationsController } from './feedback-conversations-controller'

describe('feedbackConversationsController.list', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('returns array of conversations', async () => {
    const conversations = [
      { id: 'c1', name: 'Chat 1', session_id: 's1', created_at: '2026-01-01', updated_at: '2026-01-02', pinned: false, starred: false, message_count: 5 },
      { id: 'c2', name: 'Chat 2', session_id: 's2', created_at: '2026-01-03', updated_at: '2026-01-04', pinned: true, starred: true, message_count: 10 },
    ]
    apiClient.apiGet.mockResolvedValue(conversations)

    const result = await feedbackConversationsController.list()
    expect(result).toHaveLength(2)
    expect(result[0].id).toBe('c1')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/feedback/conversations')
  })

  it('handles {conversations: [...]} wrapper', async () => {
    const conversations = [
      { id: 'c1', name: 'Chat 1', session_id: 's1', created_at: '2026-01-01', updated_at: '2026-01-02', pinned: false, starred: false, message_count: 5 },
    ]
    apiClient.apiGet.mockResolvedValue({ conversations })

    const result = await feedbackConversationsController.list()
    expect(result).toHaveLength(1)
    expect(result[0].id).toBe('c1')
  })

  it('returns [] on null/undefined', async () => {
    apiClient.apiGet.mockResolvedValue(null)

    const result = await feedbackConversationsController.list()
    expect(result).toEqual([])
  })

  it('returns conversations with all fields', async () => {
    const conversations = [
      { id: 'c1', name: 'Test', session_id: 's1', created_at: '2026-01-01', updated_at: '2026-01-02', pinned: true, starred: false, message_count: 3 },
    ]
    apiClient.apiGet.mockResolvedValue(conversations)

    const result = await feedbackConversationsController.list()
    expect(result[0]).toMatchObject({
      id: 'c1',
      name: 'Test',
      session_id: 's1',
      created_at: '2026-01-01',
      updated_at: '2026-01-02',
      pinned: true,
      starred: false,
      message_count: 3,
    })
  })
})

describe('feedbackConversationsController.create', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('calls apiPost with correct args', async () => {
    const conversation = { id: 'c1', name: 'New Chat', session_id: 's1', created_at: '2026-01-01', updated_at: '2026-01-01', pinned: false, starred: false, message_count: 0 }
    apiClient.apiPost.mockResolvedValue(conversation)

    const result = await feedbackConversationsController.create('New Chat')
    expect(result.id).toBe('c1')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/feedback/conversations', { name: 'New Chat' })
  })

  it('returns created conversation', async () => {
    const conversation = { id: 'c2', name: 'Created', session_id: 's2', created_at: '2026-01-01', updated_at: '2026-01-01', pinned: false, starred: false, message_count: 0 }
    apiClient.apiPost.mockResolvedValue(conversation)

    const result = await feedbackConversationsController.create('Created')
    expect(result.name).toBe('Created')
  })
})

describe('feedbackConversationsController.delete', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('calls apiDelete with correct path', async () => {
    apiClient.apiDelete.mockResolvedValue(undefined)

    await feedbackConversationsController.delete('c1')
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/feedback/conversations/c1')
  })
})

describe('feedbackConversationsController.togglePin', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('calls apiPatch with {pinned: true}', async () => {
    const conversation = { id: 'c1', name: 'Chat', session_id: 's1', created_at: '2026-01-01', updated_at: '2026-01-02', pinned: true, starred: false, message_count: 5 }
    apiClient.apiPatch.mockResolvedValue(conversation)

    const result = await feedbackConversationsController.togglePin('c1', true)
    expect(result.pinned).toBe(true)
    expect(apiClient.apiPatch).toHaveBeenCalledWith('/feedback/conversations/c1', { pinned: true })
  })

  it('returns updated conversation', async () => {
    const conversation = { id: 'c1', name: 'Chat', session_id: 's1', created_at: '2026-01-01', updated_at: '2026-01-02', pinned: false, starred: false, message_count: 5 }
    apiClient.apiPatch.mockResolvedValue(conversation)

    const result = await feedbackConversationsController.togglePin('c1', false)
    expect(result.pinned).toBe(false)
  })
})

describe('feedbackConversationsController.toggleStar', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('calls apiPatch with {starred: false}', async () => {
    const conversation = { id: 'c1', name: 'Chat', session_id: 's1', created_at: '2026-01-01', updated_at: '2026-01-02', pinned: false, starred: false, message_count: 5 }
    apiClient.apiPatch.mockResolvedValue(conversation)

    const result = await feedbackConversationsController.toggleStar('c1', false)
    expect(result.starred).toBe(false)
    expect(apiClient.apiPatch).toHaveBeenCalledWith('/feedback/conversations/c1', { starred: false })
  })
})
