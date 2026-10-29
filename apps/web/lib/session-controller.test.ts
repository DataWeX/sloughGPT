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

import { sessionController } from './session-controller'

describe('sessionController.list', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /chat/sessions and returns sessions', async () => {
    const sessions = [{ id: 's1', name: 'Chat 1', created_at: '2026-01-01', updated_at: '2026-01-01', pinned: false, starred: false, message_count: 0 }]
    apiClient.apiGet.mockResolvedValue({ sessions })

    const result = await sessionController.list()
    expect(result).toHaveLength(1)
    expect(result[0].id).toBe('s1')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/chat/sessions')
  })

  it('returns empty array when no sessions field', async () => {
    apiClient.apiGet.mockResolvedValue({})
    const result = await sessionController.list()
    expect(result).toEqual([])
  })
})

describe('sessionController.getCurrent', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /chat/sessions/current', async () => {
    apiClient.apiGet.mockResolvedValue({ id: 's1', name: 'Current', created_at: '', updated_at: '', pinned: false, starred: false, message_count: 0 })

    const result = await sessionController.getCurrent()
    expect(result?.id).toBe('s1')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/chat/sessions/current')
  })

  it('returns null on error', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('404'))
    const result = await sessionController.getCurrent()
    expect(result).toBeNull()
  })
})

describe('sessionController.create', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /chat/sessions with name', async () => {
    apiClient.apiPost.mockResolvedValue({ id: 's2', name: 'New Chat', created_at: '', updated_at: '', pinned: false, starred: false, message_count: 0 })

    const result = await sessionController.create('New Chat')
    expect(result.id).toBe('s2')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/chat/sessions', { name: 'New Chat', session_id: undefined })
  })
})

describe('sessionController.update', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('PUTs to /chat/sessions/{id}', async () => {
    apiClient.apiPut.mockResolvedValue(undefined)

    await sessionController.update('s1', { name: 'Renamed', starred: true })
    expect(apiClient.apiPut).toHaveBeenCalledWith('/chat/sessions/s1', { name: 'Renamed', starred: true, pinned: undefined })
  })
})

describe('sessionController.delete', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('DELETEs /chat/sessions/{id}', async () => {
    apiClient.apiDelete.mockResolvedValue(undefined)

    await sessionController.delete('s1')
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/chat/sessions/s1')
  })
})

describe('sessionController.saveContext', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /session/{id}/context', async () => {
    apiClient.apiPost.mockResolvedValue(undefined)

    await sessionController.saveContext('s1', [{ role: 'user', content: 'hi' }])
    expect(apiClient.apiPost).toHaveBeenCalledWith('/session/s1/context', { messages: [{ role: 'user', content: 'hi' }] })
  })
})

describe('sessionController.fetchMessages', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /session/{id}/messages', async () => {
    apiClient.apiGet.mockResolvedValue({ messages: [{ role: 'user', content: 'hello' }] })

    const result = await sessionController.fetchMessages('s1')
    expect(result).toHaveLength(1)
    expect(result[0].content).toBe('hello')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/session/s1/messages')
  })

  it('returns empty array on missing messages', async () => {
    apiClient.apiGet.mockResolvedValue({})
    const result = await sessionController.fetchMessages('s1')
    expect(result).toEqual([])
  })

  it('forwards signal and silent opts to apiGet when provided', async () => {
    apiClient.apiGet.mockResolvedValue({ messages: [{ role: 'user', content: 'hello' }] })
    const controller = new AbortController()

    const result = await sessionController.fetchMessages('s1', { signal: controller.signal, silent: true })

    expect(result).toHaveLength(1)
    expect(apiClient.apiGet).toHaveBeenCalledWith(
      '/session/s1/messages',
      undefined,
      { signal: controller.signal, silent: true },
    )
  })
})
