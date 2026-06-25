import { beforeEach, describe, expect, it, vi } from 'vitest'

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import { errorController } from './error-controller'

describe('errorController', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('getRecent GETs /errors/recent with query params', async () => {
    apiClient.apiGet.mockResolvedValue({ errors: [], unread_count: 0, total: 0, offset: 0, limit: 50 })
    const result = await errorController.getRecent(50, 10)
    expect(result.unread_count).toBe(0)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/errors/recent', { limit: '50', offset: '10' })
  })

  it('getRecent uses default limit/offset', async () => {
    apiClient.apiGet.mockResolvedValue({ errors: [], unread_count: 0, total: 0, offset: 0, limit: 50 })
    await errorController.getRecent()
    expect(apiClient.apiGet).toHaveBeenCalledWith('/errors/recent', { limit: '50', offset: '0' })
  })

  it('report POSTs to /errors/log', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', logged: 1 })
    const result = await errorController.report('test error', 'web')
    expect(result.logged).toBe(1)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/errors/log', {
      errors: [
        expect.objectContaining({ message: 'test error', source: 'web' }),
      ],
    })
  })

  it('report includes extra fields', async () => {
    apiClient.apiPost.mockResolvedValue({ status: 'ok', logged: 1 })
    await errorController.report('crash', 'app', { stack: 'at line 1', url: 'http://example.com' })
    const body = apiClient.apiPost.mock.calls[0][1]
    expect(body.errors[0].stack).toBe('at line 1')
    expect(body.errors[0].url).toBe('http://example.com')
  })

  it('clear DELETEs /errors/clear', async () => {
    apiClient.apiDelete.mockResolvedValue({})
    await errorController.clear()
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/errors/clear')
  })

  it('getUnreadCount GETs /errors/unread', async () => {
    apiClient.apiGet.mockResolvedValue({ unread_count: 5 })
    const count = await errorController.getUnreadCount()
    expect(count).toBe(5)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/errors/unread')
  })
})
