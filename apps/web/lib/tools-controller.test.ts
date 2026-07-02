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

import { toolsController } from './tools-controller'

describe('toolsController', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('list returns tools from /chat/tools', async () => {
    apiClient.apiGet.mockResolvedValue({
      tools: [
        { name: 'web_search', description: 'Search the web', parameters: [], requires_approval: false },
        { name: 'code_exec', description: 'Run code', parameters: [], requires_approval: true },
      ],
    })
    const result = await toolsController.list()
    expect(result.length).toBe(2)
    expect(result[0].name).toBe('web_search')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/chat/tools')
  })

  it('list returns empty array when no tools', async () => {
    apiClient.apiGet.mockResolvedValue({})
    const result = await toolsController.list()
    expect(result).toEqual([])
  })

  it('list handles error response', async () => {
    apiClient.apiGet.mockRejectedValue(new Error('API Error'))
    await expect(toolsController.list()).rejects.toThrow('API Error')
  })
})
