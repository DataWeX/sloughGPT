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

import { agentsController } from './agents-controller'

describe('agentsController.list', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('GETs /agents', async () => {
    apiClient.apiGet.mockResolvedValue([{ id: 'a1', name: 'Agent 1', description: '', instructions: '', tools: [], avatar: '' }])

    const result = await agentsController.list()
    expect(result).toHaveLength(1)
    expect(result[0].id).toBe('a1')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/agents')
  })
})

describe('agentsController.create', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /agents with data', async () => {
    apiClient.apiPost.mockResolvedValue({ id: 'a2', name: 'New Agent', description: 'desc', instructions: '', tools: [], avatar: '' })

    const result = await agentsController.create({ name: 'New Agent', description: 'desc' })
    expect(result.id).toBe('a2')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/agents', { name: 'New Agent', description: 'desc', instructions: undefined, tools: undefined, avatar: undefined })
  })
})

describe('agentsController.update', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('PUTs to /agents/{id}', async () => {
    apiClient.apiPut.mockResolvedValue({ id: 'a1', name: 'Renamed', description: '', instructions: '', tools: [], avatar: '' })

    const result = await agentsController.update('a1', { name: 'Renamed' })
    expect(result.name).toBe('Renamed')
    expect(apiClient.apiPut).toHaveBeenCalledWith('/agents/a1', { name: 'Renamed', description: undefined, instructions: undefined, tools: undefined, avatar: undefined })
  })
})

describe('agentsController.delete', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('DELETEs /agents/{id}', async () => {
    apiClient.apiDelete.mockResolvedValue(undefined)

    await agentsController.delete('a1')
    expect(apiClient.apiDelete).toHaveBeenCalledWith('/agents/a1')
  })
})

describe('agentsController.execute', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('POSTs to /agents/{id}/execute', async () => {
    apiClient.apiPost.mockResolvedValue({ response: 'done', tools_used: [] })

    const result = await agentsController.execute('a1', 'hello')
    expect(result.response).toBe('done')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/agents/a1/execute', { request: 'hello', session_id: '' })
  })
})
