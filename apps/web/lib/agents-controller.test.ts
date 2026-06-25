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

describe('agentsController.orchestrate', () => {
  beforeEach(() => { vi.clearAllMocks() })

  function mockFetchSSE(events: string[]) {
    const encoder = new TextEncoder()
    const chunks = events.map(e => encoder.encode(`data: ${e}\n\n`))
    const stream = new ReadableStream({
      async start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(chunk)
        }
        controller.close()
      },
    })
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      body: stream,
      status: 200,
      headers: new Headers(),
    } as Response)
  }

  it('POSTs to /agents/orchestrate with goal and context', async () => {
    mockFetchSSE([
      JSON.stringify({ stream: 'agent-orchestrate', phase: 'PLAN', status: 'success', data: { tasks: [{ id: '1', description: 'research', agent: 'researcher', status: 'pending', result_preview: '', depends_on: [] }], task_count: 1 } }),
      JSON.stringify({ stream: 'agent-orchestrate', phase: 'EXECUTE', status: 'success', data: { task_id: '1', agent: 'researcher', description: 'research', result_preview: 'findings...' } }),
      JSON.stringify({ stream: 'agent-orchestrate', phase: 'COMPOSE', status: 'working' }),
      JSON.stringify({ stream: 'agent-orchestrate', phase: 'COMPLETE', status: 'complete', data: { response: 'final result', tasks: [] } }),
    ])

    const planFn = vi.fn()
    const statusFn = vi.fn()
    const composeFn = vi.fn()
    const completeFn = vi.fn()

    await agentsController.orchestrate('test goal', 'context', {
      onPlan: planFn,
      onTaskStatus: statusFn,
      onCompose: composeFn,
      onComplete: completeFn,
      onError: vi.fn(),
    })

    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:9/agents/orchestrate',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ goal: 'test goal', context: 'context' }),
      }),
    )
    expect(planFn).toHaveBeenCalledWith(expect.arrayContaining([expect.objectContaining({ id: '1' })]))
    expect(statusFn).toHaveBeenCalled()
    expect(composeFn).toHaveBeenCalled()
    expect(completeFn).toHaveBeenCalledWith('final result', [])
  })

  it('calls onError on HTTP error', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Server Error',
    } as Response)

    const errorFn = vi.fn()
    await agentsController.orchestrate('test', '', { onComplete: vi.fn(), onError: errorFn })
    expect(errorFn).toHaveBeenCalledWith('HTTP 500: Server Error')
  })

  it('calls onError on stream error event', async () => {
    mockFetchSSE([
      JSON.stringify({ stream: 'agent-orchestrate', phase: 'ERROR', status: 'error', data: { error: 'LLM failed' } }),
    ])

    const errorFn = vi.fn()
    await agentsController.orchestrate('test', '', { onComplete: vi.fn(), onError: errorFn })
    expect(errorFn).toHaveBeenCalledWith('LLM failed')
  })
})
