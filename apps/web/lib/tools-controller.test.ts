/**
 * Tests for tools-controller — listTools and generateTool.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { listTools, generateTool } from './tools-controller'

vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn(),
  streamSSE: vi.fn(),
}))

import { apiGet, streamSSE } from '@/lib/http-client'

describe('listTools', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('calls GET /tools and returns tools', async () => {
    vi.mocked(apiGet).mockResolvedValue({
      tools: [{ id: 'writing', name: 'Writing Assistant' }],
    })

    const tools = await listTools()
    expect(apiGet).toHaveBeenCalledWith('/tools')
    expect(tools).toHaveLength(1)
    expect(tools[0].id).toBe('writing')
  })

  it('returns empty array when apiGet rejects', async () => {
    vi.mocked(apiGet).mockRejectedValue(new Error('Network down'))
    const tools = await listTools()
    expect(tools).toEqual([])
  })

  it('returns empty array when apiGet returns no tools key', async () => {
    vi.mocked(apiGet).mockResolvedValue({})
    const tools = await listTools()
    expect(tools).toEqual([])
  })
})

describe('generateTool', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('streams tokens into accumulated onToken output', async () => {
    async function* mockGen() {
      yield { status: 'working', data: { token: 'Hello' } } as any
      yield { status: 'working', data: { token: ' world' } } as any
      yield { status: 'complete' } as any
    }
    vi.mocked(streamSSE).mockReturnValue(mockGen())

    const onToken = vi.fn()
    const onDone = vi.fn()
    const onError = vi.fn()

    await generateTool('writing', { text: 'hi' }, { onToken, onDone, onError })

    expect(streamSSE).toHaveBeenCalledWith('/tools/writing/generate', {
      body: { payload: { text: 'hi' } },
    })
    expect(onToken).toHaveBeenCalledTimes(2)
    expect(onToken).toHaveBeenNthCalledWith(1, 'Hello')
    expect(onToken).toHaveBeenNthCalledWith(2, 'Hello world')
    expect(onDone).toHaveBeenCalledTimes(1)
    expect(onError).not.toHaveBeenCalled()
  })

  it('passes max_tokens and temperature in body', async () => {
    async function* mockGen() {
      yield { status: 'complete' } as any
    }
    vi.mocked(streamSSE).mockReturnValue(mockGen())

    await generateTool(
      'translate',
      { text: 'bonjour' },
      {},
      { max_tokens: 300, temperature: 0.5 },
    )

    expect(streamSSE).toHaveBeenCalledWith('/tools/translate/generate', {
      body: { payload: { text: 'bonjour' }, max_tokens: 300, temperature: 0.5 },
    })
  })

  it('calls onError with event message for error events', async () => {
    async function* mockGen() {
      yield { status: 'error', message: 'No provider available' } as any
    }
    vi.mocked(streamSSE).mockReturnValue(mockGen())

    const onError = vi.fn()
    await generateTool('writing', {}, { onError })

    expect(onError).toHaveBeenCalledWith('No provider available')
  })

  it('prefers data.error when message is empty', async () => {
    async function* mockGen() {
      yield { status: 'error', message: '', data: { error: 'backend boom' } } as any
    }
    vi.mocked(streamSSE).mockReturnValue(mockGen())

    const onError = vi.fn()
    await generateTool('writing', {}, { onError })

    expect(onError).toHaveBeenCalledWith('backend boom')
  })

  it('falls back to generic message for bare error events', async () => {
    async function* mockGen() {
      yield { status: 'error' } as any
    }
    vi.mocked(streamSSE).mockReturnValue(mockGen())

    const onError = vi.fn()
    await generateTool('writing', {}, { onError })

    expect(onError).toHaveBeenCalledWith('Generation failed')
  })

  it('calls onDone when stream ends without complete event', async () => {
    async function* mockGen() {
      yield { status: 'working', data: { token: 'x' } } as any
    }
    vi.mocked(streamSSE).mockReturnValue(mockGen())

    const onDone = vi.fn()
    await generateTool('writing', {}, { onDone })

    expect(onDone).toHaveBeenCalledTimes(1)
  })

  it('catches generator throw and calls onError', async () => {
    async function* mockGen() {
      throw new Error('Network failure')
    }
    vi.mocked(streamSSE).mockReturnValue(mockGen())

    const onError = vi.fn()
    await generateTool('writing', {}, { onError })

    expect(onError).toHaveBeenCalledWith('Connection error: Network failure')
  })
})