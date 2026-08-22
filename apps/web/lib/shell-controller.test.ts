/**
 * Tests for shell controller — shellExec and shellExecStream.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { shellExec, shellExecStream } from './shell-controller'

// Mock http-client
vi.mock('@/lib/http-client', () => ({
  apiPost: vi.fn(),
  streamSSE: vi.fn(),
}))

import { apiPost, streamSSE } from '@/lib/http-client'

describe('shellExec', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('calls POST /shell/exec with command', async () => {
    vi.mocked(apiPost).mockResolvedValue({
      output: 'hello\n',
      exit_code: 0,
      elapsed_ms: 1.5,
    })

    const result = await shellExec('echo hello')

    expect(apiPost).toHaveBeenCalledWith('/shell/exec', {
      command: 'echo hello',
      timeout_ms: 30000,
    }, { signal: undefined })
    expect(result.output).toBe('hello\n')
    expect(result.exit_code).toBe(0)
  })

  it('passes custom timeout', async () => {
    vi.mocked(apiPost).mockResolvedValue({
      output: '',
      exit_code: 0,
      elapsed_ms: 0,
    })

    await shellExec('ls', 5000)

    expect(apiPost).toHaveBeenCalledWith('/shell/exec', {
      command: 'ls',
      timeout_ms: 5000,
    }, { signal: undefined })
  })
})

describe('shellExecStream', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('yields line events to onLine callback', async () => {
    const mockEvents = [
      { stream: 'shell', phase: 'STREAMING', status: 'working', data: { line: 'hello', index: 0 }, message: '' },
      { stream: 'shell', phase: 'STREAMING', status: 'working', data: { line: 'world', index: 1 }, message: '' },
      { stream: 'shell', phase: 'STREAMING', status: 'complete', data: { exit_code: 0, lines: 2 }, meta: { elapsed_ms: 1.5 }, message: '' },
    ]

    async function* mockGen() {
      for (const e of mockEvents) yield e as any
    }
    vi.mocked(streamSSE).mockReturnValue(mockGen())

    const onLine = vi.fn()
    const onComplete = vi.fn()

    await shellExecStream('echo hello', { onLine, onComplete })

    expect(onLine).toHaveBeenCalledTimes(2)
    expect(onLine).toHaveBeenCalledWith('hello', 0)
    expect(onLine).toHaveBeenCalledWith('world', 1)
    expect(onComplete).toHaveBeenCalledWith(0, 2, 1.5)
  })

  it('calls onError on error events', async () => {
    async function* mockGen() {
      yield { status: 'error', message: 'Connection failed' } as any
    }
    vi.mocked(streamSSE).mockReturnValue(mockGen())

    const onError = vi.fn()
    await shellExecStream('bad cmd', { onError })

    expect(onError).toHaveBeenCalledWith('Connection failed')
  })

  it('calls onError on STREAMING error phase', async () => {
    async function* mockGen() {
      yield {
        stream: 'shell', phase: 'STREAMING', status: 'error',
        data: { error: 'Command not found' }, message: 'Error: Command not found',
      } as any
    }
    vi.mocked(streamSSE).mockReturnValue(mockGen())

    const onError = vi.fn()
    await shellExecStream('nope', { onError })

    // The STREAMING/error phase is handled by the phase/status check
    expect(onError).toHaveBeenCalled()
  })

  it('handles generator throwing (connection error propagation)', async () => {
    async function* mockGen() {
      throw new Error('Network failure')
    }
    vi.mocked(streamSSE).mockReturnValue(mockGen())

    const onError = vi.fn()
    // Should not throw — onError should catch it
    await shellExecStream('cmd', { onError })

    expect(onError).toHaveBeenCalledWith('Network failure')
  })

  it('passes data.error from STREAMING error events', async () => {
    async function* mockGen() {
      yield {
        stream: 'shell', phase: 'STREAMING', status: 'error',
        data: { error: 'No such file or directory' }, message: '',
      } as any
    }
    vi.mocked(streamSSE).mockReturnValue(mockGen())

    const onError = vi.fn()
    await shellExecStream('bad cmd', { onError })

    expect(onError).toHaveBeenCalledWith('No such file or directory')
  })

  it('transport error with empty message falls back to Unknown error', async () => {
    async function* mockGen() {
      yield { status: 'error', message: '' } as any
    }
    vi.mocked(streamSSE).mockReturnValue(mockGen())

    const onError = vi.fn()
    await shellExecStream('cmd', { onError })

    expect(onError).toHaveBeenCalledWith('Unknown error')
  })

  it('falls back to console.error when onError is not provided', async () => {
    async function* mockGen() {
      yield { status: 'error', message: 'some error' } as any
    }
    vi.mocked(streamSSE).mockReturnValue(mockGen())

    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    await shellExecStream('cmd', {})

    expect(spy).toHaveBeenCalledWith('[shell]', 'some error')
    spy.mockRestore()
  })

  it('calls onComplete with exit code 1 when generator yields no events', async () => {
    async function* mockGen() {
      // empty generator — no events
    }
    vi.mocked(streamSSE).mockReturnValue(mockGen())

    const onComplete = vi.fn()
    await shellExecStream('cmd', { onComplete })

    expect(onComplete).toHaveBeenCalledWith(1, 0, 0)
  })

  it('does not call onComplete when onError already fired (pre-aborted signal)', async () => {
    async function* mockGen() {
      yield { status: 'error', message: 'aborted' } as any
    }
    vi.mocked(streamSSE).mockReturnValue(mockGen())

    const onError = vi.fn()
    const onComplete = vi.fn()
    await shellExecStream('cmd', { onError, onComplete })

    expect(onError).toHaveBeenCalledWith('aborted')
    expect(onComplete).not.toHaveBeenCalled()
  })
})
