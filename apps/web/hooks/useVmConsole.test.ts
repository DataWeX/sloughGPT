// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, cleanup, act } from '@testing-library/react'
import { useVmConsole } from './useVmConsole'

const mockApiPost = vi.fn()
const mockApiDelete = vi.fn()
vi.mock('@/lib/http-client', () => ({
  apiPost: (...args: unknown[]) => mockApiPost(...args),
  apiDelete: (...args: unknown[]) => mockApiDelete(...args),
}))
vi.mock('@/lib/config', () => ({
  PUBLIC_API_URL: 'http://localhost:8000',
}))

class MockEventSource {
  static CLOSED = 2
  static instances: MockEventSource[] = []
  url: string
  onmessage: ((e: MessageEvent) => void) | null = null
  onerror: (() => void) | null = null
  readyState = 1
  closed = false
  constructor(url: string) {
    this.url = url
    MockEventSource.instances.push(this)
  }
  close() {
    this.closed = true
    this.readyState = MockEventSource.CLOSED
  }
}
;(globalThis as Record<string, unknown>).EventSource = MockEventSource

function emit(payload: unknown, index = 0) {
  const es = MockEventSource.instances[index]
  es.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent)
}

beforeEach(() => {
  vi.clearAllMocks()
  MockEventSource.instances = []
  mockApiPost.mockResolvedValue({ session_id: 'sid-1' })
  mockApiDelete.mockResolvedValue({ closed: true })
})

afterEach(() => cleanup())

describe('useVmConsole', () => {
  it('start creates a session and opens the SSE stream', async () => {
    const { result } = renderHook(() => useVmConsole())
    await act(async () => {
      await result.current.start()
    })
    expect(mockApiPost).toHaveBeenCalledWith('/vm/session', { role: 'user' })
    expect(result.current.sessionId).toBe('sid-1')
    expect(result.current.phase).toBe('connecting')
    expect(MockEventSource.instances).toHaveLength(1)
    expect(MockEventSource.instances[0].url).toBe('http://localhost:8000/vm/session/sid-1/stream')
  })

  it('start failure sets phase error and message', async () => {
    mockApiPost.mockRejectedValueOnce(new Error('server down'))
    const { result } = renderHook(() => useVmConsole())
    await act(async () => {
      await result.current.start()
    })
    expect(result.current.phase).toBe('error')
    expect(result.current.error).toBe('server down')
    expect(MockEventSource.instances).toHaveLength(0)
  })

  it('stream events update phase and append output', async () => {
    const { result } = renderHook(() => useVmConsole())
    await act(async () => {
      await result.current.start()
    })

    act(() => {
      emit({ stream: 'vm_console', phase: 'start', status: 'continue', data: {} })
    })
    expect(result.current.phase).toBe('live')

    act(() => {
      emit({
        stream: 'vm_console',
        phase: 'output',
        status: 'continue',
        data: { text: 'sloughvm> ' },
      })
      emit({
        stream: 'vm_console',
        phase: 'output',
        status: 'continue',
        data: { text: 'commands: help\n' },
      })
    })
    expect(result.current.output).toBe('sloughvm> commands: help\n')

    act(() => {
      emit({
        stream: 'vm_console',
        phase: 'status',
        status: 'complete',
        data: { status: 'complete' },
      })
    })
    expect(result.current.phase).toBe('complete')
    expect(MockEventSource.instances[0].closed).toBe(true)
  })

  it('ignores events from other streams', async () => {
    const { result } = renderHook(() => useVmConsole())
    await act(async () => {
      await result.current.start()
    })
    act(() => {
      emit({ stream: 'auto-train', phase: 'output', data: { text: 'nope' } })
    })
    expect(result.current.output).toBe('')
    expect(result.current.phase).toBe('connecting')
  })

  it('sendInput posts to the session input endpoint', async () => {
    const { result } = renderHook(() => useVmConsole())
    await act(async () => {
      await result.current.start()
    })
    await act(async () => {
      await result.current.sendInput('help\n')
    })
    expect(mockApiPost).toHaveBeenCalledWith('/vm/session/sid-1/input', {
      text: 'help\n',
    })
  })

  it('sendInput without a session is a no-op', async () => {
    const { result } = renderHook(() => useVmConsole())
    await act(async () => {
      await result.current.sendInput('x')
    })
    expect(mockApiPost).not.toHaveBeenCalled()
  })

  it('disconnect closes the stream and deletes the session', async () => {
    const { result } = renderHook(() => useVmConsole())
    await act(async () => {
      await result.current.start()
    })
    await act(async () => {
      await result.current.disconnect()
    })
    expect(MockEventSource.instances[0].closed).toBe(true)
    expect(mockApiDelete).toHaveBeenCalledWith('/vm/session/sid-1')
    expect(result.current.sessionId).toBeNull()
    expect(result.current.phase).toBe('closed')
  })
})
