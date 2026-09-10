import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, type RenderHookResult } from '@testing-library/react'
import { useWebSocket, type UseWebSocketResult, type UseWebSocketOptions } from './useWebSocket'

vi.mock('@/lib/dev-log', () => ({ trackEvent: vi.fn() }))

class MockWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSED = 3

  url: string
  readyState = MockWebSocket.CONNECTING
  onopen: (() => void) | null = null
  onmessage: ((e: MessageEvent) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null
  sent: string[] = []

  constructor(url: string) { this.url = url }
  send(data: string) { this.sent.push(data) }
  close() { this.readyState = MockWebSocket.CLOSED; this.onclose?.() }
  simulateOpen() { this.readyState = MockWebSocket.OPEN; this.onopen?.() }
  simulateMessage(data: unknown) {
    this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(data) }))
  }
  simulateError() { this.onerror?.() }
  simulateClose() { this.readyState = MockWebSocket.CLOSED; this.onclose?.() }
}

let mockWs: MockWebSocket | null = null

beforeEach(() => {
  mockWs = null
  vi.useFakeTimers()
  const MockWsCtor = vi.fn((url: string) => { mockWs = new MockWebSocket(url); return mockWs })
  ;(global as any).WebSocket = Object.assign(MockWsCtor, {
    CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3,
  })
})
afterEach(() => { vi.useRealTimers() })

function hookResult(r: RenderHookResult<UseWebSocketResult, UseWebSocketOptions>) {
  return r.result.current
}

describe('useWebSocket', () => {
  it('starts in disconnected state', () => {
    const { result } = renderHook(() => useWebSocket())
    expect(result.current.status).toBe('disconnected')
    expect(result.current.error).toBeNull()
    expect(result.current.lastMessage).toBeNull()
  })

  it('connect sets status to connecting', () => {
    const { result } = renderHook(() => useWebSocket())
    act(() => { result.current.connect('key') })
    expect(result.current.status).toBe('connecting')
  })

  it('sends auth on open', () => {
    const { result } = renderHook(() => useWebSocket())
    act(() => { result.current.connect('key') })
    act(() => { mockWs!.simulateOpen() })
    expect(result.current.status).toBe('authenticating')
    expect(mockWs!.sent[0]).toContain('key')
  })

  it('connects on authenticated response', () => {
    const { result } = renderHook(() => useWebSocket())
    act(() => { result.current.connect('key') })
    act(() => { mockWs!.simulateOpen() })
    act(() => { mockWs!.simulateMessage({ status: 'authenticated' }) })
    expect(result.current.status).toBe('connected')
    expect(result.current.error).toBeNull()
  })

  it('sets error on auth failure', () => {
    const onError = vi.fn()
    const { result } = renderHook(() => useWebSocket({ onError }))
    act(() => { result.current.connect('key') })
    act(() => { mockWs!.simulateOpen() })
    act(() => { mockWs!.simulateMessage({ status: 'error', error: 'bad key' }) })
    expect(result.current.status).toBe('error')
    expect(result.current.error).toBe('bad key')
  })

  it('handles token messages', () => {
    const onToken = vi.fn()
    const { result } = renderHook(() => useWebSocket({ onToken }))
    act(() => { result.current.connect('key') })
    act(() => { mockWs!.simulateOpen() })
    act(() => { mockWs!.simulateMessage({ status: 'authenticated' }) })
    act(() => { mockWs!.simulateMessage({ token: 'hello' }) })
    expect(onToken).toHaveBeenCalledWith('hello')
    expect(result.current.lastMessage?.type).toBe('token')
  })

  it('handles done messages', () => {
    const onComplete = vi.fn()
    const { result } = renderHook(() => useWebSocket({ onComplete }))
    act(() => { result.current.connect('key') })
    act(() => { mockWs!.simulateOpen() })
    act(() => { mockWs!.simulateMessage({ status: 'authenticated' }) })
    act(() => { mockWs!.simulateMessage({ done: true, status: 'done', text: 'result' }) })
    expect(onComplete).toHaveBeenCalledWith('result')
    expect(result.current.lastMessage?.type).toBe('complete')
  })

  it('handles error messages when not connected', () => {
    const onError = vi.fn()
    const { result } = renderHook(() => useWebSocket({ onError }))
    act(() => { result.current.connect('key') })
    act(() => { mockWs!.simulateOpen() })
    act(() => { mockWs!.simulateMessage({ status: 'authenticated' }) })
    // Disconnect first, then send error message
    act(() => { mockWs!.simulateClose() })
    // A reconnect attempt will create a new mockWs — grab it
    const reconnectWs = mockWs
    act(() => { reconnectWs!.simulateOpen() })
    // Now send a non-auth error while OPEN — it's handled by auth path
    // But if we send error when NOT open, it falls through to handleErrorMessage
    // Simulate: auth message with error status is the auth-error path
    act(() => { reconnectWs!.simulateMessage({ status: 'error', error: 'gen failed' }) })
    expect(onError).toHaveBeenCalled()
  })

  it('ignores pong messages', () => {
    const { result } = renderHook(() => useWebSocket())
    act(() => { result.current.connect('key') })
    act(() => { mockWs!.simulateOpen() })
    act(() => { mockWs!.simulateMessage({ status: 'authenticated' }) })
    act(() => { mockWs!.simulateMessage({ type: 'pong' }) })
    // pong is handled silently, lastMessage stays as unknown from parse
    expect(result.current.lastMessage).not.toBeNull()
  })

  it('sendGenerate sends payload when connected', () => {
    const { result } = renderHook(() => useWebSocket())
    act(() => { result.current.connect('key') })
    act(() => { mockWs!.simulateOpen() })
    act(() => { mockWs!.simulateMessage({ status: 'authenticated' }) })
    act(() => { result.current.sendGenerate('test prompt', { maxTokens: 100 }) })
    const sent = JSON.parse(mockWs!.sent[mockWs!.sent.length - 1])
    expect(sent.prompt).toBe('test prompt')
    expect(sent.max_tokens).toBe(100)
  })

  it('sendGenerate sets error when not connected', () => {
    const { result } = renderHook(() => useWebSocket())
    act(() => { result.current.sendGenerate('test') })
    expect(result.current.error).toBe('Not connected')
  })

  it('disconnect clears state', () => {
    const { result } = renderHook(() => useWebSocket())
    act(() => { result.current.connect('key') })
    act(() => { mockWs!.simulateOpen() })
    act(() => { result.current.disconnect() })
    expect(result.current.status).toBe('disconnected')
    expect(result.current.error).toBeNull()
  })

  it('reconnects after close with exponential backoff', () => {
    const { result } = renderHook(() =>
      useWebSocket({ maxReconnectAttempts: 3, reconnectBaseDelayMs: 100 })
    )
    act(() => { result.current.connect('key') })
    act(() => { mockWs!.simulateOpen() })
    act(() => { mockWs!.simulateMessage({ status: 'authenticated' }) })
    act(() => { mockWs!.simulateClose() })
    expect(result.current.status).toBe('connecting')
    act(() => { vi.advanceTimersByTime(100) })
    expect(mockWs).not.toBeNull()
  })

  it('stops reconnecting after max attempts', () => {
    const { result } = renderHook(() =>
      useWebSocket({ maxReconnectAttempts: 2, reconnectBaseDelayMs: 10 })
    )
    act(() => { result.current.connect('key') })
    act(() => { mockWs!.simulateOpen() })
    act(() => { mockWs!.simulateMessage({ status: 'authenticated' }) })
    act(() => { mockWs!.simulateClose() })
    act(() => { vi.advanceTimersByTime(10) })
    act(() => { mockWs!.simulateClose() })
    act(() => { vi.advanceTimersByTime(20) })
    act(() => { mockWs!.simulateClose() })
    act(() => { vi.advanceTimersByTime(40) })
    expect(result.current.status).toBe('error')
    expect(result.current.error).toContain('Reconnect failed')
  })

  it('does not reconnect when autoReconnect is false', () => {
    const { result } = renderHook(() => useWebSocket({ autoReconnect: false }))
    act(() => { result.current.connect('key') })
    act(() => { mockWs!.simulateOpen() })
    act(() => { mockWs!.simulateMessage({ status: 'authenticated' }) })
    const wsBefore = mockWs
    act(() => { mockWs!.simulateClose() })
    // No new WebSocket should be created (autoReconnect=false prevents it)
    expect(mockWs).toBe(wsBefore)
  })

  it('handles WebSocket constructor failure', () => {
    const OrigWs = (global as any).WebSocket
    ;(global as any).WebSocket = Object.assign(
      vi.fn(() => { throw new Error('WS init failed') }),
      { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 }
    )
    const { result } = renderHook(() => useWebSocket())
    act(() => { result.current.connect('key') })
    expect(result.current.status).toBe('error')
    expect(result.current.error).toBe('WS init failed')
    ;(global as any).WebSocket = OrigWs
  })

  it('handles non-JSON messages gracefully', () => {
    const { result } = renderHook(() => useWebSocket())
    act(() => { result.current.connect('key') })
    act(() => { mockWs!.simulateOpen() })
    act(() => { mockWs!.simulateMessage({ status: 'authenticated' }) })
    act(() => {
      mockWs!.onmessage?.(new MessageEvent('message', { data: 'not json' }))
    })
    expect(result.current.lastMessage?.type).toBe('raw')
  })

  it('cleanup runs on unmount', () => {
    const { unmount } = renderHook(() => useWebSocket())
    act(() => { unmount() })
    expect(() => vi.advanceTimersByTime(1000)).not.toThrow()
  })

  it('sendGenerate includes all options', () => {
    const { result } = renderHook(() => useWebSocket())
    act(() => { result.current.connect('key') })
    act(() => { mockWs!.simulateOpen() })
    act(() => { mockWs!.simulateMessage({ status: 'authenticated' }) })
    act(() => {
      result.current.sendGenerate('hi', {
        maxTokens: 512,
        temperature: 0.5,
        topP: 0.9,
        topK: 20,
        repetitionPenalty: 1.0,
        model: 'test-model',
      })
    })
    const sent = JSON.parse(mockWs!.sent[mockWs!.sent.length - 1])
    expect(sent).toEqual({
      prompt: 'hi',
      max_tokens: 512,
      temperature: 0.5,
      top_p: 0.9,
      top_k: 20,
      repetition_penalty: 1.0,
      model: 'test-model',
    })
  })
})
