import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useShell } from './useShell'
import type { ShellStreamCallbacks } from '@/lib/shell-controller'

vi.mock('@/lib/shell-controller', () => ({
  shellExec: vi.fn(),
  shellExecStream: vi.fn(),
}))

import { shellExecStream } from '@/lib/shell-controller'

beforeEach(() => { vi.clearAllMocks() })

function mockStream(events: Array<Record<string, unknown>>) {
  vi.mocked(shellExecStream).mockImplementation(
    async (_cmd: string, callbacks: ShellStreamCallbacks) => {
      for (const e of events) {
        const phase = e.phase as string | undefined
        const status = e.status as string | undefined
        const data = (e.data ?? {}) as Record<string, unknown>

        if (phase === 'STREAMING' && status === 'working' && data.line !== undefined) {
          callbacks.onLine?.(data.line as string, (data.index as number) ?? 0)
        } else if (phase === 'STREAMING' && status === 'complete') {
          callbacks.onComplete?.(
            (data.exit_code as number) ?? 1,
            (data.lines as number) ?? 0,
            ((e.meta as Record<string, unknown>)?.elapsed_ms as number) ?? 0,
          )
        } else if (phase === 'STREAMING' && status === 'error') {
          callbacks.onError?.((data.error as string) ?? 'error')
        } else if (status === 'error') {
          callbacks.onError?.((e.message as string) ?? 'error')
        }
      }
    },
  )
}

describe('useShell', () => {
  it('initial state has empty lines and not running', () => {
    const { result } = renderHook(() => useShell())
    expect(result.current.state.lines).toEqual([])
    expect(result.current.state.isRunning).toBe(false)
    expect(result.current.state.exitCode).toBeNull()
    expect(result.current.state.error).toBeNull()
  })

  it('execute processes stream events', async () => {
    mockStream([
      { stream: 'shell', phase: 'STREAMING', status: 'working', data: { line: 'hello', index: 0 }, message: '' },
      { stream: 'shell', phase: 'STREAMING', status: 'working', data: { line: 'world', index: 1 }, message: '' },
      { stream: 'shell', phase: 'STREAMING', status: 'complete', data: { exit_code: 0, lines: 2 }, meta: { elapsed_ms: 1.2 }, message: '' },
    ])

    const { result } = renderHook(() => useShell())

    await act(async () => {
      await result.current.execute('echo hello')
    })

    expect(result.current.state.lines).toHaveLength(2)
    expect(result.current.state.lines[0].text).toBe('hello')
    expect(result.current.state.lines[1].text).toBe('world')
    expect(result.current.state.isRunning).toBe(false)
    expect(result.current.state.exitCode).toBe(0)
  })

  it('execute with empty command does nothing', async () => {
    const { result } = renderHook(() => useShell())
    await act(async () => {
      await result.current.execute('')
    })
    expect(result.current.state.lines).toEqual([])
    expect(shellExecStream).not.toHaveBeenCalled()
  })

  it('execute with whitespace-only does nothing', async () => {
    const { result } = renderHook(() => useShell())
    await act(async () => {
      await result.current.execute('   ')
    })
    expect(shellExecStream).not.toHaveBeenCalled()
  })

  it('clear resets state', async () => {
    mockStream([
      { stream: 'shell', phase: 'STREAMING', status: 'working', data: { line: 'data', index: 0 }, message: '' },
      { stream: 'shell', phase: 'STREAMING', status: 'complete', data: { exit_code: 0, lines: 1 }, meta: { elapsed_ms: 0.5 }, message: '' },
    ])

    const { result } = renderHook(() => useShell())

    await act(async () => {
      await result.current.execute('something')
    })
    expect(result.current.state.lines.length).toBeGreaterThan(0)

    act(() => { result.current.clear() })

    expect(result.current.state.lines).toEqual([])
    expect(result.current.state.exitCode).toBeNull()
    expect(result.current.state.error).toBeNull()
    expect(result.current.state.isRunning).toBe(false)
  })

  it('cancel stops running', async () => {
    let resolveStream: (() => void) | null = null
    vi.mocked(shellExecStream).mockImplementation(
      async () => new Promise<void>(r => { resolveStream = r }),
    )

    const { result } = renderHook(() => useShell())

    act(() => { result.current.execute('long command') })
    expect(result.current.state.isRunning).toBe(true)

    act(() => { result.current.cancel() })
    expect(result.current.state.isRunning).toBe(false)
    resolveStream?.()
  })

  it('onError sets error state', async () => {
    mockStream([
      { status: 'error', message: 'Connection refused' },
    ])

    const { result } = renderHook(() => useShell())

    await act(async () => {
      await result.current.execute('fail')
    })

    expect(result.current.state.error).toBe('Connection refused')
    expect(result.current.state.isRunning).toBe(false)
  })

  it('maxLines caps the line count', async () => {
    const events: Array<Record<string, unknown>> = []
    for (let i = 0; i < 10; i++) {
      events.push({ stream: 'shell', phase: 'STREAMING', status: 'working', data: { line: `line-${i}`, index: i }, message: '' })
    }
    events.push({ stream: 'shell', phase: 'STREAMING', status: 'complete', data: { exit_code: 0, lines: 10 }, meta: { elapsed_ms: 1 }, message: '' })
    mockStream(events)

    const { result } = renderHook(() => useShell(5))

    await act(async () => {
      await result.current.execute('many lines')
    })

    expect(result.current.state.lines.length).toBeLessThanOrEqual(5)
  })
})
