import { describe, it, expect, vi, beforeEach } from 'vitest'
import { useErrorStore } from './error-store'
import { logStateEvent } from './state-events'

vi.mock('./dev-log', () => ({
  logger: { debug: vi.fn(), info: vi.fn(), warning: vi.fn(), error: vi.fn() },
  trackEvent: vi.fn(),
}))

describe('state-events', () => {
  beforeEach(() => {
    useErrorStore.getState().clearErrors()
    useErrorStore.getState().clearStateEvents()
    vi.clearAllMocks()
  })

  it('writes to the error-store buffer', () => {
    logStateEvent('connection_status_changed', { kind: 'connection', from: 'a', to: 'b' })
    const [e] = useErrorStore.getState().getStateEvents()
    expect(e.event).toBe('connection_status_changed')
    expect(e.from).toBe('a')
    expect(e.to).toBe('b')
  })

  it('forwards to trackEvent (backend ingest)', async () => {
    const { trackEvent } = await import('./dev-log')
    logStateEvent('startup_stage_changed', { kind: 'startup', from: 'init', to: 'ready' })
    expect(trackEvent).toHaveBeenCalledWith(
      'startup_stage_changed',
      expect.objectContaining({ from: 'init', to: 'ready' }),
    )
  })

  it('never throws when the store fails', () => {
    const spy = vi.spyOn(useErrorStore, 'getState').mockImplementation(() => {
      throw new Error('store down')
    })
    expect(() => logStateEvent('sse_open', { kind: 'sse' })).not.toThrow()
    spy.mockRestore()
  })
})
