import { describe, expect, it, beforeEach } from 'vitest'
import { useApiMonitor } from './api-monitor-store'

beforeEach(() => {
  useApiMonitor.setState({ status: 'connecting', lastOnline: null })
})

describe('useApiMonitor', () => {
  it('default status is connecting', () => {
    const { status } = useApiMonitor.getState()
    expect(status).toBe('connecting')
  })

  it('default lastOnline is null', () => {
    const { lastOnline } = useApiMonitor.getState()
    expect(lastOnline).toBeNull()
  })

  it('setStatus changes status', () => {
    useApiMonitor.getState().setStatus('offline')
    expect(useApiMonitor.getState().status).toBe('offline')
  })

  it('setStatus connected sets lastOnline', () => {
    const before = Date.now()
    useApiMonitor.getState().setStatus('connected')
    const { lastOnline } = useApiMonitor.getState()
    expect(lastOnline).not.toBeNull()
    expect(lastOnline!).toBeGreaterThanOrEqual(before)
  })

  it('setStatus offline does not change lastOnline', () => {
    useApiMonitor.getState().setStatus('connected')
    const onlineTs = useApiMonitor.getState().lastOnline
    useApiMonitor.getState().setStatus('offline')
    expect(useApiMonitor.getState().lastOnline).toBe(onlineTs)
  })

  it('setStatus connected updates lastOnline', async () => {
    useApiMonitor.getState().setStatus('connected')
    const ts1 = useApiMonitor.getState().lastOnline
    useApiMonitor.getState().setStatus('reloading')
    await new Promise((r) => setTimeout(r, 5))
    useApiMonitor.getState().setStatus('connected')
    const ts2 = useApiMonitor.getState().lastOnline
    expect(ts2).toBeGreaterThan(ts1!)
  })
})
