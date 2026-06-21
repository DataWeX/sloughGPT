import { describe, expect, it, beforeEach } from 'vitest'
import { useApiMonitor } from '../api-monitor-store'

describe('useApiMonitor', () => {
  beforeEach(() => { useApiMonitor.getState().setStatus('connecting') })

  it('starts in connecting state with null lastOnline', () => {
    const s = useApiMonitor.getState()
    expect(s.status).toBe('connecting')
    expect(s.lastOnline).toBeNull()
  })

  it('setStatus updates status', () => {
    useApiMonitor.getState().setStatus('connected')
    expect(useApiMonitor.getState().status).toBe('connected')
  })

  it('records lastOnline on connected', () => {
    const t0 = Date.now()
    useApiMonitor.getState().setStatus('connected')
    const s = useApiMonitor.getState()
    expect(s.lastOnline).toBeGreaterThanOrEqual(t0)
  })

  it('preserves lastOnline when transitioning away from connected', () => {
    useApiMonitor.getState().setStatus('connected')
    const t = useApiMonitor.getState().lastOnline
    useApiMonitor.getState().setStatus('offline')
    expect(useApiMonitor.getState().lastOnline).toBe(t)
  })

  it('cycles through all statuses', () => {
    useApiMonitor.getState().setStatus('connected')
    expect(useApiMonitor.getState().status).toBe('connected')
    useApiMonitor.getState().setStatus('offline')
    expect(useApiMonitor.getState().status).toBe('offline')
    useApiMonitor.getState().setStatus('reloading')
    expect(useApiMonitor.getState().status).toBe('reloading')
    useApiMonitor.getState().setStatus('connecting')
    expect(useApiMonitor.getState().status).toBe('connecting')
  })
})
