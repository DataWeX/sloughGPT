import { describe, expect, it } from 'vitest'
import type { ApiHealthSnapshot } from './useApiHealth'
import type { HealthStatus } from '@/lib/model-controller'

describe('useApiHealth exports', () => {
  it('allows null and "offline" snapshot states', () => {
    const snapshots: ApiHealthSnapshot[] = [null, 'offline']
    expect(snapshots[0]).toBeNull()
    expect(snapshots[1]).toBe('offline')
  })

  it('accepts a HealthStatus as a valid snapshot', () => {
    const status = { status: 'healthy', model_loaded: true, model_type: 'gpt2' } as HealthStatus
    const snapshot: ApiHealthSnapshot = status
    expect(snapshot).toEqual(status)
  })
})
