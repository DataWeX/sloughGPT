import { describe, it, expect } from 'vitest'
import { inferStateKind, isStateEventKind } from './event-kinds'

describe('event-kinds', () => {
  it('accepts the full kind catalog', () => {
    for (const kind of [
      'connection',
      'startup',
      'sse',
      'health',
      'overlay',
      'api',
      'auth',
      'chat',
      'training',
      'model',
      'system',
      'ui',
    ]) {
      expect(isStateEventKind(kind)).toBe(true)
    }
    expect(isStateEventKind('nope')).toBe(false)
    expect(isStateEventKind(undefined)).toBe(false)
  })

  it('classifies state transitions', () => {
    expect(inferStateKind('connection_status_changed')).toBe('connection')
    expect(inferStateKind('startup_stage_changed')).toBe('startup')
    expect(inferStateKind('sse_open')).toBe('sse')
    expect(inferStateKind('health_fallback_error')).toBe('health')
    expect(inferStateKind('overlay_shown')).toBe('overlay')
    expect(inferStateKind('api_connection_changed')).toBe('api')
  })

  it('classifies domain events', () => {
    expect(inferStateKind('auth_login')).toBe('auth')
    expect(inferStateKind('session_created')).toBe('chat')
    expect(inferStateKind('training_started')).toBe('training')
    expect(inferStateKind('model_loaded')).toBe('model')
    expect(inferStateKind('vm_booted')).toBe('system')
    expect(inferStateKind('route_changed')).toBe('ui')
  })

  it('defaults unknown events to system', () => {
    expect(inferStateKind('something_totally_new')).toBe('system')
  })
})
