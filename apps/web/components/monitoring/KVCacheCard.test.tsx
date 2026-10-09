import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { KVCacheCard } from './KVCacheCard'

const kvSessions = {
  enabled: true,
  active_sessions: 3,
  max_sessions: 10,
  cached_tokens: 5120,
  ttl_seconds: 600,
  oldest_session_age: 42,
}

describe('KVCacheCard', () => {
  afterEach(cleanup)

  it('renders nothing when kv cache is disabled', () => {
    const { container } = render(<KVCacheCard kvSessions={{ enabled: false } as any} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders active session count', () => {
    render(<KVCacheCard kvSessions={kvSessions as any} />)
    expect(screen.getByText('3')).toBeDefined()
  })

  it('renders cached token count', () => {
    render(<KVCacheCard kvSessions={kvSessions as any} />)
    expect(screen.getByText('5120')).toBeDefined()
  })

  it('renders TTL in minutes', () => {
    render(<KVCacheCard kvSessions={kvSessions as any} />)
    expect(screen.getByText('10m')).toBeDefined()
  })

  it('renders oldest session age in seconds', () => {
    render(<KVCacheCard kvSessions={kvSessions as any} />)
    expect(screen.getByText('42s')).toBeDefined()
  })

  it('renders cross-turn reuse badge', () => {
    render(<KVCacheCard kvSessions={kvSessions as any} />)
    expect(screen.getByText('cross-turn reuse')).toBeDefined()
  })

  it('renders LRU cap when max sessions provided', () => {
    render(<KVCacheCard kvSessions={kvSessions as any} />)
    expect(screen.getByText(/LRU cap: 10 simultaneous sessions/)).toBeDefined()
  })

  it('shows skeleton age when oldest session age missing', () => {
    render(<KVCacheCard kvSessions={{ ...kvSessions, oldest_session_age: undefined } as any} />)
    const skeletons = document.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBeGreaterThanOrEqual(1)
  })
})
