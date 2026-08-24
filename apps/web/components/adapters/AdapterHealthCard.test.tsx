// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { AdapterHealthCard } from './AdapterHealthCard'
import type { UserAdapterInfo } from '@/lib/user-adapters-controller'

afterEach(() => { cleanup() })

function makeAdapter(overrides: Partial<UserAdapterInfo> = {}): UserAdapterInfo {
  return {
    user_id: 'user1',
    feedback_count: 5,
    rank: 1,
    alpha: 1.0,
    model_dim: 64,
    created_at: new Date(Date.now() - 86400000 * 10).toISOString(),
    updated_at: new Date().toISOString(),
    ...overrides,
  }
}

const adapters = [
  makeAdapter({ user_id: 'user1', feedback_count: 10, rank: 1 }),
  makeAdapter({ user_id: 'user2', feedback_count: 5, rank: 0 }),
  makeAdapter({ user_id: 'user3', feedback_count: 3, rank: 1 }),
]

describe('AdapterHealthCard', () => {
  it('renders empty state for empty adapters', () => {
    render(<AdapterHealthCard adapters={[]} />)
    expect(screen.getAllByText('No adapters yet').length).toBeGreaterThanOrEqual(1)
  })

  it('renders health card', () => {
    render(<AdapterHealthCard adapters={adapters} />)
    expect(screen.getAllByTestId('adapter-health').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Adapter Health').length).toBeGreaterThanOrEqual(1)
  })

  it('shows adapter count', () => {
    render(<AdapterHealthCard adapters={adapters} />)
    expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1)
  })

  it('shows total feedback', () => {
    render(<AdapterHealthCard adapters={adapters} />)
    expect(screen.getAllByText('18').length).toBeGreaterThanOrEqual(1)
  })

  it('shows average feedback', () => {
    render(<AdapterHealthCard adapters={adapters} />)
    expect(screen.getAllByText('6.0').length).toBeGreaterThanOrEqual(1)
  })

  it('shows rank breakdown', () => {
    render(<AdapterHealthCard adapters={adapters} />)
    expect(screen.getAllByText(/Rank 1/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Rank 0/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows most active users', () => {
    render(<AdapterHealthCard adapters={adapters} />)
    expect(screen.getAllByText('Most Active').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('user1').length).toBeGreaterThanOrEqual(1)
  })

  it('shows average age', () => {
    render(<AdapterHealthCard adapters={adapters} />)
    expect(screen.getAllByText('Avg Age').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/d/).length).toBeGreaterThanOrEqual(1)
  })
})
