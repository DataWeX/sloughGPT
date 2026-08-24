// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { RegistryHealthCard } from './RegistryHealthCard'
import type { RegisteredModel, RegistryStats } from '@/lib/registry-controller'

beforeEach(() => { vi.clearAllMocks() })
afterEach(() => { cleanup() })

const loadedModel: RegisteredModel = {
  model_id: 'gpt2',
  status: 'loaded',
  registered_at: new Date(Date.now() - 3600000).toISOString(),
  metrics: { request_count: 42, failure_count: 0 },
}

const failedModel: RegisteredModel = {
  model_id: 'qwen',
  status: 'failed',
  registered_at: new Date(Date.now() - 7200000).toISOString(),
  metrics: { failure_count: 3 },
}

const idleModel: RegisteredModel = {
  model_id: 'smollm',
  status: 'idle',
  registered_at: new Date(Date.now() - 86400000).toISOString(),
}

const stats: RegistryStats = {
  total_models: 3,
  loaded_models: 1,
  failed_models: 1,
  circuit_breaker_open: false,
}

const statsOpen: RegistryStats = {
  total_models: 3,
  loaded_models: 1,
  failed_models: 1,
  circuit_breaker_open: true,
}

describe('RegistryHealthCard', () => {
  it('renders empty state when no models', () => {
    render(<RegistryHealthCard models={[]} stats={stats} />)
    expect(screen.getAllByText('No models registered').length).toBeGreaterThanOrEqual(1)
  })

  it('renders health card with models', () => {
    render(<RegistryHealthCard models={[loadedModel, failedModel]} stats={stats} />)
    expect(screen.getAllByTestId('registry-health').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Health').length).toBeGreaterThanOrEqual(1)
  })

  it('shows loaded and failed counts', () => {
    render(<RegistryHealthCard models={[loadedModel, failedModel, idleModel]} stats={stats} />)
    expect(screen.getAllByText('1 loaded').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('1 failed').length).toBeGreaterThanOrEqual(1)
  })

  it('shows circuit breaker open badge', () => {
    render(<RegistryHealthCard models={[loadedModel]} stats={statsOpen} />)
    expect(screen.getAllByText('Circuit Open').length).toBeGreaterThanOrEqual(1)
  })

  it('does not show circuit breaker badge when closed', () => {
    render(<RegistryHealthCard models={[loadedModel]} stats={stats} />)
    expect(screen.queryAllByText('Circuit Open').length).toBe(0)
  })

  it('shows model names and statuses', () => {
    render(<RegistryHealthCard models={[loadedModel, failedModel]} stats={stats} />)
    expect(screen.getAllByText('gpt2').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('qwen').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('loaded').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('failed').length).toBeGreaterThanOrEqual(1)
  })

  it('shows failure count for failed models', () => {
    render(<RegistryHealthCard models={[failedModel]} stats={stats} />)
    expect(screen.getAllByText('3 failures').length).toBeGreaterThanOrEqual(1)
  })

  it('shows request count when available', () => {
    render(<RegistryHealthCard models={[loadedModel]} stats={stats} />)
    expect(screen.getAllByText('42 reqs').length).toBeGreaterThanOrEqual(1)
  })

  it('shows registered time when no last health check', () => {
    render(<RegistryHealthCard models={[idleModel]} stats={stats} />)
    expect(screen.getAllByText(/ago/).length).toBeGreaterThanOrEqual(1)
  })
})
