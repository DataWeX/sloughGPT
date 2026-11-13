import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { AutoTrainCard } from './AutoTrainCard'

const status = {
  enabled: true,
  threshold: 10,
  interval_s: 300,
  pending_conversations: 3,
  total_trains: 5,
  last_train: '2026-08-06T10:00:00Z',
  last_loss: 0.1234,
  last_checkpoint: 'ckpt-1',
  session_count: 20,
  response_log_count: 400,
  captured_count: 55,
} as any

describe('AutoTrainCard', () => {
  afterEach(cleanup)

  it('renders "Running" when enabled', () => {
    render(<AutoTrainCard status={status} />)
    expect(screen.getByText('Running')).toBeDefined()
  })

  it('renders "Off" when disabled', () => {
    render(<AutoTrainCard status={{ ...status, enabled: false }} />)
    expect(screen.getByText('Off')).toBeDefined()
  })

  it('renders queue and total trains', () => {
    render(<AutoTrainCard status={status} />)
    expect(screen.getByText('3/10')).toBeDefined()
    expect(screen.getByText('5')).toBeDefined()
  })

  it('renders last loss with four decimals', () => {
    render(<AutoTrainCard status={status} />)
    expect(screen.getByText('0.1234')).toBeDefined()
  })

  it('renders placeholder when loss is null', () => {
    render(<AutoTrainCard status={{ ...status, last_loss: null }} />)
    expect(screen.getByText('...')).toBeDefined()
  })

  it('renders last train line with checkpoint', () => {
    render(<AutoTrainCard status={status} />)
    expect(screen.getByText(/ckpt-1/)).toBeDefined()
  })

  it('renders sessions, logs and interval line', () => {
    render(<AutoTrainCard status={status} />)
    expect(screen.getByText('20 sessions · 400 logs · 300s')).toBeDefined()
  })
})
