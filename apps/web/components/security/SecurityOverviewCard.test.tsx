// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { SecurityOverviewCard } from './SecurityOverviewCard'

afterEach(() => { cleanup() })

const authLog = { event_type: 'auth.login', timestamp: new Date(Date.now() - 300000).toISOString(), ip: '192.168.1.1', user: 'admin' }
const modelLog = { event_type: 'model.loaded', timestamp: new Date(Date.now() - 600000).toISOString(), ip: '10.0.0.1' }
const trainLog = { event_type: 'training.started', timestamp: new Date(Date.now() - 900000).toISOString() }
const deleteLog = { event_type: 'file.deleted', timestamp: new Date(Date.now() - 1200000).toISOString(), ip: '192.168.1.1' }
const uploadLog = { event_type: 'dataset.uploaded', timestamp: new Date(Date.now() - 1500000).toISOString(), user: 'admin' }

describe('SecurityOverviewCard', () => {
  it('renders with empty logs', () => {
    render(<SecurityOverviewCard logs={[]} apiKeyConfigured={false} apiKeyCount={0} />)
    expect(screen.getAllByTestId('security-overview').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('No audit events recorded.').length).toBeGreaterThanOrEqual(1)
  })

  it('shows API key configured status', () => {
    render(<SecurityOverviewCard logs={[]} apiKeyConfigured={true} apiKeyCount={3} />)
    expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1)
  })

  it('shows API key not configured', () => {
    render(<SecurityOverviewCard logs={[]} apiKeyConfigured={false} apiKeyCount={0} />)
    expect(screen.getAllByText('None').length).toBeGreaterThanOrEqual(1)
  })

  it('counts events, IPs, and users', () => {
    render(<SecurityOverviewCard logs={[authLog, modelLog, trainLog]} apiKeyConfigured={true} apiKeyCount={1} />)
    expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1)
  })

  it('categorizes events correctly', () => {
    const { container } = render(<SecurityOverviewCard logs={[authLog, modelLog, { ...deleteLog, event_type: 'file.delete' }]} apiKeyConfigured={true} apiKeyCount={1} />)
    expect(screen.getAllByText(/Auth/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Model/).length).toBeGreaterThanOrEqual(1)
    const allText = container.textContent ?? ''
    expect(allText).toContain('Destructive')
  })

  it('shows recent activity feed', () => {
    render(<SecurityOverviewCard logs={[authLog, modelLog]} apiKeyConfigured={true} apiKeyCount={1} />)
    expect(screen.getAllByText('auth.login').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('model.loaded').length).toBeGreaterThanOrEqual(1)
  })

  it('limits recent feed to 5 items', () => {
    const logs = Array.from({ length: 10 }, (_, i) => ({
      event_type: `event_${i}`,
      timestamp: new Date(Date.now() - i * 60000).toISOString(),
    }))
    render(<SecurityOverviewCard logs={logs} apiKeyConfigured={true} apiKeyCount={1} />)
    expect(screen.getAllByText('event_0').length).toBeGreaterThanOrEqual(1)
    expect(screen.queryAllByText('event_9').length).toBe(0)
  })

  it('shows user when present', () => {
    render(<SecurityOverviewCard logs={[authLog]} apiKeyConfigured={true} apiKeyCount={1} />)
    expect(screen.getAllByText('@admin').length).toBeGreaterThanOrEqual(1)
  })
})
