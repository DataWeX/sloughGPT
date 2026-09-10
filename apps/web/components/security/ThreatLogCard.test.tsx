// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
}))

vi.mock('@/lib/time-ago', () => ({
  timeAgo: () => '2m ago',
}))

import { ThreatLogCard } from './ThreatLogCard'

afterEach(() => { cleanup() })

const makeLog = (overrides: Partial<{ event_type: string; timestamp: string; detail: string }> = {}) => ({
  event_type: 'user.login',
  timestamp: new Date().toISOString(),
  ...overrides,
})

describe('ThreatLogCard', () => {
  it('renders with no logs', () => {
    render(<ThreatLogCard logs={[]} />)
    expect(screen.getByText('Threat Log')).toBeTruthy()
    expect(screen.getByText('No events recorded yet.')).toBeTruthy()
  })

  it('renders logs with severity badges', () => {
    const logs = [
      makeLog({ event_type: 'user.login' }),
      makeLog({ event_type: 'dataset.delete' }),
      makeLog({ event_type: 'model.load' }),
    ]
    render(<ThreatLogCard logs={logs} />)
    expect(screen.getByText('Threat Log')).toBeTruthy()
    expect(screen.getByText(/high/)).toBeTruthy()
    expect(screen.getByText(/low/)).toBeTruthy()
  })

  it('detects brute force anomaly', () => {
    const logs = Array.from({ length: 5 }, () =>
      makeLog({ event_type: 'auth.fail', detail: 'invalid password' })
    )
    render(<ThreatLogCard logs={logs} />)
    const btn = screen.getByRole('button', { name: /Anomalies/ })
    expect(btn).not.toHaveProperty('disabled', true)
    expect(btn.textContent).toContain('1')
  })

  it('detects bulk delete anomaly', () => {
    const logs = Array.from({ length: 6 }, () =>
      makeLog({ event_type: 'dataset.delete' })
    )
    render(<ThreatLogCard logs={logs} />)
    const btn = screen.getByRole('button', { name: /Anomalies/ })
    expect(btn).not.toHaveProperty('disabled', true)
    expect(btn.textContent).toContain('1')
  })

  it('toggles to anomalies view', () => {
    const logs = Array.from({ length: 5 }, () =>
      makeLog({ event_type: 'auth.fail', detail: 'fail' })
    )
    render(<ThreatLogCard logs={logs} />)
    const btn = screen.getByRole('button', { name: /Anomalies/ })
    fireEvent.click(btn)
    expect(screen.getByText(/brute force/i)).toBeTruthy()
  })

  it('expands to show all events when more than 10', () => {
    const logs = Array.from({ length: 15 }, (_, i) =>
      makeLog({ event_type: `event.${i}` })
    )
    render(<ThreatLogCard logs={logs} />)
    expect(screen.getByText(/Show all 15 events/)).toBeTruthy()
    fireEvent.click(screen.getByText(/Show all 15 events/))
    expect(screen.getByText(/Show less/)).toBeTruthy()
  })

  it('disables anomaly button when no anomalies', () => {
    const logs = [makeLog({ event_type: 'model.load' })]
    render(<ThreatLogCard logs={logs} />)
    const btn = screen.getByRole('button', { name: /Anomalies/ })
    expect(btn).toHaveProperty('disabled', true)
  })
})
