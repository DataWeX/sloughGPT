// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
}))

import { ExperimentStatsCard } from './ExperimentStatsCard'

afterEach(() => cleanup())

describe('ExperimentStatsCard', () => {
  it('renders nothing when empty', () => {
    const { container } = render(<ExperimentStatsCard />)
    expect(container.firstChild).toBeNull()
  })

  it('shows metrics', () => {
    render(<ExperimentStatsCard metrics={[
      { metric: 'accuracy', value: 0.95, step: 1, timestamp: '2026-09-09' },
      { metric: 'loss', value: 0.05, step: 1, timestamp: '2026-09-09' },
    ]} />)
    expect(screen.getByText('Experiment Data')).toBeTruthy()
    expect(screen.getByText('Metrics')).toBeTruthy()
    expect(screen.getByText('0.9500')).toBeTruthy()
    expect(screen.getByText('0.0500')).toBeTruthy()
  })

  it('shows params', () => {
    render(<ExperimentStatsCard params={[
      { param: 'lr', value: '0.001', timestamp: '2026-09-09' },
      { param: 'batch_size', value: '32', timestamp: '2026-09-09' },
    ]} />)
    expect(screen.getByText('Parameters')).toBeTruthy()
    expect(screen.getByText('lr')).toBeTruthy()
    expect(screen.getByText('0.001')).toBeTruthy()
    expect(screen.getByText('batch_size')).toBeTruthy()
    expect(screen.getByText('32')).toBeTruthy()
  })

  it('shows status', () => {
    render(<ExperimentStatsCard status={{ status: 'completed', completed_at: '2026-09-09T12:00:00Z' }} />)
    expect(screen.getByText('completed')).toBeTruthy()
    expect(screen.getByText(/Completed/)).toBeTruthy()
  })

  it('shows metric entry counts', () => {
    render(<ExperimentStatsCard metrics={[
      { metric: 'acc', value: 0.9, step: 1, timestamp: '' },
      { metric: 'acc', value: 0.95, step: 2, timestamp: '' },
      { metric: 'acc', value: 0.97, step: 3, timestamp: '' },
    ]} />)
    expect(screen.getByText('3 entries')).toBeTruthy()
  })
})
