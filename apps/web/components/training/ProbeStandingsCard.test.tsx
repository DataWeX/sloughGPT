// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children }: any) => <div data-testid="card">{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
}))

import { ProbeStandingsCard } from './ProbeStandingsCard'
import type { ProbeStanding, TrainedOnEntry } from '@/lib/training-controller'

const standings: ProbeStanding[] = [
  { stage: 0, dataset: 'alpha@v1', global_step: 100, probes: { 'alpha@v1': 2.5 } },
  {
    stage: 1,
    dataset: 'beta@v1',
    global_step: 200,
    probes: { 'alpha@v1': 2.7, 'beta@v1': 2.4 },
  },
]

const chain: TrainedOnEntry[] = [
  { ref: 'alpha@v1', stage: 0, global_step: 100 },
  { ref: 'beta@v1', stage: 1, global_step: 200 },
]

describe('ProbeStandingsCard', () => {
  afterEach(cleanup)

  it('renders the stage × probe matrix with forgetting deltas', () => {
    render(<ProbeStandingsCard standings={standings} trainedOn={chain} />)

    expect(screen.getByText('Probe standings')).toBeTruthy()
    expect(screen.getByText('1: alpha@v1')).toBeTruthy()
    expect(screen.getByText('2: beta@v1')).toBeTruthy()
    expect(screen.getByText('2.500', { exact: false })).toBeTruthy()
    expect(screen.getByText('2.700', { exact: false })).toBeTruthy()

    // alpha probe rose 2.500 → 2.700: forgetting highlight + in-cell arrow.
    expect(screen.getByText(/forgot \+0\.200/)).toBeTruthy()
    expect(screen.getByText('↑0.200')).toBeTruthy()

    // Trained-on chain caption.
    expect(screen.getByText(/trained on: alpha@v1 → beta@v1/)).toBeTruthy()
  })

  it('renders nothing for empty standings', () => {
    const { container } = render(<ProbeStandingsCard standings={[]} />)
    expect(container.firstChild).toBeNull()
  })

  it('omits drift chips when only one measurement exists', () => {
    render(
      <ProbeStandingsCard
        standings={[{ stage: 0, dataset: 'solo@v1', global_step: 5, probes: { 'solo@v1': 1.5 } }]}
      />,
    )
    expect(screen.getAllByText(/solo@v1/).length).toBeGreaterThan(0)
    expect(screen.queryByText(/forgot \+/)).toBeNull()
    expect(screen.queryByText(/trained on:/)).toBeNull()
  })
})
