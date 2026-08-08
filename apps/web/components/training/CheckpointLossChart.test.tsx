// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { CheckpointLossChart } from './CheckpointLossChart'
import type { Checkpoint } from '@/lib/souls-controller'

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  CartesianGrid: () => null,
  ReferenceDot: () => null,
}))

function mkCp(overrides: Partial<Checkpoint> = {}): Checkpoint {
  return { name: 'test', soul: 'test', ...overrides }
}

describe('CheckpointLossChart', () => {
  it('returns null for empty checkpoints', () => {
    const { container } = render(<CheckpointLossChart checkpoints={[]} />)
    expect(container.innerHTML).toBe('')
  })

  it('returns null for single checkpoint', () => {
    const { container } = render(<CheckpointLossChart checkpoints={[mkCp({ loss: 1.0 })]} />)
    expect(container.innerHTML).toBe('')
  })

  it('returns null when no checkpoints have loss', () => {
    const { container } = render(<CheckpointLossChart checkpoints={[mkCp(), mkCp({ name: 'b' })]} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders chart with 2+ checkpoints with loss', () => {
    render(
      <CheckpointLossChart
        checkpoints={[mkCp({ name: 'a', loss: 2.0 }), mkCp({ name: 'b', loss: 1.0 })]}
      />
    )
    expect(screen.getByText('Loss trend')).toBeDefined()
    expect(screen.getByTestId('line-chart')).toBeDefined()
  })

  it('shows checkpoint count and best loss', () => {
    render(
      <CheckpointLossChart
        checkpoints={[mkCp({ name: 'a', loss: 2.0 }), mkCp({ name: 'b', loss: 1.0 })]}
      />
    )
    expect(screen.getAllByText(/2 checkpoints/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/best.*1\.0000/).length).toBeGreaterThanOrEqual(1)
  })

  it('ignores zero/negative loss checkpoints', () => {
    const { container } = render(
      <CheckpointLossChart
        checkpoints={[mkCp({ name: 'a', loss: 0 }), mkCp({ name: 'b', loss: -1 })]}
      />
    )
    expect(container.innerHTML).toBe('')
  })
})
