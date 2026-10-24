import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Badge: ({ children, className }: any) => <span className={className}>{children}</span>,
}))

vi.mock('lucide-react', () => ({
  TrendingUp: ({ className }: any) => <span data-testid="trend-up" className={className} />,
  TrendingDown: ({ className }: any) => <span data-testid="trend-down" className={className} />,
  Minus: ({ className }: any) => <span data-testid="trend-minus" className={className} />,
}))

import { QualityBadge, RunStatusBadge, TrendIcon, TrainingChartTooltip, formatRunDate } from './training-shared'

describe('training-shared', () => {
  afterEach(cleanup)

  it('formatRunDate formats unix seconds as short date', () => {
    expect(formatRunDate(0)).toBe('—')
    expect(formatRunDate(1767225600)).toMatch(/Dec|Jan/)
  })

  it('QualityBadge returns null for missing score', () => {
    const { container } = render(<QualityBadge />)
    expect(container.innerHTML).toBe('')
  })

  it('QualityBadge thresholds green/yellow/red', () => {
    const { rerender } = render(<QualityBadge score={0.9} />)
    expect(screen.getByText('90%').className).toMatch('green')
    rerender(<QualityBadge score={0.7} />)
    expect(screen.getByText('70%').className).toMatch('yellow')
    rerender(<QualityBadge score={0.3} />)
    expect(screen.getByText('30%').className).toMatch('red')
  })

  it('RunStatusBadge picks converged / early-stopped / completed', () => {
    const { rerender } = render(<RunStatusBadge converged />)
    expect(screen.getByText('converged')).toBeTruthy()
    rerender(<RunStatusBadge earlyStopped />)
    expect(screen.getByText('early stopped')).toBeTruthy()
    rerender(<RunStatusBadge />)
    expect(screen.getByText('completed')).toBeTruthy()
  })

  it('TrendIcon renders per trend', () => {
    const { rerender } = render(<TrendIcon trend="improving" />)
    expect(screen.getByTestId('trend-up')).toBeTruthy()
    rerender(<TrendIcon trend="declining" />)
    expect(screen.getByTestId('trend-down')).toBeTruthy()
    rerender(<TrendIcon trend="stable" />)
    expect(screen.getByTestId('trend-minus')).toBeTruthy()
  })

  it('TrainingChartTooltip returns null when inactive', () => {
    const { container } = render(<TrainingChartTooltip active={false} payload={[]} />)
    expect(container.innerHTML).toBe('')
  })

  it('TrainingChartTooltip renders entries with 4-decimal values', () => {
    render(<TrainingChartTooltip active payload={[{ name: 'Quality', value: 0.123456, color: 'red' }]} label="Run 1" />)
    expect(screen.getByText('Run 1')).toBeTruthy()
    expect(screen.getByText(/0\.1235/)).toBeTruthy()
  })
})
