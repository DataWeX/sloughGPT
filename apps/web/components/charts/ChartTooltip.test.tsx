// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

import { ChartTooltip } from './ChartTooltip'

describe('ChartTooltip', () => {
  afterEach(cleanup)

  it('renders nothing when inactive', () => {
    const { container } = render(<ChartTooltip active={false} payload={[{ name: 'Train', value: 0.5 }]} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders nothing when payload is empty', () => {
    const { container } = render(<ChartTooltip active payload={[]} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders label, series name and 4-decimal value with token swatch', () => {
    render(
      <ChartTooltip
        active
        label={12}
        labelFormatter={l => `Step ${l}`}
        payload={[{ name: 'Train loss', value: 0.12345678, color: 'rgb(var(--chart-1))' }]}
      />,
    )
    expect(screen.getByText('Step 12')).toBeDefined()
    expect(screen.getByText('Train loss')).toBeDefined()
    expect(screen.getByText('0.1235')).toBeDefined()
    const swatch = document.querySelector('[aria-hidden="true"]')!
    expect(swatch.getAttribute('style')).toContain('rgb(var(--chart-1))')
  })

  it('stringifies non-numeric values', () => {
    render(<ChartTooltip active payload={[{ name: 'Reward', value: 'n/a' }]} />)
    expect(screen.getByText('n/a')).toBeDefined()
  })

  it('applies valueFormatter when given', () => {
    render(
      <ChartTooltip
        active
        payload={[{ name: 'Score', value: 72.3456 }]}
        valueFormatter={v => `${Number(v).toFixed(0)}%`}
      />,
    )
    expect(screen.getByText('72%')).toBeDefined()
    expect(screen.queryByText('72.3456')).toBeNull()
  })
})
