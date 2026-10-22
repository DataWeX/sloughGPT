// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

import { ChartFrame, chartFrameState } from './ChartFrame'

describe('ChartFrame', () => {
  afterEach(cleanup)

  it('ready: labeled, focusable chart region with children', () => {
    render(
      <ChartFrame label="Loss chart" height={200}>
        <svg data-testid="chart-svg" />
      </ChartFrame>,
    )
    const frame = screen.getByTestId('chart-frame')
    expect(frame.getAttribute('data-state')).toBe('ready')
    const region = screen.getByRole('img')
    expect(region.getAttribute('aria-label')).toBe('Loss chart')
    expect(region.getAttribute('tabindex')).toBe('0')
    expect(screen.getByTestId('chart-svg')).toBeDefined()
  })

  it('loading: aria-busy skeleton, children withheld', () => {
    render(
      <ChartFrame label="Loss chart" loading>
        <svg data-testid="chart-svg" />
      </ChartFrame>,
    )
    expect(screen.getByTestId('chart-frame').getAttribute('data-state')).toBe('loading')
    const busy = document.querySelector('[aria-busy="true"]')
    expect(busy).not.toBeNull()
    expect(screen.queryByTestId('chart-svg')).toBeNull()
  })

  it('error: role=alert banner with message and retry wiring', () => {
    const onRetry = vi.fn()
    render(
      <ChartFrame label="Loss chart" error="fetch failed" onRetry={onRetry} />,
    )
    expect(screen.getByTestId('chart-frame').getAttribute('data-state')).toBe('error')
    const alert = screen.getByRole('alert')
    expect(alert.textContent).toContain('fetch failed')
    fireEvent.click(screen.getByLabelText('Retry'))
    expect(onRetry).toHaveBeenCalledTimes(1)
  })

  it('empty: EmptyCard message, children withheld', () => {
    render(
      <ChartFrame label="Loss chart" empty emptyMessage="No loss data yet">
        <svg data-testid="chart-svg" />
      </ChartFrame>,
    )
    expect(screen.getByTestId('chart-frame').getAttribute('data-state')).toBe('empty')
    expect(screen.getByText('No loss data yet')).toBeDefined()
    expect(screen.queryByTestId('chart-svg')).toBeNull()
  })

  it('state precedence: loading > error > empty', () => {
    expect(chartFrameState({ loading: true, error: 'x', empty: true })).toBe('loading')
    expect(chartFrameState({ loading: false, error: 'x', empty: true })).toBe('error')
    expect(chartFrameState({ loading: false, error: null, empty: true })).toBe('empty')
    expect(chartFrameState({ loading: false, error: null, empty: false })).toBe('ready')
  })

  it('renders header title + action slot', () => {
    render(
      <ChartFrame label="Loss chart" title="Progress" action={<button type="button">Export</button>}>
        <svg />
      </ChartFrame>,
    )
    expect(screen.getByText('Progress')).toBeDefined()
    expect(screen.getByText('Export')).toBeDefined()
  })

  it('renders sr-only summary outside the img region', () => {
    render(
      <ChartFrame label="Loss chart" srSummary="Loss over 40 steps">
        <svg />
      </ChartFrame>,
    )
    const sr = document.querySelector('.sr-only')
    expect(sr?.textContent).toBe('Loss over 40 steps')
    // The summary must NOT be swallowed by the role="img" region.
    expect(screen.getByRole('img').contains(sr!)).toBe(false)
  })
})
