import { render, screen, cleanup } from '@testing-library/react'
import '@testing-library/jest-dom/vitest'
import { describe, it, expect, afterEach } from 'vitest'
import { ProgressBar } from './progress-bar'

afterEach(() => cleanup())

describe('ProgressBar', () => {
  it('renders a progressbar role', () => {
    render(<ProgressBar value={50} />)
    expect(screen.getByRole('progressbar')).toBeInTheDocument()
  })

  it('sets aria-valuemin and aria-valuemax', () => {
    render(<ProgressBar value={30} max={200} />)
    const bar = screen.getByRole('progressbar')
    expect(bar).toHaveAttribute('aria-valuemin', '0')
    expect(bar).toHaveAttribute('aria-valuemax', '200')
  })

  it('sets aria-valuenow to the value', () => {
    render(<ProgressBar value={42} />)
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '42')
  })

  it('sets aria-valuetext to percentage string', () => {
    render(<ProgressBar value={75} />)
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuetext', '75%')
  })

  it('removes aria-valuenow and aria-valuetext when indeterminate', () => {
    render(<ProgressBar indeterminate />)
    const bar = screen.getByRole('progressbar')
    expect(bar).not.toHaveAttribute('aria-valuenow')
    expect(bar).not.toHaveAttribute('aria-valuetext')
  })

  it('clamps visual width when value exceeds max', () => {
    const { container } = render(<ProgressBar value={150} max={100} />)
    const fill = container.querySelector('.rounded-full.bg-primary')
    expect(fill).toHaveStyle({ width: '100%' })
  })

  it('clamps visual width to 0 when value is negative', () => {
    const { container } = render(<ProgressBar value={-10} />)
    const fill = container.querySelector('.rounded-full.bg-primary')
    expect(fill).toHaveStyle({ width: '0%' })
  })

  it('shows label text when provided', () => {
    render(<ProgressBar value={60} label="Uploading..." />)
    expect(screen.getByText('Uploading...')).toBeInTheDocument()
  })

  it('hides label when not provided', () => {
    const { container } = render(<ProgressBar value={60} />)
    const labelRow = container.querySelector('.flex.justify-between')
    expect(labelRow).toBeNull()
  })

  it('shows percentage text when showValue is true', () => {
    render(<ProgressBar value={33} showValue />)
    expect(screen.getByText('33%')).toBeInTheDocument()
  })

  it('does not show percentage by default', () => {
    const { container } = render(<ProgressBar value={33} />)
    const labelRow = container.querySelector('.flex.justify-between')
    expect(labelRow).toBeNull()
  })

  it('applies default size class (h-2)', () => {
    const { container } = render(<ProgressBar value={50} />)
    const track = container.querySelector('[role="progressbar"]')
    expect(track).toHaveClass('h-2')
  })

  it('applies xs size class (h-1)', () => {
    const { container } = render(<ProgressBar value={50} size="xs" />)
    const track = container.querySelector('[role="progressbar"]')
    expect(track).toHaveClass('h-1')
  })

  it('applies lg size class (h-3)', () => {
    const { container } = render(<ProgressBar value={50} size="lg" />)
    const track = container.querySelector('[role="progressbar"]')
    expect(track).toHaveClass('h-3')
  })

  it('merges custom className on wrapper', () => {
    const { container } = render(<ProgressBar value={50} className="my-custom" />)
    expect(container.firstChild).toHaveClass('my-custom')
  })

  it('passes extra props to the track element', () => {
    render(<ProgressBar value={50} data-testid="my-bar" />)
    expect(screen.getByTestId('my-bar')).toBeInTheDocument()
  })

  it('defaults to max=100', () => {
    render(<ProgressBar value={50} />)
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuemax', '100')
  })

  it('defaults to value=0', () => {
    render(<ProgressBar />)
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '0')
  })
})
