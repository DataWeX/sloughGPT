/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { Progress } from './progress'

afterEach(cleanup)

describe('Progress', () => {
  it('renders with progressbar role', () => {
    render(<Progress value={50} />)
    expect(screen.getByRole('progressbar')).toBeInTheDocument()
  })

  it('sets aria-valuenow to value', () => {
    render(<Progress value={75} />)
    const el = screen.getByRole('progressbar')
    expect(el).toHaveAttribute('aria-valuenow', '75')
  })

  it('has aria-valuemin of 0', () => {
    render(<Progress value={50} />)
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuemin', '0')
  })

  it('has aria-valuemax of 100', () => {
    render(<Progress value={50} />)
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuemax', '100')
  })

  it('renders fill with correct width', () => {
    const { container } = render(<Progress value={60} />)
    const fill = container.querySelector('.h-full')
    expect(fill).toHaveStyle({ width: '60%' })
  })

  it('clamps value to 0 minimum', () => {
    const { container } = render(<Progress value={-10} />)
    const fill = container.querySelector('.h-full')
    expect(fill).toHaveStyle({ width: '0%' })
  })

  it('clamps value to 100 maximum', () => {
    const { container } = render(<Progress value={150} />)
    const fill = container.querySelector('.h-full')
    expect(fill).toHaveStyle({ width: '100%' })
  })

  it('defaults to 0 when no value', () => {
    const { container } = render(<Progress />)
    const fill = container.querySelector('.h-full')
    expect(fill).toHaveStyle({ width: '0%' })
  })

  it('has transition class on fill', () => {
    const { container } = render(<Progress value={50} />)
    const fill = container.querySelector('.h-full')
    expect(fill).toHaveClass('transition-all')
  })

  it('applies custom className', () => {
    const { container } = render(<Progress value={50} className="custom-track" />)
    const track = container.firstChild as HTMLElement
    expect(track).toHaveClass('custom-track')
  })
})
