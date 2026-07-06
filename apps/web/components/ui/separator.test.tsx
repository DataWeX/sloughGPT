/**
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { Separator } from '@sloughgpt/strui'

afterEach(cleanup)

describe('Separator', () => {
  it('renders (no role when decorative)', () => {
    const { container } = render(<Separator />)
    expect(container.firstChild).toBeInTheDocument()
  })

  it('renders as horizontal by default', () => {
    const { container } = render(<Separator />)
    const el = container.firstChild as HTMLElement
    expect(el).toHaveAttribute('aria-orientation', 'horizontal')
    expect(el.className).toContain('h-px')
    expect(el.className).toContain('w-full')
  })

  it('renders as vertical when orientation vertical', () => {
    const { container } = render(<Separator orientation="vertical" />)
    const el = container.firstChild as HTMLElement
    expect(el).toHaveAttribute('aria-orientation', 'vertical')
    expect(el.className).toContain('w-px')
    expect(el.className).toContain('h-full')
  })

  it('is decorative by default', () => {
    const { container } = render(<Separator />)
    expect(container.firstChild).toHaveAttribute('aria-orientation')
  })

  it('applies custom className', () => {
    const { container } = render(<Separator className="my-4" />)
    expect(container.firstChild).toHaveClass('my-4')
  })

  it('has bg-border class', () => {
    const { container } = render(<Separator />)
    expect(container.firstChild).toHaveClass('bg-border')
  })

  it('forwards ref', () => {
    const ref = vi.fn()
    render(<Separator ref={ref} />)
    expect(ref).toHaveBeenCalledOnce()
  })
})
