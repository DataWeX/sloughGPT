/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { SectionHeader, SectionList, SectionBox, SectionScroll } from './section'

afterEach(cleanup)

describe('SectionHeader', () => {
  it('renders title text', () => {
    render(<SectionHeader title="General" />)
    expect(screen.getByText('General')).toBeInTheDocument()
  })

  it('has uppercase tracking class', () => {
    const { container } = render(<SectionHeader title="Test" />)
    expect(container.firstChild).toHaveClass('uppercase')
    expect(container.firstChild).toHaveClass('tracking-wider')
  })

  it('applies custom className', () => {
    const { container } = render(<SectionHeader title="Test" className="my-2" />)
    expect(container.firstChild).toHaveClass('my-2')
  })
})

describe('SectionList', () => {
  it('renders children', () => {
    render(<SectionList><span data-testid="child" /></SectionList>)
    expect(screen.getByTestId('child')).toBeInTheDocument()
  })

  it('has space-y-1 class', () => {
    const { container } = render(<SectionList><div /></SectionList>)
    expect(container.firstChild).toHaveClass('space-y-1')
  })

  it('applies custom className', () => {
    const { container } = render(<SectionList className="mt-2"><div /></SectionList>)
    expect(container.firstChild).toHaveClass('mt-2')
  })
})

describe('SectionBox', () => {
  it('renders children', () => {
    render(<SectionBox><span data-testid="child" /></SectionBox>)
    expect(screen.getByTestId('child')).toBeInTheDocument()
  })

  it('has border and rounded classes', () => {
    const { container } = render(<SectionBox><div /></SectionBox>)
    const el = container.firstChild as HTMLElement
    expect(el.className).toContain('border')
    expect(el.className).toContain('rounded-lg')
  })

  it('applies custom className', () => {
    const { container } = render(<SectionBox className="p-4"><div /></SectionBox>)
    expect(container.firstChild).toHaveClass('p-4')
  })
})

describe('SectionScroll', () => {
  it('renders children', () => {
    render(<SectionScroll><span data-testid="child" /></SectionScroll>)
    expect(screen.getByTestId('child')).toBeInTheDocument()
  })

  it('has overflow-y-auto class', () => {
    const { container } = render(<SectionScroll><div /></SectionScroll>)
    expect(container.firstChild).toHaveClass('overflow-y-auto')
  })

  it('has h-full class', () => {
    const { container } = render(<SectionScroll><div /></SectionScroll>)
    expect(container.firstChild).toHaveClass('h-full')
  })

  it('applies custom className', () => {
    const { container } = render(<SectionScroll className="p-2"><div /></SectionScroll>)
    expect(container.firstChild).toHaveClass('p-2')
  })
})
