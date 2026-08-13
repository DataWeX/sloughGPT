import { cleanup, render, screen } from '@testing-library/react'
import { describe, expect, it, afterEach } from 'vitest'

import { Progress } from './progress'

afterEach(() => cleanup())

describe('Progress', () => {
  it('renders a progressbar with default aria values', () => {
    const { container } = render(<Progress />)
    const bar = container.querySelector<HTMLElement>('[role="progressbar"]')!
    expect(bar).toBeTruthy()
    expect(bar.getAttribute('aria-valuenow')).toBe('0')
    expect(bar.getAttribute('aria-valuemin')).toBe('0')
    expect(bar.getAttribute('aria-valuemax')).toBe('100')
  })

  it('sets aria-valuenow to the value', () => {
    const { container } = render(<Progress value={42} />)
    expect(container.querySelector<HTMLElement>('[role="progressbar"]')!.getAttribute('aria-valuenow')).toBe('42')
  })

  it('respects a custom max', () => {
    const { container } = render(<Progress value={50} max={200} />)
    expect(container.querySelector<HTMLElement>('[role="progressbar"]')!.getAttribute('aria-valuemax')).toBe('200')
  })

  it('clamps visual width above max', () => {
    const { container } = render(<Progress value={150} max={100} />)
    expect(container.querySelector<HTMLElement>('div.bg-primary')!.style.width).toBe('100%')
  })

  it('clamps visual width below zero', () => {
    const { container } = render(<Progress value={-10} />)
    expect(container.querySelector<HTMLElement>('div.bg-primary')!.style.width).toBe('0%')
  })

  it('renders fill width as a percentage', () => {
    const { container } = render(<Progress value={50} />)
    expect(container.querySelector<HTMLElement>('div.bg-primary')!.style.width).toBe('50%')
  })

  it('applies default fill variant', () => {
    const { container } = render(<Progress value={50} />)
    expect(container.querySelector<HTMLElement>('div.bg-primary')).toBeTruthy()
  })

  it('applies success fill variant', () => {
    const { container } = render(<Progress value={50} variant="success" />)
    expect(container.querySelector<HTMLElement>('div.bg-success')).toBeTruthy()
  })

  it('applies warning fill variant', () => {
    const { container } = render(<Progress value={50} variant="warning" />)
    expect(container.querySelector<HTMLElement>('div.bg-warning')).toBeTruthy()
  })

  it('applies error fill variant', () => {
    const { container } = render(<Progress value={50} variant="error" />)
    expect(container.querySelector<HTMLElement>('div.bg-destructive')).toBeTruthy()
  })

  it('applies xs size class', () => {
    const { container } = render(<Progress value={50} size="xs" />)
    expect(container.querySelector<HTMLElement>('[role="progressbar"]')!.classList.contains('h-1')).toBe(true)
  })

  it('applies sm size class', () => {
    const { container } = render(<Progress value={50} size="sm" />)
    expect(container.querySelector<HTMLElement>('[role="progressbar"]')!.classList.contains('h-1.5')).toBe(true)
  })

  it('applies default size class', () => {
    const { container } = render(<Progress value={50} />)
    expect(container.querySelector<HTMLElement>('[role="progressbar"]')!.classList.contains('h-2')).toBe(true)
  })

  it('applies lg size class', () => {
    const { container } = render(<Progress value={50} size="lg" />)
    expect(container.querySelector<HTMLElement>('[role="progressbar"]')!.classList.contains('h-3')).toBe(true)
  })

  it('removes aria values and sets aria-busy when indeterminate', () => {
    const { container } = render(<Progress indeterminate />)
    const bar = container.querySelector<HTMLElement>('[role="progressbar"]')!
    expect(bar.getAttribute('aria-valuenow')).toBeNull()
    expect(bar.getAttribute('aria-valuemax')).toBeNull()
    expect(bar.getAttribute('aria-busy')).toBe('true')
  })

  it('renders a sliding fill when indeterminate', () => {
    const { container } = render(<Progress indeterminate variant="success" />)
    const fill = container.querySelector<HTMLElement>('div.bg-success')!
    expect(fill.classList.contains('w-1/3')).toBe(true)
  })

  it('shows label text and sets aria-label', () => {
    const { container } = render(<Progress value={50} label="Uploading" />)
    expect(screen.getByText('Uploading')).toBeTruthy()
    expect(container.querySelector<HTMLElement>('[role="progressbar"]')!.getAttribute('aria-label')).toBe('Uploading')
  })

  it('shows percentage text when showValue is true', () => {
    render(<Progress value={33} showValue />)
    expect(screen.getByText('33%')).toBeTruthy()
  })

  it('hides the label row by default', () => {
    const { container } = render(<Progress value={33} />)
    expect(container.querySelector('div.flex.justify-between')).toBeNull()
  })

  it('merges custom className on the wrapper', () => {
    const { container } = render(<Progress value={50} className="my-progress" />)
    expect((container.firstChild as HTMLElement).classList.contains('my-progress')).toBe(true)
  })
})
