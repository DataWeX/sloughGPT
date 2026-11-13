import { cleanup, fireEvent, render } from '@testing-library/react'
import { describe, expect, it, vi, afterEach } from 'vitest'

import { Textarea } from './textarea'

afterEach(() => cleanup())

describe('Textarea', () => {
  it('renders a textarea element', () => {
    const { container } = render(<Textarea />)
    expect(container.querySelector('textarea')).toBeTruthy()
  })

  it('passes placeholder prop', () => {
    const { container } = render(<Textarea placeholder="Enter text" />)
    expect(container.querySelector<HTMLTextAreaElement>('textarea')!.getAttribute('placeholder')).toBe('Enter text')
  })

  it('renders a controlled value', () => {
    const { container } = render(<Textarea value="Hello" onChange={() => {}} />)
    expect(container.querySelector<HTMLTextAreaElement>('textarea')!.value).toBe('Hello')
  })

  it('sets disabled when disabled', () => {
    const { container } = render(<Textarea disabled />)
    expect(container.querySelector<HTMLTextAreaElement>('textarea')!.getAttribute('disabled')).not.toBeNull()
  })

  it('applies error classes when error is true', () => {
    const { container } = render(<Textarea error />)
    expect(container.querySelector<HTMLTextAreaElement>('textarea')!.className).toContain('border-destructive')
  })

  it('uses border-input by default', () => {
    const { container } = render(<Textarea />)
    expect(container.querySelector<HTMLTextAreaElement>('textarea')!.className).toContain('border-input')
  })

  it('merges custom className', () => {
    const { container } = render(<Textarea className="my-textarea" />)
    expect(container.querySelector<HTMLTextAreaElement>('textarea')!.classList.contains('my-textarea')).toBe(true)
  })

  it('is vertically resizable by default', () => {
    const { container } = render(<Textarea />)
    expect(container.querySelector<HTMLTextAreaElement>('textarea')!.className).toContain('resize-y')
  })

  it('disables resize and hides overflow with autoResize', () => {
    const { container } = render(<Textarea autoResize />)
    const el = container.querySelector<HTMLTextAreaElement>('textarea')!
    expect(el.className).toContain('resize-none')
    expect(el.className).toContain('overflow-hidden')
  })

  it('applies a minHeight style', () => {
    const { container } = render(<Textarea />)
    expect(container.querySelector<HTMLTextAreaElement>('textarea')!.style.minHeight).toBe('80px')
  })

  it('caps autoResize height at maxRows', () => {
    const { container } = render(<Textarea autoResize maxRows={2} />)
    const el = container.querySelector<HTMLTextAreaElement>('textarea')!
    Object.defineProperty(el, 'scrollHeight', { configurable: true, value: 500 })
    fireEvent.input(el)
    expect(el.style.height).toBe('56px')
  })

  it('grows to scrollHeight when under the maxRows cap', () => {
    const { container } = render(<Textarea autoResize />)
    const el = container.querySelector<HTMLTextAreaElement>('textarea')!
    Object.defineProperty(el, 'scrollHeight', { configurable: true, value: 100 })
    fireEvent.input(el)
    expect(el.style.height).toBe('100px')
  })

  it('calls the onInput callback', () => {
    const onInput = vi.fn()
    const { container } = render(<Textarea onInput={onInput} />)
    fireEvent.input(container.querySelector<HTMLTextAreaElement>('textarea')!, { target: { value: 'x' } })
    expect(onInput).toHaveBeenCalledTimes(1)
  })
})
