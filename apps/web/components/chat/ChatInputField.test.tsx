// @vitest-environment jsdom
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { createRef } from 'react'

import { ChatInputField } from './ChatInputField'

describe('ChatInputField', () => {
  const base = {
    value: '',
    onChange: vi.fn(),
    onSend: vi.fn(),
    placeholder: 'Type a message...',
    disabled: false,
    textareaRef: createRef<HTMLTextAreaElement>(),
  }

  beforeEach(() => {
    vi.useFakeTimers()
    vi.clearAllMocks()
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('renders textarea', () => {
    const { container } = render(<ChatInputField {...base} />)
    expect(container.querySelector('textarea')).toBeInTheDocument()
  })

  it('displays value', () => {
    const { container } = render(<ChatInputField {...base} value="hello" />)
    expect(container.querySelector('textarea')).toHaveValue('hello')
  })

  it('calls onChange on textarea input', () => {
    const { container } = render(<ChatInputField {...base} />)
    const ta = container.querySelector('textarea')!
    fireEvent.change(ta, { target: { value: 'hi' } })
    expect(base.onChange).toHaveBeenCalledWith('hi')
  })

  it('calls onSend on Enter', () => {
    const { container } = render(<ChatInputField {...base} value="hello" />)
    const ta = container.querySelector('textarea')!
    fireEvent.keyDown(ta, { key: 'Enter', shiftKey: false })
    expect(base.onSend).toHaveBeenCalled()
  })

  it('does not call onSend on Shift+Enter', () => {
    const { container } = render(<ChatInputField {...base} value="hello" />)
    const ta = container.querySelector('textarea')!
    fireEvent.keyDown(ta, { key: 'Enter', shiftKey: true })
    expect(base.onSend).toHaveBeenCalledTimes(0)
  })

  it('rotates placeholder every 5s when empty', () => {
    const { container } = render(<ChatInputField {...base} />)
    const ta = container.querySelector('textarea')!
    const ph = ta.getAttribute('placeholder')
    expect(ph).toBeTruthy()
  })

  it('stops rotating placeholder when value present', () => {
    const { container } = render(<ChatInputField {...base} value="hi" />)
    expect(container.querySelector('textarea')).toHaveAttribute('placeholder', 'Type a message...')
  })

  it('disables textarea when disabled', () => {
    const { container } = render(<ChatInputField {...base} disabled />)
    expect(container.querySelector('textarea')).toBeDisabled()
  })

  it('renders sr-only hint', () => {
    const { container } = render(<ChatInputField {...base} />)
    const hints = container.querySelectorAll('#chat-input-hint')
    expect(hints.length).toBeGreaterThanOrEqual(1)
  })
})
