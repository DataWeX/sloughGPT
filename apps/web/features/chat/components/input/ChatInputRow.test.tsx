import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ChatInputRow } from './ChatInputRow'

afterEach(cleanup)

vi.mock('./ChatInputAccessories', () => ({
  ChatInputAccessories: ({ onImage, disabled }: any) => (
    <button
      data-testid="mock-accessories"
      disabled={disabled}
      onClick={() => onImage?.('data:url')}
    >
      Accessories
    </button>
  ),
}))

vi.mock('./ChatInputField', () => ({
  ChatInputField: ({ value, onChange, onSend, disabled }: any) => (
    <input
      data-testid="mock-input"
      value={value}
      disabled={disabled}
      onChange={(e) => onChange?.(e.target.value)}
      onKeyDown={(e) => e.key === 'Enter' && onSend?.()}
    />
  ),
}))

vi.mock('./ChatSendButton', () => ({
  ChatSendButton: ({ onSend, onStop, loading, disabled, hasContent }: any) => (
    // Mirrors real behavior: enabled while loading (stop action),
    // enabled when hasContent, otherwise disabled.
    <button
      data-testid="mock-send"
      disabled={disabled || (!loading && !hasContent)}
      onClick={loading ? onStop : onSend}
    >
      Send
    </button>
  ),
}))

describe('ChatInputRow', () => {
  const defaultProps = {
    value: '',
    onChange: vi.fn(),
    onSend: vi.fn(),
    loading: false,
    disabled: false,
    placeholder: 'Message...',
    textareaRef: { current: null } as unknown as React.RefObject<HTMLTextAreaElement>,
    onImage: vi.fn(),
    onTranscript: vi.fn(),
    hasContent: false,
  }

  it('renders accessories, input field, and send button', () => {
    render(<ChatInputRow {...defaultProps} />)
    expect(screen.getByTestId('mock-accessories')).toBeDefined()
    expect(screen.getByTestId('mock-input')).toBeDefined()
    expect(screen.getByTestId('mock-send')).toBeDefined()
  })

  it('shows token estimate when value has text', () => {
    render(<ChatInputRow {...defaultProps} value="Hello" />)
    expect(screen.getByLabelText('Estimated 2 tokens')).toBeDefined()
  })

  it('does not show token estimate when empty', () => {
    const { container } = render(<ChatInputRow {...defaultProps} value="" />)
    expect(container.querySelector('[aria-label^="Estimated"]')).toBeNull()
  })

  it('shows warning color at 2000 chars', () => {
    render(<ChatInputRow {...defaultProps} value={'x'.repeat(2500)} />)
    const count = screen.getByLabelText('Estimated 2 tokens')
    expect(count.className).toContain('warning')
  })

  it('shows destructive color at 4000 chars', () => {
    render(<ChatInputRow {...defaultProps} value={'x'.repeat(4500)} />)
    const count = screen.getByLabelText('Estimated 2 tokens')
    expect(count.className).toContain('destructive')
  })

  it('passes loading to send button (stop action stays enabled)', () => {
    render(<ChatInputRow {...defaultProps} loading={true} value="Hi" />)
    expect(screen.getByTestId('mock-send')).toBeEnabled()
  })

  it('renders exactly ONE stop control when loading with onCancel (no duplicate)', () => {
    render(
      <ChatInputRow
        {...defaultProps}
        loading={true}
        value="Hi"
        onStop={vi.fn()}
        onCancel={vi.fn()}
      />,
    )
    // The old standalone "Cancel generation" button must be gone —
    // ChatSendButton is the single stop control.
    expect(screen.queryByLabelText('Cancel generation')).toBeNull()
    expect(screen.getAllByTestId('mock-send').length).toBe(1)
  })

  it('routes the fuller server-side cancel into the send/stop button', () => {
    const onStop = vi.fn()
    const onCancel = vi.fn()
    render(
      <ChatInputRow
        {...defaultProps}
        loading={true}
        value="Hi"
        onStop={onStop}
        onCancel={onCancel}
      />,
    )
    fireEvent.click(screen.getByTestId('mock-send'))
    // onCancel = cancelStream + abort (superset of onStop) — it owns the stop slot.
    expect(onCancel).toHaveBeenCalledTimes(1)
    expect(onStop).not.toHaveBeenCalled()
  })

  it('falls back to local onStop when no onCancel is provided', () => {
    const onStop = vi.fn()
    render(<ChatInputRow {...defaultProps} loading={true} value="Hi" onStop={onStop} />)
    fireEvent.click(screen.getByTestId('mock-send'))
    expect(onStop).toHaveBeenCalledTimes(1)
  })

  it('has aria-label on the group', () => {
    render(<ChatInputRow {...defaultProps} />)
    expect(screen.getByRole('group', { name: 'Message composition' })).toBeDefined()
  })
})
