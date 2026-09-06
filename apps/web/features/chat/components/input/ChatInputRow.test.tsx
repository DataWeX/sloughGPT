import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ChatInputRow } from './ChatInputRow'

afterEach(cleanup)

vi.mock('./ChatInputAccessories', () => ({
  ChatInputAccessories: ({ onImage, disabled }: any) => (
    <button data-testid="mock-accessories" disabled={disabled} onClick={() => onImage?.('data:url')}>Accessories</button>
  ),
}))

vi.mock('./ChatInputField', () => ({
  ChatInputField: ({ value, onChange, onSend, disabled }: any) => (
    <input data-testid="mock-input" value={value} disabled={disabled}
      onChange={(e) => onChange?.(e.target.value)}
      onKeyDown={(e) => e.key === 'Enter' && onSend?.()} />
  ),
}))

vi.mock('./ChatSendButton', () => ({
  ChatSendButton: ({ onSend, onStop, loading, disabled }: any) => (
    <button data-testid="mock-send" disabled={disabled || loading} onClick={loading ? onStop : onSend}>Send</button>
  ),
}))

describe('ChatInputRow', () => {
  const defaultProps = {
    value: '', onChange: vi.fn(), onSend: vi.fn(), loading: false, disabled: false,
    placeholder: 'Message...', textareaRef: { current: null } as React.RefObject<HTMLTextAreaElement | null>,
    onImage: vi.fn(), onTranscript: vi.fn(), hasContent: false,
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

  it('passes loading to send button', () => {
    render(<ChatInputRow {...defaultProps} loading={true} value="Hi" />)
    expect(screen.getByTestId('mock-send')).toBeDisabled()
  })

  it('has aria-label on the group', () => {
    render(<ChatInputRow {...defaultProps} />)
    expect(screen.getByRole('group', { name: 'Message composition' })).toBeDefined()
  })
})
