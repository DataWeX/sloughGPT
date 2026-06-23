// @vitest-environment jsdom
import { render, screen, cleanup } from '@testing-library/react'
import { describe, it, expect, vi, afterEach } from 'vitest'

vi.mock('./ImageUpload', () => ({
  ImagePreview: ({ image, onRemove }: any) => (
    <div data-testid="image-preview">
      <span>{image.id}</span>
      <button onClick={() => onRemove(image.id)}>remove</button>
    </div>
  ),
  ImageUpload: ({ onImage, disabled }: any) => (
    <button data-testid="image-upload" onClick={() => onImage?.('data:image/png,test')} disabled={disabled}>Upload image</button>
  ),
}))

vi.mock('./ChatInputRow', () => ({
  ChatInputRow: (props: any) => (
    <div data-testid="chat-input-row">
      <span>value: {props.value}</span>
      <span>placeholder: {props.placeholder}</span>
      <span>disabled: {String(props.disabled)}</span>
      <span>loading: {String(props.loading)}</span>
      <span>hasContent: {String(props.hasContent)}</span>
    </div>
  ),
}))

import { ChatInput } from './ChatInput'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'

describe('ChatInput', () => {
  const base = {
    value: '',
    onChange: vi.fn(),
    onSend: vi.fn(),
    onStop: vi.fn(),
    loading: false,
    health: { model_loaded: true, model_type: 'gpt2' } as unknown as ApiHealthSnapshot,
  }

  afterEach(() => cleanup())

  it('renders section element', () => {
    const { container } = render(<ChatInput {...base} />)
    expect(container.querySelector('section')).toBeInTheDocument()
  })

  it('shows loading indicator when loading', () => {
    const { container } = render(<ChatInput {...base} loading />)
    expect(container.querySelector('[role="status"]')).toBeInTheDocument()
  })

  it('does not show loading indicator when not loading', () => {
    const { container } = render(<ChatInput {...base} />)
    expect(container.querySelector('[role="status"]')).not.toBeInTheDocument()
  })

  it('shows image previews when images present', () => {
    const { container } = render(<ChatInput {...base} images={[{ id: 'img1', dataUrl: 'data:image/png,test', name: 'test.png' }]} />)
    expect(container.querySelectorAll('[data-testid="image-preview"]').length).toBeGreaterThanOrEqual(1)
  })

  it('shows API offline placeholder when health is offline', () => {
    const { container } = render(<ChatInput {...base} health={'offline' as unknown as ApiHealthSnapshot} />)
    const row = container.querySelector('[data-testid="chat-input-row"]')
    expect(row?.textContent).toContain('placeholder: API offline...')
  })

  it('shows Loading model... when no model loaded', () => {
    const health = { model_loaded: false } as unknown as ApiHealthSnapshot
    const { container } = render(<ChatInput {...base} health={health} />)
    const row = container.querySelector('[data-testid="chat-input-row"]')
    expect(row?.textContent).toContain('placeholder: Loading model...')
  })

  it('passes content to ChatInputRow', () => {
    const { container } = render(<ChatInput {...base} value="hello" />)
    const row = container.querySelector('[data-testid="chat-input-row"]')
    expect(row?.textContent).toContain('hello')
  })
})
