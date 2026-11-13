import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { useState } from 'react'
import { describe, it, expect, vi, afterEach } from 'vitest'

vi.mock('./ImageUpload', () => ({
  ImagePreview: ({ image, onRemove }: any) => (
    <div data-testid="image-preview">
      <span>{image.id}</span>
      <button onClick={() => onRemove(image.id)}>remove</button>
    </div>
  ),
  ImageUpload: ({ onImage, disabled }: any) => (
    <button
      data-testid="image-upload"
      onClick={() => onImage?.('data:image/png,test')}
      disabled={disabled}
    >
      Upload image
    </button>
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
      <button
        data-testid="voice-fire"
        onClick={() => props.onTranscript?.((globalThis as any).__voiceText ?? 'spoken')}
      >
        voice
      </button>
    </div>
  ),
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (key: string) => key, locale: 'en' }),
}))

vi.mock('@/hooks/useConsciousnessStatus', () => ({
  useConsciousnessStatus: () => ({ status: null }),
  getConsciousnessLevelLabel: () => 'Off',
  getQualiaMood: () => '',
}))

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    listPersonas: async () => ({ personas: [] }),
    activatePersona: async () => {},
  },
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

  it('shows tool_call band when loading with tool_call status', () => {
    const { container } = render(
      <ChatInput {...base} loading streamingStatus="tool_call" streamingToolName="search" />,
    )
    expect(container.querySelector('[role="status"]')).toBeInTheDocument()
  })

  it('does NOT duplicate the generating indicator (inline turn dots own it)', () => {
    const { container } = render(<ChatInput {...base} loading streamingStatus="generating" />)
    expect(container.querySelector('[role="status"]')).not.toBeInTheDocument()
  })

  it('does not show loading indicator when not loading', () => {
    const { container } = render(<ChatInput {...base} />)
    expect(container.querySelector('[role="status"]')).not.toBeInTheDocument()
  })

  it('shows image previews when images present', () => {
    const { container } = render(
      <ChatInput
        {...base}
        images={[{ id: 'img1', dataUrl: 'data:image/png,test', name: 'test.png' }]}
      />,
    )
    expect(
      container.querySelectorAll('[data-testid="image-preview"]').length,
    ).toBeGreaterThanOrEqual(1)
  })

  it('shows API offline placeholder when health is offline', () => {
    const { container } = render(
      <ChatInput {...base} health={'offline' as unknown as ApiHealthSnapshot} />,
    )
    const row = container.querySelector('[data-testid="chat-input-row"]')
    expect(row?.textContent).toContain('placeholder: Service offline...')
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

  it('appends each final voice transcript exactly once (append semantic)', () => {
    // The seam contract with VoiceInput: VoiceInput emits each FINAL utterance
    // once, and this handler appends it. Spoken text must land once — never be
    // re-appended, never drop text the user already typed.
    const w = globalThis as any
    function Harness() {
      const [value, setValue] = useState('')
      return <ChatInput {...base} value={value} onChange={setValue} />
    }
    render(<Harness />)
    const row = () => screen.getByTestId('chat-input-row').textContent ?? ''

    w.__voiceText = 'spoken one'
    fireEvent.click(screen.getByTestId('voice-fire'))
    expect(row()).toContain('value: spoken one')

    w.__voiceText = 'spoken two'
    fireEvent.click(screen.getByTestId('voice-fire'))
    expect(row()).toContain('value: spoken one spoken two')
    // Exactly once each — no doubling of the first utterance.
    expect(row()).not.toContain('spoken one spoken one')
    expect(row()).not.toContain('spoken two spoken two')
    delete w.__voiceText
  })
})
