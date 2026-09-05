// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

vi.mock('@/lib/voice-controller', () => ({
  voiceController: { getStatus: vi.fn(), tts: vi.fn() },
}))
vi.mock('@/components/voice/VoicePresetCard', () => ({
  VoicePresetCard: () => <div data-testid="voice-preset" />,
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (s: any) => s({ addToast: vi.fn() }),
}))
vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...p }: any) => <div data-testid="card" {...p}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...p }: any) => <div data-testid="card-title" {...p}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...p }: any) => <button onClick={onClick} disabled={disabled} {...p}>{children}</button>,
  Textarea: ({ value, onChange, ...p }: any) => <textarea value={value} onChange={e => onChange(e)} {...p} />,
  StatCard: ({ label, value, ...p }: any) => <div data-testid="stat-card" data-label={label}>{String(value)}</div>,
  KpiGrid: ({ children, ...p }: any) => <div data-testid="kpi-grid" {...p}>{children}</div>,
  ActionCard: ({ children, title, actions, ...p }: any) => (
    <div data-testid="action-card" {...p}>
      {title && <div data-testid="action-card-title">{title}</div>}
      {actions}
      {children}
    </div>
  ),
  IconRefresh: () => <span>↻</span>,
}))

import { VoiceSection } from './VoiceSection'
import { voiceController } from '@/lib/voice-controller'

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(voiceController.getStatus).mockResolvedValue({
    available: true, model: 'bark-small', backend: 'huggingface', sample_rate: 22050,
  } as any)
  vi.mocked(voiceController.tts).mockResolvedValue({
    duration_ms: 1000, backend: 'hf-model', sample_rate: 22050, audio: 'base64data',
  } as any)
  ;(global as any).Audio = vi.fn().mockImplementation(() => ({
    play: vi.fn().mockResolvedValue(undefined),
    onended: null,
  }))
})

afterEach(() => cleanup())

describe('VoiceSection', () => {
  it('calls getStatus on mount', async () => {
    render(<VoiceSection />)
    await waitFor(() => {
      expect(voiceController.getStatus).toHaveBeenCalled()
    })
  })

  it('shows voice preset card', () => {
    render(<VoiceSection />)
    expect(screen.getByTestId('voice-preset')).toBeDefined()
  })

  it('generates speech on button click', async () => {
    render(<VoiceSection />)
    await waitFor(() => { expect(screen.getByText('Generate & Play')).toBeDefined() })
    fireEvent.change(screen.getByPlaceholderText('Enter text to speak...'), { target: { value: 'Hello world' } })
    fireEvent.click(screen.getByText('Generate & Play'))
    expect(voiceController.tts).toBeDefined()
  })
})
