import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import type { VoiceSettings } from '@/features/chat/hooks/useVoiceChat'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Select: ({ children, value, onValueChange, ...props }: any) => (
    <select data-testid="select" value={value} onChange={(e) => onValueChange?.(e.target.value)} {...props}>
      {children}
    </select>
  ),
  SelectTrigger: ({ children }: any) => <>{children}</>,
  SelectValue: ({ placeholder }: any) => <span>{placeholder}</span>,
  SelectContent: ({ children }: any) => <>{children}</>,
  SelectItem: ({ children, value, ...props }: any) => <option value={value} {...props}>{children}</option>,
  Slider: ({ label, value, onValueChange, ...props }: any) => (
    <div>
      <label>{label}</label>
      <input
        type="range"
        aria-label={label}
        value={value?.[0] ?? 0}
        onChange={(e) => onValueChange?.([parseFloat(e.target.value)])}
      />
    </div>
  ),
  Switch: ({ checked, onCheckedChange, ...props }: any) => (
    <input
      type="checkbox"
      checked={checked}
      onChange={(e) => onCheckedChange?.(e.target.checked)}
      {...props}
    />
  ),
}))

import { VoiceSettingsPanel } from './VoiceSettingsPanel'

const defaultSettings: VoiceSettings = {
  rate: 1,
  pitch: 1,
  interruptThreshold: 0.15,
  autoResume: true,
  pushToTalk: false,
  streamingTTS: false,
  voiceName: null,
}

afterEach(cleanup)

describe('VoiceSettingsPanel', () => {
  const updateSettings = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders without crashing', () => {
    render(
      <VoiceSettingsPanel
        settings={defaultSettings}
        availableVoices={[]}
        updateSettings={updateSettings}
      />,
    )
    expect(screen.getByText('Speech Rate')).toBeDefined()
  })

  it('renders all slider labels', () => {
    render(
      <VoiceSettingsPanel
        settings={defaultSettings}
        availableVoices={[]}
        updateSettings={updateSettings}
      />,
    )
    expect(screen.getByText('Speech Rate')).toBeDefined()
    expect(screen.getByText('Pitch')).toBeDefined()
    expect(screen.getByText('Interrupt Sensitivity')).toBeDefined()
  })

  it('renders toggle switches', () => {
    render(
      <VoiceSettingsPanel
        settings={defaultSettings}
        availableVoices={[]}
        updateSettings={updateSettings}
      />,
    )
    expect(screen.getByLabelText('Auto-resume listening')).toBeDefined()
    expect(screen.getByLabelText('Push to talk')).toBeDefined()
    expect(screen.getByLabelText('Stream speech')).toBeDefined()
  })

  it('calls updateSettings when a switch is toggled', () => {
    render(
      <VoiceSettingsPanel
        settings={defaultSettings}
        availableVoices={[]}
        updateSettings={updateSettings}
      />,
    )
    const autoResumeSwitch = screen.getByLabelText('Auto-resume listening')
    fireEvent.click(autoResumeSwitch)
    expect(updateSettings).toHaveBeenCalledWith({ autoResume: expect.any(Boolean) })
  })

  it('renders voice select when voices are available', () => {
    render(
      <VoiceSettingsPanel
        settings={defaultSettings}
        availableVoices={[
          { name: 'Alice', lang: 'en-US' },
          { name: 'Bob', lang: 'fr-FR' },
        ]}
        updateSettings={updateSettings}
      />,
    )
    expect(screen.getByText('Voice')).toBeDefined()
    expect(screen.getByText(/Alice/)).toBeDefined()
  })

  it('hides voice select when no voices available', () => {
    render(
      <VoiceSettingsPanel
        settings={defaultSettings}
        availableVoices={[]}
        updateSettings={updateSettings}
      />,
    )
    expect(screen.queryByText('Voice')).toBeNull()
  })
})
