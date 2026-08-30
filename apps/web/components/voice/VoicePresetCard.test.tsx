// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const STORAGE_KEY = 'sloughgpt-voice-presets'

vi.mock('@/lib/db', () => ({
  chatDB: {
    getKV: vi.fn((key: string) => {
      const raw = localStorage.getItem(key)
      return Promise.resolve(raw ? JSON.parse(raw) : undefined)
    }),
    setKV: vi.fn((key: string, value: unknown) => {
      localStorage.setItem(key, JSON.stringify(value))
      return Promise.resolve()
    }),
    deleteKV: vi.fn((key: string) => {
      localStorage.removeItem(key)
      return Promise.resolve()
    }),
  },
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Slider: ({ label, value, onValueChange, formatValue, ...props }: any) => (
    <div data-testid={`slider-${label}`}>
      <input
        type="range"
        value={value?.[0] ?? 0}
        onChange={(e) => onValueChange?.([parseFloat(e.target.value)])}
        {...props}
      />
    </div>
  ),
  Select: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  SelectTrigger: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  SelectValue: ({ placeholder }: any) => <span>{placeholder}</span>,
  SelectContent: ({ children }: any) => <div>{children}</div>,
  SelectItem: ({ children, value, ...props }: any) => <div data-value={value} {...props}>{children}</div>,
}))

import { VoicePresetCard } from './VoicePresetCard'

beforeEach(() => {
  localStorage.clear()
  vi.stubGlobal('speechSynthesis', {
    getVoices: vi.fn().mockReturnValue([]),
    onvoiceschanged: null,
    cancel: vi.fn(),
    speak: vi.fn(),
  })
  vi.stubGlobal('SpeechSynthesisUtterance', vi.fn().mockImplementation(() => ({
    rate: 1, pitch: 1, voice: null, text: '',
  })))
})
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('VoicePresetCard', () => {
  it('renders default presets', async () => {
    render(<VoicePresetCard />)
    await waitFor(() => {
      expect(screen.getAllByTestId('voice-preset').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText('Natural').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Fast').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Deep').length).toBeGreaterThanOrEqual(1)
  })

  it('shows rate and pitch for each preset', async () => {
    render(<VoicePresetCard />)
    await waitFor(() => {
      expect(screen.getAllByText('Natural').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText('1.0x · 1.0p').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('1.5x · 1.0p').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('0.7x · 1.0p').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onApply when preset clicked', async () => {
    const onApply = vi.fn()
    render(<VoicePresetCard onApply={onApply} />)
    await waitFor(() => {
      expect(screen.getAllByText('Natural').length).toBeGreaterThanOrEqual(1)
    })
    const natural = screen.getAllByText('Natural')[0]
    fireEvent.click(natural)
    expect(onApply).toHaveBeenCalledWith(expect.objectContaining({ name: 'Natural', rate: 1.0 }))
  })

  it('opens edit mode on Edit click', async () => {
    render(<VoicePresetCard />)
    await waitFor(() => {
      expect(screen.getAllByText('Edit').length).toBeGreaterThanOrEqual(1)
    })
    const editButtons = screen.getAllByText('Edit')
    fireEvent.click(editButtons[0])
    expect(screen.getAllByText('Save').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Cancel').length).toBeGreaterThanOrEqual(1)
  })

  it('can add a new preset', async () => {
    render(<VoicePresetCard />)
    await waitFor(() => {
      expect(screen.getAllByText('+ Add').length).toBeGreaterThanOrEqual(1)
    })
    fireEvent.click(screen.getAllByText('+ Add')[0])
    expect(screen.getAllByText(/Preset \d/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Save').length).toBeGreaterThanOrEqual(1)
  })

  it('persists custom presets to localStorage', async () => {
    const { unmount } = render(<VoicePresetCard />)
    await waitFor(() => {
      expect(screen.getAllByText('+ Add').length).toBeGreaterThanOrEqual(1)
    })
    fireEvent.click(screen.getAllByText('+ Add')[0])
    fireEvent.click(screen.getAllByText('Save')[0])
    unmount()
    const stored = JSON.parse(localStorage.getItem('sloughgpt-voice-presets') ?? '[]')
    expect(stored.length).toBe(6)
    expect(stored[5].name).toMatch(/Preset \d/)
  })

  it('loads presets from localStorage', async () => {
    localStorage.setItem('sloughgpt-voice-presets', JSON.stringify([
      { name: 'Custom', rate: 1.2, pitch: 0.8, voice: '' },
    ]))
    render(<VoicePresetCard />)
    await waitFor(() => {
      expect(screen.getAllByText('Custom').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText('1.2x · 0.8p').length).toBeGreaterThanOrEqual(1)
  })

  it('can delete custom preset', async () => {
    localStorage.setItem('sloughgpt-voice-presets', JSON.stringify([
      { name: 'Natural', rate: 1.0, pitch: 1.0, voice: '' },
      { name: 'Custom', rate: 1.2, pitch: 0.8, voice: '' },
    ]))
    render(<VoicePresetCard />)
    await waitFor(() => {
      expect(screen.getAllByText('Del').length).toBeGreaterThanOrEqual(1)
    })
    const delButtons = screen.getAllByText('Del')
    fireEvent.click(delButtons[0])
    expect(screen.queryAllByText('Custom').length).toBe(0)
  })

  it('does not show Del for default presets', async () => {
    render(<VoicePresetCard />)
    await waitFor(() => {
      expect(screen.getAllByText('Natural').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.queryAllByText('Del').length).toBe(0)
  })

  it('calls speechSynthesis.speak on Test click', async () => {
    render(<VoicePresetCard />)
    await waitFor(() => {
      expect(screen.getAllByText('Test').length).toBeGreaterThanOrEqual(1)
    })
    const testButtons = screen.getAllByText('Test')
    fireEvent.click(testButtons[0])
    expect(window.speechSynthesis.speak).toHaveBeenCalled()
  })
})
