// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@/lib/db', () => ({
  chatDB: {
    getKV: vi.fn((key: string) => {
      const raw = localStorage.getItem(key)
      return Promise.resolve(raw ? JSON.parse(raw) : undefined)
    }),
    setKV: vi.fn().mockResolvedValue(undefined),
    deleteKV: vi.fn().mockResolvedValue(undefined),
  },
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Select: ({ children, value, onValueChange, ...props }: any) => (
    <select data-testid={props['aria-label']} value={value} onChange={(e) => onValueChange(e.target.value)}>{children}</select>
  ),
  SelectTrigger: ({ children }: any) => <div>{children}</div>,
  SelectValue: ({ placeholder }: any) => <span>{placeholder}</span>,
  SelectContent: ({ children }: any) => <div>{children}</div>,
  SelectItem: ({ children, value }: any) => <option value={value}>{children}</option>,
}))

vi.stubGlobal('Audio', vi.fn().mockImplementation(() => ({
  play: vi.fn(),
  addEventListener: vi.fn((event: string, cb: Function, opts?: any) => {
    if (event === 'canplaythrough') setTimeout(cb, 0)
    if (event === 'error') { /* noop */ }
  }),
  load: vi.fn(),
})))

vi.stubGlobal('AudioContext', vi.fn().mockImplementation(() => ({
  createMediaElementSource: vi.fn().mockReturnValue({ connect: vi.fn() }),
  createAnalyser: vi.fn().mockReturnValue({
    fftSize: 256,
    frequencyBinCount: 128,
    getFloatTimeDomainData: vi.fn(),
    getFloatFrequencyData: vi.fn(),
  }),
  destination: {},
  close: vi.fn(),
})))

import { VoiceWaveformCard } from './VoiceWaveformCard'

beforeEach(() => { localStorage.clear() })
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('VoiceWaveformCard', () => {
  it('renders the component', () => {
    render(<VoiceWaveformCard />)
    expect(screen.getByTestId('voice-waveform')).toBeTruthy()
  })

  it('shows empty state message when no recordings', async () => {
    render(<VoiceWaveformCard />)
    await waitFor(() => {
      expect(screen.getByText('No recordings yet. Record something first.')).toBeTruthy()
    })
  })

  it('renders select with recordings', async () => {
    localStorage.setItem('sloughgpt-voice-recordings', JSON.stringify([
      { id: 'r1', url: 'blob:a', duration: 5000, label: 'Voice Note', timestamp: 1 },
    ]))
    render(<VoiceWaveformCard />)
    await waitFor(() => {
      expect(screen.getByText('Voice Note (5.0s)')).toBeTruthy()
    })
  })

  it('shows title', () => {
    render(<VoiceWaveformCard />)
    expect(screen.getByText('Waveform Analysis')).toBeTruthy()
  })
})
