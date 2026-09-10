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
  addEventListener: vi.fn(),
  load: vi.fn(),
})))

vi.stubGlobal('AudioContext', vi.fn().mockImplementation(() => ({
  createMediaElementSource: vi.fn().mockReturnValue({ connect: vi.fn() }),
  createAnalyser: vi.fn().mockReturnValue({
    fftSize: 128,
    frequencyBinCount: 64,
    getFloatTimeDomainData: vi.fn(),
  }),
  destination: {},
  close: vi.fn(),
})))

import { VoiceComparisonCard } from './VoiceComparisonCard'

const STORAGE_KEY = 'sloughgpt-voice-recordings'

beforeEach(() => { localStorage.clear() })
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('VoiceComparisonCard', () => {
  it('shows message when no recordings exist', async () => {
    render(<VoiceComparisonCard />)
    await waitFor(() => {
      expect(screen.getByText('Record at least 2 clips to compare.')).toBeTruthy()
    })
  })

  it('shows message when only 1 recording exists', async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([
      { id: 'r1', url: 'blob:a', duration: 3000, label: 'Clip A', timestamp: 1 },
    ]))
    render(<VoiceComparisonCard />)
    await waitFor(() => {
      expect(screen.getByText('Record at least 1 more clip to compare.')).toBeTruthy()
    })
  })

  it('renders comparison form with 2+ recordings', async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([
      { id: 'r1', url: 'blob:a', duration: 3000, label: 'Clip A', timestamp: 1 },
      { id: 'r2', url: 'blob:b', duration: 4000, label: 'Clip B', timestamp: 2 },
    ]))
    render(<VoiceComparisonCard />)
    await waitFor(() => {
      expect(screen.getByText('Voice Comparison')).toBeTruthy()
      expect(screen.getByText('Compare')).toBeTruthy()
    })
  })

  it('Compare button is disabled without selections', async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([
      { id: 'r1', url: 'blob:a', duration: 3000, label: 'Clip A', timestamp: 1 },
      { id: 'r2', url: 'blob:b', duration: 4000, label: 'Clip B', timestamp: 2 },
    ]))
    render(<VoiceComparisonCard />)
    await waitFor(() => {
      expect(screen.getByText('Compare')).toBeTruthy()
    })
    expect(screen.getByText('Compare')).toHaveProperty('disabled', true)
  })

  it('displays recording options in select', async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([
      { id: 'r1', url: 'blob:a', duration: 3000, label: 'Morning Voice', timestamp: 1 },
      { id: 'r2', url: 'blob:b', duration: 4000, label: 'Evening Voice', timestamp: 2 },
    ]))
    render(<VoiceComparisonCard />)
    await waitFor(() => {
      expect(screen.getByText('Voice Comparison')).toBeTruthy()
      const options = document.querySelectorAll('option')
      expect(options.length).toBeGreaterThanOrEqual(3)
    })
  })
})
