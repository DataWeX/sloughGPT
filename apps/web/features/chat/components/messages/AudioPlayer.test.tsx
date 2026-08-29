import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@/lib/format-bytes', () => ({
  formatDuration: (ms: number) => {
    const s = Math.floor(ms / 1000)
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
  },
}))

vi.mock('@/lib/dev-log', () => ({
  logger: { info: vi.fn(), warning: vi.fn(), error: vi.fn() },
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  IconPlay: (p: any) => <svg {...p} data-testid="icon-play" />,
}))

import { AudioPlayer } from './AudioPlayer'

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(cleanup)

describe('AudioPlayer', () => {
  it('renders without crashing', () => {
    render(<AudioPlayer src="/audio.mp3" />)
    expect(screen.getByRole('button', { name: /Play/i })).toBeDefined()
  })

  it('renders with custom className', () => {
    render(<AudioPlayer src="/audio.mp3" className="custom-class" />)
    expect(screen.getByRole('button', { name: /Play/i })).toBeDefined()
  })

  it('shows play button with play label', () => {
    render(<AudioPlayer src="/audio.mp3" />)
    expect(screen.getByLabelText('Play')).toBeDefined()
  })

  it('toggles play/pause on click', () => {
    render(<AudioPlayer src="/audio.mp3" />)
    const btn = screen.getByRole('button', { name: /Play/i })
    fireEvent.click(btn)
    expect(screen.getByLabelText('Pause')).toBeDefined()
    fireEvent.click(btn)
    expect(screen.getByLabelText('Play')).toBeDefined()
  })

  it('displays formatted duration text', () => {
    render(<AudioPlayer src="/audio.mp3" durationMs={60000} />)
    expect(screen.getByText('1:00')).toBeDefined()
  })

  it('displays audio element with src', () => {
    render(<AudioPlayer src="/test-audio.wav" />)
    const audio = document.querySelector('audio')
    expect(audio).toBeDefined()
    expect(audio?.getAttribute('src')).toBe('/test-audio.wav')
  })

  it('renders progress bar structure', () => {
    render(<AudioPlayer src="/audio.mp3" />)
    expect(document.querySelector('[class*="bg-muted"]')).toBeDefined()
    expect(document.querySelector('[class*="bg-primary"]')).toBeDefined()
  })

  it('displays time labels', () => {
    render(<AudioPlayer src="/audio.mp3" durationMs={120000} />)
    const timeLabels = screen.getAllByText(/0:\d{2}/)
    expect(timeLabels.length).toBeGreaterThanOrEqual(2)
  })
})
