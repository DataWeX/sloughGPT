// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'
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
  it('renders default presets', () => {
    render(<VoicePresetCard />)
    expect(screen.getAllByTestId('voice-preset').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Natural').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Fast').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Deep').length).toBeGreaterThanOrEqual(1)
  })

  it('shows rate and pitch for each preset', () => {
    render(<VoicePresetCard />)
    expect(screen.getAllByText('1.0x · 1.0p').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('1.5x · 1.0p').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('0.7x · 1.0p').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onApply when preset clicked', () => {
    const onApply = vi.fn()
    render(<VoicePresetCard onApply={onApply} />)
    const natural = screen.getAllByText('Natural')[0]
    fireEvent.click(natural)
    expect(onApply).toHaveBeenCalledWith(expect.objectContaining({ name: 'Natural', rate: 1.0 }))
  })

  it('opens edit mode on Edit click', () => {
    render(<VoicePresetCard />)
    const editButtons = screen.getAllByText('Edit')
    fireEvent.click(editButtons[0])
    expect(screen.getAllByText('Save').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Cancel').length).toBeGreaterThanOrEqual(1)
  })

  it('can add a new preset', () => {
    render(<VoicePresetCard />)
    fireEvent.click(screen.getAllByText('+ Add')[0])
    expect(screen.getAllByText(/Preset \d/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Save').length).toBeGreaterThanOrEqual(1)
  })

  it('persists custom presets to localStorage', () => {
    const { unmount } = render(<VoicePresetCard />)
    fireEvent.click(screen.getAllByText('+ Add')[0])
    fireEvent.click(screen.getAllByText('Save')[0])
    unmount()
    const stored = JSON.parse(localStorage.getItem('sloughgpt-voice-presets') ?? '[]')
    expect(stored.length).toBe(6)
    expect(stored[5].name).toMatch(/Preset \d/)
  })

  it('loads presets from localStorage', () => {
    localStorage.setItem('sloughgpt-voice-presets', JSON.stringify([
      { name: 'Custom', rate: 1.2, pitch: 0.8, voice: '' },
    ]))
    render(<VoicePresetCard />)
    expect(screen.getAllByText('Custom').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('1.2x · 0.8p').length).toBeGreaterThanOrEqual(1)
  })

  it('can delete custom preset', () => {
    localStorage.setItem('sloughgpt-voice-presets', JSON.stringify([
      { name: 'Natural', rate: 1.0, pitch: 1.0, voice: '' },
      { name: 'Custom', rate: 1.2, pitch: 0.8, voice: '' },
    ]))
    render(<VoicePresetCard />)
    const delButtons = screen.getAllByText('Del')
    fireEvent.click(delButtons[0])
    expect(screen.queryAllByText('Custom').length).toBe(0)
  })

  it('does not show Del for default presets', () => {
    render(<VoicePresetCard />)
    expect(screen.queryAllByText('Del').length).toBe(0)
  })

  it('calls speechSynthesis.speak on Test click', () => {
    render(<VoicePresetCard />)
    const testButtons = screen.getAllByText('Test')
    fireEvent.click(testButtons[0])
    expect(window.speechSynthesis.speak).toHaveBeenCalled()
  })
})
