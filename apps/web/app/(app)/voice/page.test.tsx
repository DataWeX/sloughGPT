import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

const { mockTts, mockAddToast } = vi.hoisted(() => ({
  mockTts: vi.fn(), mockAddToast: vi.fn(),
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    Card: passthrough, CardContent: passthrough, Button: ({ children, onClick, disabled }: any) => <button onClick={onClick} disabled={disabled}>{children}</button>,
  }
})
vi.mock('@/lib/voice-controller', () => ({
  voiceController: { tts: (...a: unknown[]) => mockTts(...a), getStatus: vi.fn().mockResolvedValue({}) },
}))
vi.mock('@/lib/toast-store', () => ({ useToastStore: (sel: any) => sel({ addToast: mockAddToast }) }))
vi.mock('@/components/PageContainer', () => ({ PageContainer: ({ title, subtitle, children }: any) => <div><h1>{title}</h1><p>{subtitle}</p>{children}</div> }))

import VoicePage from './page'

afterEach(cleanup)
beforeEach(() => {
  vi.clearAllMocks()
  mockTts.mockResolvedValue({ audio: 'base64data', duration_ms: 1000, backend: 'hf-model', sample_rate: 22050 })
})

describe('VoicePage — Talk Out Loud', () => {
  it('renders Talk Out Loud header', () => {
    render(<VoicePage />)
    expect(screen.getByText('Talk Out Loud')).toBeDefined()
    expect(screen.getByText(/Tap mic/)).toBeDefined()
  })
  it('shows mic button', () => {
    render(<VoicePage />)
    expect(screen.getByLabelText(/Start listening/)).toBeDefined()
  })
  it('shows transcript placeholder', () => {
    render(<VoicePage />)
    expect(screen.getByText(/Words appear here/)).toBeDefined()
  })
  it('send disabled when transcript empty', () => {
    render(<VoicePage />)
    const btn = screen.getByText('Send to AI') as HTMLButtonElement
    expect(btn.disabled).toBe(true)
  })
})
