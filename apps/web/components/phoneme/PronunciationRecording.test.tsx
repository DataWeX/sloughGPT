import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import PronunciationRecording from './PronunciationRecording'

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    addToHistory: vi.fn(),
  }),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (state: unknown) => unknown) => selector({
    addToast: vi.fn(),
  }),
}))

describe('PronunciationRecording', () => {
  it('renders without errors', () => {
    render(<PronunciationRecording />)
    expect(screen.getByText('Pronunciation Recording')).toBeDefined()
  })

  it('shows recording instructions', () => {
    render(<PronunciationRecording />)
    expect(screen.getAllByText(/Click to start recording/).length).toBeGreaterThan(0)
  })
})
