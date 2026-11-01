import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import PronunciationGame from './PronunciationGame'

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

vi.mock('@/lib/phoneme-controller', () => ({
  phonemeController: { encode: vi.fn() },
  PHONEME_LANGUAGES: [
    { value: 'en', label: 'English' },
  ],
}))

describe('PronunciationGame', () => {
  it('renders without errors', () => {
    render(<PronunciationGame />)
    expect(screen.getByText('Pronunciation Game')).toBeDefined()
  })

  it('shows start button', () => {
    render(<PronunciationGame />)
    expect(screen.getAllByText(/Start Game/).length).toBeGreaterThan(0)
  })
})
