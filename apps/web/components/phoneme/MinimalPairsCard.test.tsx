import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import MinimalPairsCard from './MinimalPairsCard'

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
  phonemeController: { encode: vi.fn(), score: vi.fn() },
  PHONEME_LANGUAGES: [
    { value: 'en', label: 'English' },
  ],
}))

describe('MinimalPairsCard', () => {
  it('renders without errors', () => {
    render(<MinimalPairsCard />)
    expect(screen.getByText('Minimal Pairs')).toBeDefined()
  })
})
