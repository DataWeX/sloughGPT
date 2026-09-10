import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import PronunciationTips from './PronunciationTips'

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
  }),
}))

vi.mock('@/lib/phoneme-controller', () => ({
  PHONEME_LANGUAGES: [
    { value: 'en', label: 'English' },
  ],
}))

describe('PronunciationTips', () => {
  it('renders without errors', () => {
    render(<PronunciationTips />)
    expect(screen.getByText('Pronunciation Tips')).toBeDefined()
  })

  it('shows tips count', () => {
    render(<PronunciationTips />)
    expect(screen.getAllByText('5 tips').length).toBeGreaterThan(0)
  })
})
