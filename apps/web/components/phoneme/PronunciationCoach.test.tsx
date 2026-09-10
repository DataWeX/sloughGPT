import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import PronunciationCoach from './PronunciationCoach'

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
  }),
}))

describe('PronunciationCoach', () => {
  it('renders without errors', () => {
    render(<PronunciationCoach />)
    expect(screen.getByText('Pronunciation Coach')).toBeDefined()
  })

  it('shows general tip when insufficient history', () => {
    render(<PronunciationCoach />)
    expect(screen.getAllByText('Pronunciation Coach').length).toBeGreaterThan(0)
    expect(screen.getAllByText('0 tips').length).toBeGreaterThan(0)
  })
})
