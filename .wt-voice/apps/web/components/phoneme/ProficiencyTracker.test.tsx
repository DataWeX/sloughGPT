import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import ProficiencyTracker from './ProficiencyTracker'

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

describe('ProficiencyTracker', () => {
  it('renders without errors', () => {
    render(<ProficiencyTracker />)
    expect(screen.getByText('Language Proficiency')).toBeDefined()
  })

  it('shows empty state when no history', () => {
    render(<ProficiencyTracker />)
    expect(screen.getAllByText(/Start practicing/).length).toBeGreaterThan(0)
  })
})
