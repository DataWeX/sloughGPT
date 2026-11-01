import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import PronunciationSpeed from './PronunciationSpeed'

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
  }),
}))

describe('PronunciationSpeed', () => {
  it('renders without errors', () => {
    render(<PronunciationSpeed />)
    expect(screen.getByText('Practice Speed')).toBeDefined()
  })

  it('shows empty state when no history', () => {
    render(<PronunciationSpeed />)
    expect(screen.getAllByText(/Practice more to see/).length).toBeGreaterThan(0)
  })
})
