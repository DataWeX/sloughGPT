import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import PhonemeFrequency from './PhonemeFrequency'

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
  }),
}))

describe('PhonemeFrequency', () => {
  it('renders without errors', () => {
    render(<PhonemeFrequency />)
    expect(screen.getByText('Phoneme Frequency')).toBeDefined()
  })

  it('shows empty state when no history', () => {
    render(<PhonemeFrequency />)
    expect(screen.getAllByText(/Practice more to see/).length).toBeGreaterThan(0)
  })
})
