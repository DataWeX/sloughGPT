import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import PhonemeNetwork from './PhonemeNetwork'

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
  }),
}))

vi.mock('@/lib/phoneme-controller', () => ({
  toIPA: vi.fn((phonemes: string[]) => phonemes.map(p => `[${p}]`)),
}))

describe('PhonemeNetwork', () => {
  it('renders without errors', () => {
    render(<PhonemeNetwork />)
    expect(screen.getByText('Phoneme Network')).toBeDefined()
  })

  it('shows empty state when no history', () => {
    render(<PhonemeNetwork />)
    expect(screen.getAllByText(/Practice more to see/).length).toBeGreaterThan(0)
  })
})
