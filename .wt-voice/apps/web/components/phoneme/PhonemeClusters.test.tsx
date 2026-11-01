import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import PhonemeClusters from './PhonemeClusters'

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
  }),
}))

vi.mock('@/lib/phoneme-controller', () => ({
  toIPA: vi.fn((phonemes: string[]) => phonemes.map(p => `[${p}]`)),
}))

describe('PhonemeClusters', () => {
  it('renders without errors', () => {
    render(<PhonemeClusters />)
    expect(screen.getByText('Phoneme Clusters')).toBeDefined()
  })

  it('shows empty state when no history', () => {
    render(<PhonemeClusters />)
    expect(screen.getAllByText(/Practice more to see/).length).toBeGreaterThan(0)
  })
})
