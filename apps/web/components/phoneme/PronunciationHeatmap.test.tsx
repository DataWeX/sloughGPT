import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import PronunciationHeatmap from './PronunciationHeatmap'

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
  }),
}))

vi.mock('@/lib/phoneme-controller', () => ({
  toIPA: vi.fn((phonemes: string[]) => phonemes.map(p => `[${p}]`)),
}))

describe('PronunciationHeatmap', () => {
  it('renders without errors', () => {
    render(<PronunciationHeatmap />)
    expect(screen.getByText('Pronunciation Heatmap')).toBeDefined()
  })

  it('shows empty state when no history', () => {
    render(<PronunciationHeatmap />)
    expect(screen.getAllByText(/Practice more to see/).length).toBeGreaterThan(0)
  })
})
