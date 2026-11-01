import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import ConfusionMatrix from './ConfusionMatrix'

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
  }),
}))

describe('ConfusionMatrix', () => {
  it('renders without errors', () => {
    render(<ConfusionMatrix />)
    expect(screen.getByText('Confusion Patterns')).toBeDefined()
  })

  it('shows empty state when no history', () => {
    render(<ConfusionMatrix />)
    expect(screen.getAllByText(/Practice more to see/).length).toBeGreaterThan(0)
  })
})
