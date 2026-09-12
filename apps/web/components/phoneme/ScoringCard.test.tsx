import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import ScoringCard from './ScoringCard'

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
  phonemeController: {
    score: vi.fn(),
  },
  PHONEME_LANGUAGES: [
    { value: 'en', label: 'English' },
    { value: 'de', label: 'German' },
    { value: 'fr', label: 'French' },
    { value: 'es', label: 'Spanish' },
    { value: 'it', label: 'Italian' },
    { value: 'pt', label: 'Portuguese' },
  ],
}))

vi.mock('./PhonemeSkeleton', () => ({ default: () => <div data-testid="skeleton" /> }))

describe('ScoringCard', () => {
  it('renders without errors', () => {
    render(<ScoringCard />)
    expect(screen.getByText('Score Pronunciation')).toBeDefined()
  })

  it('shows default target and spoken values', () => {
    render(<ScoringCard />)
    const inputs = screen.getAllByRole('textbox')
    expect((inputs[0] as HTMLInputElement).value).toBe('hello')
    expect((inputs[1] as HTMLInputElement).value).toBe('hello')
  })

  it('renders Score button', () => {
    render(<ScoringCard />)
    expect(screen.getByRole('button', { name: /Score/ })).toBeDefined()
  })
})
