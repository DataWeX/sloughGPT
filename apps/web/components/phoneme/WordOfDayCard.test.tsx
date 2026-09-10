import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import WordOfDayCard from './WordOfDayCard'

// Mock the store
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

describe('WordOfDayCard', () => {
  it('renders without errors', () => {
    render(<WordOfDayCard />)
    expect(screen.getByText('Word of the Day')).toBeDefined()
  })

  it('shows practice input', () => {
    render(<WordOfDayCard />)
    expect(screen.getAllByPlaceholderText('Type your pronunciation...').length).toBeGreaterThan(0)
  })
})
