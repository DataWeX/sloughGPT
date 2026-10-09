import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import CustomWordList from './CustomWordList'

vi.mock('@/lib/phoneme-store', () => ({
  usePhonemeStore: (selector: (state: unknown) => unknown) => selector({
    history: [],
  }),
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (state: unknown) => unknown) => selector({
    addToast: vi.fn(),
  }),
}))

vi.mock('@/lib/phoneme-controller', () => ({
  phonemeController: { encode: vi.fn() },
  PHONEME_LANGUAGES: [
    { value: 'en', label: 'English' },
  ],
}))

describe('CustomWordList', () => {
  it('renders without errors', () => {
    render(<CustomWordList />)
    expect(screen.getByText('Custom Word List')).toBeDefined()
  })

  it('shows add input', () => {
    render(<CustomWordList />)
    expect(screen.getAllByPlaceholderText('Add a word to practice...').length).toBeGreaterThan(0)
  })
})
