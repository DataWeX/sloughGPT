import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import DetectLanguageCard from './DetectLanguageCard'

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (state: unknown) => unknown) => selector({
    addToast: vi.fn(),
  }),
}))

vi.mock('@/lib/phoneme-controller', () => ({
  phonemeController: {
    detectLanguage: vi.fn(),
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

describe('DetectLanguageCard', () => {
  it('renders without errors', () => {
    render(<DetectLanguageCard />)
    expect(screen.getByText('Detect Language')).toBeDefined()
  })

  it('renders Detect button', () => {
    render(<DetectLanguageCard />)
    expect(screen.getByRole('button', { name: /Detect/ })).toBeDefined()
  })

  it('shows sample word buttons', () => {
    render(<DetectLanguageCard />)
    expect(screen.getByText('hello')).toBeDefined()
    expect(screen.getByText('hallo')).toBeDefined()
  })
})
