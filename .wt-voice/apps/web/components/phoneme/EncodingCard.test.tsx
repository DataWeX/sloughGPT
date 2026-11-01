import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import EncodingCard from './EncodingCard'

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (state: unknown) => unknown) => selector({
    addToast: vi.fn(),
  }),
}))

vi.mock('@/lib/phoneme-controller', () => ({
  phonemeController: {
    encode: vi.fn(),
  },
  PHONEME_LANGUAGES: [
    { value: 'en', label: 'English' },
    { value: 'de', label: 'German' },
    { value: 'fr', label: 'French' },
    { value: 'es', label: 'Spanish' },
    { value: 'it', label: 'Italian' },
    { value: 'pt', label: 'Portuguese' },
  ],
  toIPA: (phonemes: string[]) => phonemes.map(p => `/${p.toLowerCase()}/`),
}))

vi.mock('./PhonemeSkeleton', () => ({ default: () => <div data-testid="skeleton" /> }))

describe('EncodingCard', () => {
  it('renders without errors', () => {
    render(<EncodingCard />)
    expect(screen.getByText('Encode Text to Phonemes')).toBeDefined()
  })

  it('shows default input text', () => {
    render(<EncodingCard />)
    const input = screen.getByPlaceholderText('Enter text to encode...')
    expect((input as HTMLInputElement).value).toBe('hello world')
  })

  it('renders Encode button', () => {
    render(<EncodingCard />)
    expect(screen.getByRole('button', { name: /Encode/ })).toBeDefined()
  })
})
