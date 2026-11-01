import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import BatchCard from './BatchCard'

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (state: unknown) => unknown) => selector({
    addToast: vi.fn(),
  }),
}))

vi.mock('@/lib/phoneme-controller', () => ({
  phonemeController: {
    batchEncode: vi.fn(),
    batchScore: vi.fn(),
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

describe('BatchCard', () => {
  it('renders without errors', () => {
    render(<BatchCard />)
    expect(screen.getByText('Batch Operations')).toBeDefined()
  })

  it('shows default encode input text', () => {
    render(<BatchCard />)
    const textarea = screen.getByPlaceholderText('Enter one word per line...')
    expect((textarea as HTMLTextAreaElement).value).toBe('hello\nworld\nplease')
  })

  it('renders Encode All button in encode mode', () => {
    render(<BatchCard />)
    expect(screen.getByRole('button', { name: /Encode All/ })).toBeDefined()
  })
})
