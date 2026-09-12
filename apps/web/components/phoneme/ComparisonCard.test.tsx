import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import ComparisonCard from './ComparisonCard'

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
}))

vi.mock('./PhonemeSkeleton', () => ({ default: () => <div data-testid="skeleton" /> }))

describe('ComparisonCard', () => {
  it('renders without errors', () => {
    render(<ComparisonCard />)
    expect(screen.getByText('Compare Words')).toBeDefined()
  })

  it('shows default word inputs', () => {
    render(<ComparisonCard />)
    const inputs = screen.getAllByRole('textbox')
    expect((inputs[0] as HTMLInputElement).value).toBe('hello')
    expect((inputs[1] as HTMLInputElement).value).toBe('world')
  })

  it('renders Compare button', () => {
    render(<ComparisonCard />)
    expect(screen.getByRole('button', { name: /Compare/ })).toBeDefined()
  })
})
