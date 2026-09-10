import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import PhonemeTypingMode from './PhonemeTypingMode'

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
  toIPA: vi.fn((phonemes: string[]) => phonemes.map(p => `[${p}]`)),
}))

vi.mock('@/components/phoneme/PhonemeReference', () => ({
  default: () => <div data-testid="phoneme-reference" />,
}))

describe('PhonemeTypingMode', () => {
  it('renders without errors', () => {
    render(<PhonemeTypingMode />)
    expect(screen.getByText('Phoneme Typing')).toBeDefined()
  })

  it('shows start button', () => {
    render(<PhonemeTypingMode />)
    expect(screen.getAllByText(/Start Typing/).length).toBeGreaterThan(0)
  })
})
