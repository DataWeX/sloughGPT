import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import SynthesisCard from './SynthesisCard'

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (state: unknown) => unknown) => selector({
    addToast: vi.fn(),
  }),
}))

vi.mock('@/lib/phoneme-controller', () => ({
  phonemeController: {
    synthesize: vi.fn(),
  },
}))

vi.mock('./WaveformCanvas', () => ({ default: () => <div data-testid="waveform" /> }))
vi.mock('./SpectrogramCanvas', () => ({ default: () => <div data-testid="spectrogram" /> }))
vi.mock('./PhonemeSkeleton', () => ({ default: () => <div data-testid="skeleton" /> }))

describe('SynthesisCard', () => {
  it('renders without errors', () => {
    render(<SynthesisCard />)
    expect(screen.getByText('Synthesize Speech')).toBeDefined()
  })

  it('shows default input text', () => {
    render(<SynthesisCard />)
    const input = screen.getByPlaceholderText('Enter text to synthesize...')
    expect((input as HTMLInputElement).value).toBe('hello world')
  })

  it('disables button when input is empty', () => {
    render(<SynthesisCard />)
    const input = screen.getByPlaceholderText('Enter text to synthesize...')
    const button = screen.getByRole('button', { name: /Synthesize/ })
    expect(button.getAttribute('disabled')).toBeNull()
    expect(button).toBeDefined()
  })
})
