import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import AudioPronunciationGuide from './AudioPronunciationGuide'

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (state: unknown) => unknown) => selector({
    addToast: vi.fn(),
  }),
}))

vi.mock('@/lib/phoneme-controller', () => ({
  phonemeController: {
    synthesize: vi.fn(),
  },
  PHONEME_LANGUAGES: [
    { value: 'en', label: 'English' },
    { value: 'de', label: 'German' },
    { value: 'fr', label: 'French' },
    { value: 'es', label: 'Spanish' },
  ],
}))

describe('AudioPronunciationGuide', () => {
  it('renders without errors', () => {
    render(<AudioPronunciationGuide />)
    expect(screen.getByText('Audio Pronunciation')).toBeDefined()
  })

  it('displays the default word', () => {
    render(<AudioPronunciationGuide />)
    expect(screen.getByText('hello')).toBeDefined()
  })

  it('renders Generate Audio button', () => {
    render(<AudioPronunciationGuide />)
    expect(screen.getByRole('button', { name: /Generate Audio/ })).toBeDefined()
  })
})
