/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach, beforeAll } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

const mockT = vi.fn((key: string) => {
  const map: Record<string, string> = {
    'common.starting': 'Starting...',
    'common.starting_sub': 'Model is loading, one moment',
    'chat.suggestion.chat': 'What is Man?',
    'chat.suggestion.train': 'How do I train a model?',
    'chat.suggestion.soul': 'What is a soul?',
    'chat.suggestion.models': 'What models are available?',
    'chat.send': 'Send',
  }
  return map[key] ?? key
})

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: mockT, locale: 'en' }),
}))

import { EmptyState } from './EmptyState'

beforeAll(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })),
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('EmptyState', () => {
  it('renders greeting and suggestions when hasModel is true', () => {
    render(<EmptyState hasModel onSuggestionClick={() => {}} />)
    expect(screen.getByText(/Good (morning|afternoon|evening)/)).toBeTruthy()
    expect(screen.getByRole('button', { name: /What is Man/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /train a model/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /What is a soul/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /models are available/ })).toBeTruthy()
  })

  it('shows connecting state when hasModel is false', () => {
    render(<EmptyState hasModel={false} />)
    expect(screen.getByText('Starting...')).toBeTruthy()
    expect(screen.getByText('Model is loading, one moment')).toBeTruthy()
    expect(screen.queryByText('Try asking')).toBeNull()
  })

  it('renders keyboard shortcuts hint', () => {
    render(<EmptyState hasModel />)
    expect(screen.getByText('Send')).toBeTruthy()
    expect(screen.getByText('new line')).toBeTruthy()
  })

  it('has accessible region with label', () => {
    render(<EmptyState hasModel />)
    expect(screen.getByRole('region', { name: 'Chat ready' })).toBeTruthy()
  })

  it('calls onSuggestionClick when a chip is clicked', () => {
    const onClick = vi.fn()
    render(<EmptyState hasModel onSuggestionClick={onClick} />)
    fireEvent.click(screen.getByRole('button', { name: /What is Man/ }))
    expect(onClick).toHaveBeenCalledWith('What is Man?')
  })

  it('renders mood orb element', () => {
    const { container } = render(<EmptyState hasModel />)
    const orb = container.querySelector('[aria-hidden="true"]')
    expect(orb).toBeTruthy()
  })
})
