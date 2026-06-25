/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach, beforeAll } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

const mockT = vi.fn((key: string) => {
  const map: Record<string, string> = {
    'common.starting': 'Starting...',
    'common.starting_sub': 'Model is loading, one moment',
    'chat.send': 'Send',
  }
  return map[key] ?? key
})

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: mockT, locale: 'en' }),
}))

import { EmptyState } from './EmptyState'

const mockSuggestions = [
  { text: 'Write a poem', icon: '✍️' },
  { text: 'Explain quantum physics', icon: '💡' },
  { text: 'Plan a weekend trip', icon: '🗺️' },
  { text: 'Help me practice Spanish', icon: '🌐' },
]

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
  it('renders greeting when hasModel is true', () => {
    render(<EmptyState hasModel />)
    expect(screen.getByText(/Good (morning|afternoon|evening)/)).toBeTruthy()
  })

  it('renders suggestion chips when suggestions prop provided', () => {
    render(<EmptyState hasModel suggestions={mockSuggestions} onSuggestionClick={() => {}} />)
    expect(screen.getByText('Try asking')).toBeTruthy()
    expect(screen.getByRole('button', { name: /Write a poem/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Plan a weekend trip/ })).toBeTruthy()
  })

  it('does not show suggestion section when suggestions is empty', () => {
    render(<EmptyState hasModel suggestions={[]} />)
    expect(screen.queryByText('Try asking')).toBeNull()
  })

  it('does not show suggestion section when suggestions not provided', () => {
    render(<EmptyState hasModel />)
    expect(screen.queryByText('Try asking')).toBeNull()
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
    render(<EmptyState hasModel suggestions={mockSuggestions} onSuggestionClick={onClick} />)
    fireEvent.click(screen.getByRole('button', { name: /Write a poem/ }))
    expect(onClick).toHaveBeenCalledWith('Write a poem')
  })

  it('renders mood orb element', () => {
    const { container } = render(<EmptyState hasModel />)
    const orb = container.querySelector('[aria-hidden="true"]')
    expect(orb).toBeTruthy()
  })
})
