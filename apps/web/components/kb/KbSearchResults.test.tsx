import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { KbSearchResults } from './KbSearchResults'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button data-testid="button" onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: ({ onChange, onKeyDown, ...props }: any) => (
    <input data-testid="input" onChange={onChange} onKeyDown={onKeyDown} {...props} />
  ),
}))

afterEach(() => cleanup())

describe('KbSearchResults', () => {
  const defaultProps = {
    query: '',
    results: [],
    loading: false,
    onQueryChange: vi.fn(),
    onSearch: vi.fn(),
  }

  it('renders the search input', () => {
    render(<KbSearchResults {...defaultProps} />)
    expect(screen.getByPlaceholderText('Search knowledge...')).toBeDefined()
  })

  it('renders the Search button', () => {
    render(<KbSearchResults {...defaultProps} />)
    expect(screen.getByText('Search')).toBeDefined()
  })

  it('displays results when provided', () => {
    const results = [
      { id: '1', content: 'Fact about cats', topic: 'animals', score: 0.95 },
      { id: '2', content: 'Fact about dogs', topic: 'animals', score: 0.87 },
    ]
    render(<KbSearchResults {...defaultProps} results={results} />)
    expect(screen.getByText('Fact about cats')).toBeDefined()
    expect(screen.getByText('Fact about dogs')).toBeDefined()
  })

  it('shows score for each result', () => {
    const results = [{ id: '1', content: 'Test fact', topic: 'test', score: 0.123 }]
    render(<KbSearchResults {...defaultProps} results={results} />)
    expect(screen.getByText('Score: 0.123')).toBeDefined()
  })

  it('shows topic badge for each result', () => {
    const results = [{ id: '1', content: 'Test', topic: 'science', score: 0.5 }]
    render(<KbSearchResults {...defaultProps} results={results} />)
    expect(screen.getByText('science')).toBeDefined()
  })

  it('shows Searching... when loading', () => {
    render(<KbSearchResults {...defaultProps} loading={true} />)
    expect(screen.getByText('Searching...')).toBeDefined()
  })

  it('calls onSearch when Search button clicked', () => {
    const onSearch = vi.fn()
    render(<KbSearchResults {...defaultProps} onSearch={onSearch} />)
    fireEvent.click(screen.getByText('Search'))
    expect(onSearch).toHaveBeenCalledOnce()
  })
})
