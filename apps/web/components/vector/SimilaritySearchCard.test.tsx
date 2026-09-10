import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { SimilaritySearchCard } from './SimilaritySearchCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button data-testid="button" onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: (props: any) => <input data-testid="search-input" {...props} />,
}))

const defaultProps = {
  searchQuery: '',
  searchResults: [],
  searchTime: null,
  searching: false,
  onQueryChange: vi.fn(),
  onSearch: vi.fn(),
}

describe('SimilaritySearchCard', () => {
  it('renders the card title', () => {
    render(<SimilaritySearchCard {...defaultProps} />)
    expect(screen.getByText('Similarity Search')).toBeDefined()
  })

  it('renders the search input', () => {
    render(<SimilaritySearchCard {...defaultProps} />)
    expect(screen.getByTestId('search-input')).toBeDefined()
  })

  it('renders the search button', () => {
    render(<SimilaritySearchCard {...defaultProps} />)
    expect(screen.getByText('Search')).toBeDefined()
  })

  it('displays search results with text and score', () => {
    const results = [
      { id: '1', text: 'Hello world', score: 0.95 },
      { id: '2', text: 'Test entry', score: 0.82 },
    ]
    render(<SimilaritySearchCard {...defaultProps} searchResults={results} searchTime={12.5} />)
    expect(screen.getByText('Hello world')).toBeDefined()
    expect(screen.getByText('Test entry')).toBeDefined()
    expect(screen.getByText('95.0%')).toBeDefined()
    expect(screen.getByText('82.0%')).toBeDefined()
  })

  it('shows result count and time', () => {
    const results = [{ id: '1', text: 'Item', score: 0.9 }]
    render(<SimilaritySearchCard {...defaultProps} searchResults={results} searchTime={5.2} />)
    expect(screen.getByText('1 results in 5.2ms')).toBeDefined()
  })

  it('shows no results message when empty', () => {
    render(<SimilaritySearchCard {...defaultProps} searchQuery="test" />)
    expect(screen.getByText('No results found')).toBeDefined()
  })

  it('shows searching state', () => {
    render(<SimilaritySearchCard {...defaultProps} searching={true} />)
    expect(screen.getByText('Searching...')).toBeDefined()
  })
})
