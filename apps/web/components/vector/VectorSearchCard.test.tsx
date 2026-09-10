/// <reference types="vitest" />
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { VectorSearchCard, type SearchResult } from './VectorSearchCard'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => <h3 data-testid="card-title" {...props}>{children}</h3>,
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button data-testid="button" onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: ({ placeholder, onChange, ...props }: React.InputHTMLAttributes<HTMLInputElement>) => (
    <input data-testid="input" placeholder={placeholder} onChange={onChange} {...props} />
  ),
}))

afterEach(() => cleanup())

const results: SearchResult[] = [
  { id: 'vec-1', text: 'Hello world', score: 0.95 },
  { id: 'vec-2', text: 'Test entry', score: 0.82 },
]

describe('VectorSearchCard', () => {
  const defaultProps = {
    query: '',
    results: [],
    searchTime: null,
    searching: false,
    onQueryChange: vi.fn(),
    onSearch: vi.fn(),
  }

  it('renders the card title', () => {
    render(<VectorSearchCard {...defaultProps} />)
    expect(screen.getByText('Similarity Search')).toBeDefined()
  })

  it('renders search input', () => {
    render(<VectorSearchCard {...defaultProps} />)
    expect(screen.getByPlaceholderText('Search for similar text...')).toBeDefined()
  })

  it('renders search button', () => {
    render(<VectorSearchCard {...defaultProps} />)
    expect(screen.getByText('Search')).toBeDefined()
  })

  it('renders search results with scores', () => {
    render(<VectorSearchCard {...defaultProps} results={results} searchTime={12.5} />)
    expect(screen.getByText('Hello world')).toBeDefined()
    expect(screen.getByText('95.0%')).toBeDefined()
  })

  it('shows no results message when query present but results empty', () => {
    render(<VectorSearchCard {...defaultProps} query="test" results={[]} />)
    expect(screen.getByText('No results found')).toBeDefined()
  })
})
