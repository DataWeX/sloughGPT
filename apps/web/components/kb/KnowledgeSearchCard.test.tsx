/// <reference types="vitest" />
import '@testing-library/jest-dom/vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { afterEach, describe, it, expect, vi } from 'vitest'
import { KnowledgeSearchCard } from './KnowledgeSearchCard'

afterEach(() => cleanup())

const defaultProps = {
  query: '',
  results: [],
  loading: false,
  onQueryChange: vi.fn(),
  onSearch: vi.fn(),
}

describe('KnowledgeSearchCard', () => {
  it('renders search input', () => {
    render(<KnowledgeSearchCard {...defaultProps} />)
    expect(screen.getByPlaceholderText('Search knowledge...')).toBeInTheDocument()
  })

  it('renders search button', () => {
    render(<KnowledgeSearchCard {...defaultProps} />)
    expect(screen.getByText('Search')).toBeInTheDocument()
  })

  it('shows loading state', () => {
    render(<KnowledgeSearchCard {...defaultProps} loading={true} />)
    expect(screen.getByText('Searching...')).toBeInTheDocument()
  })

  it('displays results when provided', () => {
    const results = [
      { id: '1', content: 'Fact about cats', topic: 'animals', source: 'manual', importance: 0.8, score: 0.95 },
    ]
    render(<KnowledgeSearchCard {...defaultProps} results={results} />)
    expect(screen.getByText(/Fact about cats/)).toBeInTheDocument()
  })

  it('displays topic badge for results', () => {
    const results = [
      { id: '1', content: 'Fact about cats', topic: 'animals', source: 'manual', importance: 0.8, score: 0.95 },
    ]
    render(<KnowledgeSearchCard {...defaultProps} results={results} />)
    expect(screen.getByText('animals')).toBeInTheDocument()
  })
})
