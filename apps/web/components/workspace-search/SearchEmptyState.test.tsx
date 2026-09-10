/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { SearchEmptyState } from './SearchEmptyState'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({}))

describe('SearchEmptyState', () => {
  it('renders default message when no query and no message', () => {
    render(<SearchEmptyState />)
    expect(screen.getByText('Type to search across all workspace data')).toBeDefined()
  })

  it('renders no results message when query provided', () => {
    render(<SearchEmptyState query="test" />)
    expect(screen.getByText('No results for "test"')).toBeDefined()
  })

  it('renders custom message when provided', () => {
    render(<SearchEmptyState message="Nothing found here" />)
    expect(screen.getByText('Nothing found here')).toBeDefined()
  })

  it('custom message takes precedence over query', () => {
    render(<SearchEmptyState query="test" message="Custom" />)
    expect(screen.getByText('Custom')).toBeDefined()
  })

  it('renders with empty query string', () => {
    render(<SearchEmptyState query="" />)
    expect(screen.getByText('Type to search across all workspace data')).toBeDefined()
  })
})
