/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { SearchInputCard } from './SearchInputCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Input: (props: any) => <input data-testid="search-input" {...props} />,
}))

describe('SearchInputCard', () => {
  it('renders the search input', () => {
    render(<SearchInputCard query="" onQueryChange={vi.fn()} />)
    expect(screen.getByTestId('search-input')).toBeDefined()
  })

  it('displays the current query value', () => {
    render(<SearchInputCard query="hello" onQueryChange={vi.fn()} />)
    expect((screen.getByTestId('search-input') as HTMLInputElement).value).toBe('hello')
  })

  it('calls onQueryChange when input changes', () => {
    const onQueryChange = vi.fn()
    render(<SearchInputCard query="" onQueryChange={onQueryChange} />)
    fireEvent.change(screen.getByTestId('search-input'), { target: { value: 'test' } })
    expect(onQueryChange).toHaveBeenCalledWith('test')
  })

  it('uses custom placeholder', () => {
    render(<SearchInputCard query="" onQueryChange={vi.fn()} placeholder="Custom placeholder" />)
    expect(screen.getByPlaceholderText('Custom placeholder')).toBeDefined()
  })

  it('renders with default placeholder', () => {
    render(<SearchInputCard query="" onQueryChange={vi.fn()} />)
    expect(screen.getByPlaceholderText('Search members, training jobs, datasets, knowledge...')).toBeDefined()
  })
})
