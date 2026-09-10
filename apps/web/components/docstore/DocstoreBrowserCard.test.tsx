// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: ({ value, onChange, ...props }: any) => (
    <input value={value} onChange={onChange} {...props} />
  ),
}))

import { DocstoreBrowserCard } from './DocstoreBrowserCard'

afterEach(() => cleanup())

const defaultProps = {
  collections: ['default', 'code'],
  selected: 'default',
  docs: [{ _id: 'doc-1' }, { _id: 'doc-2' }],
  page: 1,
  totalPages: 3,
  loading: false,
  searchQuery: '',
  onSelect: vi.fn(),
  onSearchChange: vi.fn(),
  onPageChange: vi.fn(),
  onCreateToggle: vi.fn(),
  showCreate: false,
}

describe('DocstoreBrowserCard', () => {
  it('renders the title', () => {
    render(<DocstoreBrowserCard {...defaultProps} />)
    expect(screen.getByText('Documents')).toBeTruthy()
  })

  it('renders collection tabs', () => {
    render(<DocstoreBrowserCard {...defaultProps} />)
    expect(screen.getByTestId('collection-default')).toBeTruthy()
    expect(screen.getByTestId('collection-code')).toBeTruthy()
    expect(screen.getByText('default')).toBeTruthy()
    expect(screen.getByText('code')).toBeTruthy()
  })

  it('renders New doc button', () => {
    render(<DocstoreBrowserCard {...defaultProps} />)
    expect(screen.getByTestId('new-doc-btn')).toBeTruthy()
    expect(screen.getByText('New doc')).toBeTruthy()
  })

  it('shows document list when docs provided', () => {
    render(<DocstoreBrowserCard {...defaultProps} />)
    expect(screen.getByTestId('doc-list')).toBeTruthy()
    expect(screen.getByText('doc-1')).toBeTruthy()
    expect(screen.getByText('doc-2')).toBeTruthy()
  })

  it('calls onSelect when a doc is clicked', () => {
    const onSelect = vi.fn()
    render(<DocstoreBrowserCard {...defaultProps} onSelect={onSelect} />)
    fireEvent.click(screen.getByTestId('doc-item-doc-1'))
    expect(onSelect).toHaveBeenCalledWith('doc-1')
  })

  it('calls onSearchChange when typing', () => {
    const onSearchChange = vi.fn()
    render(<DocstoreBrowserCard {...defaultProps} onSearchChange={onSearchChange} />)
    fireEvent.change(screen.getByTestId('doc-search'), { target: { value: 'test' } })
    expect(onSearchChange).toHaveBeenCalledWith('test')
  })

  it('calls onCreateToggle when New doc clicked', () => {
    const onCreateToggle = vi.fn()
    render(<DocstoreBrowserCard {...defaultProps} onCreateToggle={onCreateToggle} />)
    fireEvent.click(screen.getByTestId('new-doc-btn'))
    expect(onCreateToggle).toHaveBeenCalled()
  })

  it('shows pagination when totalPages > 1', () => {
    render(<DocstoreBrowserCard {...defaultProps} />)
    expect(screen.getByTestId('pagination')).toBeTruthy()
    expect(screen.getByText('1 / 3')).toBeTruthy()
  })

  it('disables Prev on first page', () => {
    render(<DocstoreBrowserCard {...defaultProps} page={1} />)
    const prevBtn = screen.getByText('Prev')
    expect(prevBtn).toHaveProperty('disabled', true)
  })

  it('disables Next on last page', () => {
    render(<DocstoreBrowserCard {...defaultProps} page={3} totalPages={3} />)
    const nextBtn = screen.getByText('Next')
    expect(nextBtn).toHaveProperty('disabled', true)
  })

  it('calls onPageChange when navigating', () => {
    const onPageChange = vi.fn()
    render(<DocstoreBrowserCard {...defaultProps} onPageChange={onPageChange} />)
    fireEvent.click(screen.getByText('Next'))
    expect(onPageChange).toHaveBeenCalledWith(2)
  })

  it('shows loading state', () => {
    render(<DocstoreBrowserCard {...defaultProps} loading={true} docs={[]} />)
    expect(screen.getByTestId('loading-indicator')).toBeTruthy()
    expect(screen.getByText('Loading...')).toBeTruthy()
  })

  it('shows empty state when no docs', () => {
    render(<DocstoreBrowserCard {...defaultProps} docs={[]} />)
    expect(screen.getByText('No documents found.')).toBeTruthy()
  })

  it('shows Cancel when showCreate is true', () => {
    render(<DocstoreBrowserCard {...defaultProps} showCreate={true} />)
    expect(screen.getByText('Cancel')).toBeTruthy()
  })
})
