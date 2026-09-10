/// <reference types="vitest" />
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { VectorAddEntriesCard } from './VectorAddEntriesCard'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => <h3 data-testid="card-title" {...props}>{children}</h3>,
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button data-testid="button" onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
}))

afterEach(() => cleanup())

describe('VectorAddEntriesCard', () => {
  const defaultProps = {
    text: '',
    loading: false,
    onTextChange: vi.fn(),
    onAdd: vi.fn(),
  }

  it('renders the card title', () => {
    render(<VectorAddEntriesCard {...defaultProps} />)
    expect(screen.getByText('Add Entries')).toBeDefined()
  })

  it('renders textarea', () => {
    render(<VectorAddEntriesCard {...defaultProps} />)
    expect(screen.getByLabelText('Vector store entries')).toBeDefined()
  })

  it('renders add button', () => {
    render(<VectorAddEntriesCard {...defaultProps} />)
    expect(screen.getByText('Add entries')).toBeDefined()
  })

  it('calls onAdd when button is clicked', () => {
    render(<VectorAddEntriesCard {...defaultProps} text="line1\nline2" />)
    fireEvent.click(screen.getByText('Add entries'))
    expect(defaultProps.onAdd).toHaveBeenCalledOnce()
  })

  it('shows Adding... when loading', () => {
    render(<VectorAddEntriesCard {...defaultProps} loading text="data" />)
    expect(screen.getByText('Adding...')).toBeDefined()
  })

  it('displays entry count', () => {
    render(<VectorAddEntriesCard {...defaultProps} text={'line1\nline2\nline3'} />)
    expect(screen.getByText('3 entries')).toBeDefined()
  })
})
