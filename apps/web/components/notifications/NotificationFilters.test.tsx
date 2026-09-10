/// <reference types="vitest" />
import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { NotificationFilters } from './NotificationFilters'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, ...props }: any) => <button data-testid="button" onClick={onClick} {...props}>{children}</button>,
  Input: ({ value, onChange, placeholder, ...props }: any) => (
    <input data-testid="filter-input" value={value} onChange={onChange} placeholder={placeholder} {...props} />
  ),
}))

afterEach(() => cleanup())

describe('NotificationFilters', () => {
  it('renders filter input', () => {
    render(<NotificationFilters filter="" onFilterChange={() => {}} onRefresh={() => {}} />)
    expect(screen.getByTestId('filter-input')).toBeDefined()
  })

  it('renders refresh button', () => {
    render(<NotificationFilters filter="" onFilterChange={() => {}} onRefresh={() => {}} />)
    expect(screen.getByText('Refresh')).toBeDefined()
  })

  it('displays current filter value', () => {
    render(<NotificationFilters filter="error" onFilterChange={() => {}} onRefresh={() => {}} />)
    const input = screen.getByTestId('filter-input') as HTMLInputElement
    expect(input.value).toBe('error')
  })

  it('calls onFilterChange when input changes', () => {
    const handleChange = vi.fn()
    render(<NotificationFilters filter="" onFilterChange={handleChange} onRefresh={() => {}} />)
    const input = screen.getByTestId('filter-input')
    fireEvent.change(input, { target: { value: 'test' } })
    expect(handleChange).toHaveBeenCalled()
  })

  it('calls onRefresh when refresh button is clicked', () => {
    const handleRefresh = vi.fn()
    render(<NotificationFilters filter="" onFilterChange={() => {}} onRefresh={handleRefresh} />)
    screen.getByText('Refresh').click()
    expect(handleRefresh).toHaveBeenCalled()
  })

  it('shows placeholder text', () => {
    render(<NotificationFilters filter="" onFilterChange={() => {}} onRefresh={() => {}} />)
    expect(screen.getByPlaceholderText('Filter notifications...')).toBeDefined()
  })
})
