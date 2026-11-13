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

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,
    Separator: () => <hr />,
    Tooltip: ({ children }: any) => <>{children}</>,
    TooltipTrigger: ({ children }: any) => <>{children}</>,
    TooltipContent: ({ children }: any) => <>{children}</>,
    Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
    Avatar: ({ children }: any) => <div>{children}</div>,
    AvatarFallback: ({ children }: any) => <div>{children}</div>,
    ScrollArea: ({ children }: any) => <div>{children}</div>,
    Table: ({ children }: any) => <table>{children}</table>,
    TableBody: ({ children }: any) => <tbody>{children}</tbody>,
    TableRow: ({ children }: any) => <tr>{children}</tr>,
    TableCell: ({ children }: any) => <td>{children}</td>,
    TableHead: ({ children }: any) => <th>{children}</th>,
    TableHeader: ({ children }: any) => <thead>{children}</thead>,
    Collapsible: ({ children }: any) => <div>{children}</div>,
    CollapsibleTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    CollapsibleContent: ({ children }: any) => <div>{children}</div>,
    Toggle: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Command: ({ children }: any) => <div>{children}</div>,
    CommandInput: ({ ...props }: any) => <input {...props} />,
    CommandList: ({ children }: any) => <div>{children}</div>,
    CommandEmpty: ({ children }: any) => <div>{children}</div>,
    CommandGroup: ({ children }: any) => <div>{children}</div>,
    CommandItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
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
