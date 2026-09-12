/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { SearchCategoryBadge } from './SearchCategoryBadge'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Badge: ({ children, ...props }: any) => <span data-testid="badge" {...props}>{children}</span>,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
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

describe('SearchCategoryBadge', () => {
  it('renders the label', () => {
    render(<SearchCategoryBadge type="member" label="Members" />)
    expect(screen.getByText('Members')).toBeDefined()
  })

  it('renders count when provided', () => {
    render(<SearchCategoryBadge type="training" label="Training Jobs" count={5} />)
    expect(screen.getByText('(5)')).toBeDefined()
  })

  it('renders without count when not provided', () => {
    const { container } = render(<SearchCategoryBadge type="dataset" label="Datasets" />)
    expect(container.textContent).toBe('Datasets')
  })

  it('applies default color for known types', () => {
    const { container } = render(<SearchCategoryBadge type="member" label="Members" />)
    const badge = container.firstChild as HTMLElement
    expect(badge.className).toContain('bg-purple-100')
  })

  it('applies custom color class when provided', () => {
    const { container } = render(<SearchCategoryBadge type="custom" label="Custom" colorClass="bg-red-500 text-white" />)
    const badge = container.firstChild as HTMLElement
    expect(badge.className).toContain('bg-red-500')
  })

  it('falls back to muted color for unknown types', () => {
    const { container } = render(<SearchCategoryBadge type="unknown" label="Unknown" />)
    const badge = container.firstChild as HTMLElement
    expect(badge.className).toContain('bg-muted')
  })
})
