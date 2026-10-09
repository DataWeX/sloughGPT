import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ShortcutSearchInput } from './ShortcutSearchInput'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Input: (props: any) => <input data-testid="search-input" {...props} />,

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

vi.mock('lucide-react', () => ({
  Search: () => <span data-testid="search-icon" />,
}))

describe('ShortcutSearchInput', () => {
  it('renders the search input', () => {
    render(<ShortcutSearchInput value="" onChange={() => {}} />)
    expect(screen.getByTestId('search-input')).toBeDefined()
  })

  it('displays the current value', () => {
    render(<ShortcutSearchInput value="test query" onChange={() => {}} />)
    expect((screen.getByTestId('search-input') as HTMLInputElement).value).toBe('test query')
  })

  it('calls onChange when typing', async () => {
    const onChange = vi.fn()
    render(<ShortcutSearchInput value="" onChange={onChange} />)
    await userEvent.type(screen.getByTestId('search-input'), 'a')
    expect(onChange).toHaveBeenCalledWith('a')
  })

  it('renders the search icon', () => {
    render(<ShortcutSearchInput value="" onChange={() => {}} />)
    expect(screen.getByTestId('search-icon')).toBeDefined()
  })

  it('has the correct placeholder', () => {
    render(<ShortcutSearchInput value="" onChange={() => {}} />)
    expect(screen.getByTestId('search-input').getAttribute('placeholder')).toBe('Search shortcuts...')
  })

  it('accepts custom placeholder', () => {
    render(<ShortcutSearchInput value="" onChange={() => {}} placeholder="Custom" />)
    expect(screen.getByTestId('search-input').getAttribute('placeholder')).toBe('Custom')
  })
})
