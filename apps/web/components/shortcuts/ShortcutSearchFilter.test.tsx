/// <reference types="vitest" />
import { render, screen, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { ShortcutSearchFilter } from './ShortcutSearchFilter'

vi.mock('@sloughgpt/strui', () => ({
  Input: (props: React.InputHTMLAttributes<HTMLInputElement>) => <input {...props} />,

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

describe('ShortcutSearchFilter', () => {
  it('renders the search input', () => {
    render(<ShortcutSearchFilter categories={['Global', 'Chat']} onFilter={vi.fn()} />)
    expect(screen.getByLabelText('Filter shortcuts by category')).toBeDefined()
  })

  it('renders with a placeholder', () => {
    render(<ShortcutSearchFilter categories={[]} onFilter={vi.fn()} />)
    expect(screen.getByPlaceholderText('Filter categories...')).toBeDefined()
  })

  it('calls onFilter with all categories when query is empty', async () => {
    const onFilter = vi.fn()
    render(<ShortcutSearchFilter categories={['Global', 'Chat', 'Nav']} onFilter={onFilter} />)
    const input = screen.getByLabelText('Filter shortcuts by category')
    await userEvent.type(input, 'X')
    await userEvent.clear(input)
    expect(onFilter).toHaveBeenLastCalledWith(['Global', 'Chat', 'Nav'])
  })

  it('filters categories by query', async () => {
    const onFilter = vi.fn()
    render(<ShortcutSearchFilter categories={['Global', 'Chat', 'Navigation']} onFilter={onFilter} />)
    const input = screen.getByLabelText('Filter shortcuts by category')
    await userEvent.type(input, 'cha')
    expect(onFilter).toHaveBeenLastCalledWith(['Chat'])
  })

  it('returns empty array when no categories match', async () => {
    const onFilter = vi.fn()
    render(<ShortcutSearchFilter categories={['Global', 'Chat']} onFilter={onFilter} />)
    const input = screen.getByLabelText('Filter shortcuts by category')
    await userEvent.type(input, 'zzz')
    expect(onFilter).toHaveBeenLastCalledWith([])
  })
})
