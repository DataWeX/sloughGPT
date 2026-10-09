import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { LoraEvalHistory } from './LoraEvalHistory'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,

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

describe('LoraEvalHistory', () => {
  it('renders the history title with count', () => {
    render(<LoraEvalHistory history={[]} loading={false} />)
    expect(screen.getByText('Eval History (0)')).toBeDefined()
  })

  it('shows loading text when loading', () => {
    render(<LoraEvalHistory history={[]} loading={true} />)
    expect(screen.getByText('Loading...')).toBeDefined()
  })

  it('shows empty state when no history', () => {
    render(<LoraEvalHistory history={[]} loading={false} />)
    expect(screen.getByText('No evaluations yet. Run an eval above.')).toBeDefined()
  })

  it('displays history entries when provided', () => {
    const history = [
      { status: 'passed', elapsed_ms: 1200, report: 'All tests passed' },
      { status: 'failed', elapsed_ms: 800, report: 'Some tests failed' },
    ]
    render(<LoraEvalHistory history={history} loading={false} />)
    expect(screen.getByText('passed')).toBeDefined()
    expect(screen.getByText('failed')).toBeDefined()
    expect(screen.getByText('1200ms')).toBeDefined()
    expect(screen.getByText('800ms')).toBeDefined()
  })

  it('renders report text when provided', () => {
    const history = [{ status: 'ok', report: 'Detailed report summary' }]
    render(<LoraEvalHistory history={history} loading={false} />)
    expect(screen.getByText('Detailed report summary')).toBeDefined()
  })

  it('hides elapsed time when not provided', () => {
    const history = [{ status: 'done' }]
    render(<LoraEvalHistory history={history} loading={false} />)
    expect(screen.getByText('done')).toBeDefined()
    expect(screen.queryByText(/ms/)).toBeNull()
  })
})
