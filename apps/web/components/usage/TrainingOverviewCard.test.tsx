import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { TrainingOverviewCard } from './TrainingOverviewCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,

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

const defaultProps = {
  completed: 10,
  running: 3,
  queued: 5,
  failed: 1,
  totalMinutes: 120,
}

describe('TrainingOverviewCard', () => {
  it('renders the card title', () => {
    render(<TrainingOverviewCard {...defaultProps} />)
    expect(screen.getByText('Training Jobs')).toBeDefined()
  })

  it('renders completed count', () => {
    render(<TrainingOverviewCard {...defaultProps} />)
    expect(screen.getByText('10')).toBeDefined()
    expect(screen.getByText('Completed')).toBeDefined()
  })

  it('renders running count', () => {
    render(<TrainingOverviewCard {...defaultProps} />)
    expect(screen.getByText('3')).toBeDefined()
    expect(screen.getByText('Running')).toBeDefined()
  })

  it('renders queued count', () => {
    render(<TrainingOverviewCard {...defaultProps} />)
    expect(screen.getByText('5')).toBeDefined()
    expect(screen.getByText('Queued')).toBeDefined()
  })

  it('renders failed count', () => {
    render(<TrainingOverviewCard {...defaultProps} />)
    expect(screen.getByText('1')).toBeDefined()
    expect(screen.getByText('Failed')).toBeDefined()
  })

  it('renders total training minutes', () => {
    render(<TrainingOverviewCard {...defaultProps} />)
    expect(screen.getByText('Total training time: 120 minutes')).toBeDefined()
  })

  it('renders all stat values', () => {
    render(<TrainingOverviewCard {...defaultProps} />)
    expect(screen.getByText('10')).toBeDefined()
    expect(screen.getByText('3')).toBeDefined()
    expect(screen.getByText('5')).toBeDefined()
    expect(screen.getByText('1')).toBeDefined()
  })
})
