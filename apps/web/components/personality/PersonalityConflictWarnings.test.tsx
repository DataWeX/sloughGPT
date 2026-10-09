/// <reference types="vitest" />
import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { PersonalityConflictWarnings } from './PersonalityConflictWarnings'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardDescription: ({ children, ...props }: any) => <div data-testid="card-description" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
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

afterEach(() => cleanup())

const mockConflicts = [
  { type: 'voice', severity: 'medium', message: 'High formality conflicts with humor', fields: ['formality', 'humor'] },
  { type: 'trait', severity: 'low', message: 'Low openness may limit creativity', fields: ['openness'] },
]

describe('PersonalityConflictWarnings', () => {
  it('renders nothing when no conflicts', () => {
    const { container } = render(<PersonalityConflictWarnings conflicts={[]} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders title when conflicts exist', () => {
    render(<PersonalityConflictWarnings conflicts={mockConflicts} />)
    expect(screen.getByText('Personality Conflicts')).toBeDefined()
  })

  it('renders conflict messages', () => {
    render(<PersonalityConflictWarnings conflicts={mockConflicts} />)
    expect(screen.getByText('High formality conflicts with humor')).toBeDefined()
    expect(screen.getByText('Low openness may limit creativity')).toBeDefined()
  })

  it('renders severity badges', () => {
    render(<PersonalityConflictWarnings conflicts={mockConflicts} />)
    const badges = screen.getAllByTestId('badge')
    expect(badges).toHaveLength(2)
    expect(screen.getByText('medium')).toBeDefined()
    expect(screen.getByText('low')).toBeDefined()
  })

  it('renders affected fields', () => {
    render(<PersonalityConflictWarnings conflicts={mockConflicts} />)
    expect(screen.getByText('formality + humor')).toBeDefined()
    expect(screen.getByText('openness')).toBeDefined()
  })

  it('renders description text', () => {
    render(<PersonalityConflictWarnings conflicts={mockConflicts} />)
    expect(screen.getByText('These settings may work against each other')).toBeDefined()
  })
})
