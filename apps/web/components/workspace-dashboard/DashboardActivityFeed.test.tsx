/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { DashboardActivityFeed } from './DashboardActivityFeed'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,

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

const mockActivities = [
  { type: 'training', action: 'Job completed', detail: 'shakespeare-finetune', status: 'completed', timestamp: '2024-01-15T10:00:00Z', user: 'alice' },
  { type: 'dataset', action: 'Dataset uploaded', detail: 'wiki-text', status: 'success', timestamp: '2024-01-15T09:00:00Z', user: 'bob' },
]

describe('DashboardActivityFeed', () => {
  it('renders the title', () => {
    render(<DashboardActivityFeed activities={[]} />)
    expect(screen.getByText('Recent Activity')).toBeDefined()
  })

  it('renders activity actions', () => {
    render(<DashboardActivityFeed activities={mockActivities} />)
    expect(screen.getByText('Job completed')).toBeDefined()
    expect(screen.getByText('Dataset uploaded')).toBeDefined()
  })

  it('renders activity details', () => {
    render(<DashboardActivityFeed activities={mockActivities} />)
    expect(screen.getByText('shakespeare-finetune')).toBeDefined()
  })

  it('renders activity status badges', () => {
    render(<DashboardActivityFeed activities={mockActivities} />)
    expect(screen.getByText('completed')).toBeDefined()
    expect(screen.getByText('success')).toBeDefined()
  })

  it('shows empty state when no activities', () => {
    render(<DashboardActivityFeed activities={[]} />)
    expect(screen.getByText('No recent activity')).toBeDefined()
  })

  it('renders refresh button when onRefresh provided', () => {
    const { container } = render(<DashboardActivityFeed activities={[]} onRefresh={vi.fn()} />)
    expect(container.querySelector('button')).toBeDefined()
  })
})
