/// <reference types="vitest" />
import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { NotificationFeed } from './NotificationFeed'

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

afterEach(() => cleanup())

const mockNotifications = [
  { type: 'training', title: 'Model trained', detail: 'GPT-4 finished fine-tuning', status: 'completed', timestamp: '2025-01-15T10:30:00Z' },
  { type: 'member', title: 'New member', detail: 'alice joined workspace', status: '', timestamp: '2025-01-15T11:00:00Z' },
]

describe('NotificationFeed', () => {
  it('renders title "Recent Events"', () => {
    render(<NotificationFeed notifications={[]} />)
    expect(screen.getByText('Recent Events')).toBeDefined()
  })

  it('shows empty message when no notifications', () => {
    render(<NotificationFeed notifications={[]} />)
    expect(screen.getByText('No notifications')).toBeDefined()
  })

  it('renders notification titles', () => {
    render(<NotificationFeed notifications={mockNotifications} />)
    expect(screen.getByText('Model trained')).toBeDefined()
    expect(screen.getByText('New member')).toBeDefined()
  })

  it('renders notification details', () => {
    render(<NotificationFeed notifications={mockNotifications} />)
    expect(screen.getByText('GPT-4 finished fine-tuning')).toBeDefined()
    expect(screen.getByText('alice joined workspace')).toBeDefined()
  })

  it('renders notification types', () => {
    render(<NotificationFeed notifications={mockNotifications} />)
    expect(screen.getByText('training')).toBeDefined()
    expect(screen.getByText('member')).toBeDefined()
  })

  it('renders status badges when present', () => {
    render(<NotificationFeed notifications={mockNotifications} />)
    expect(screen.getByText('completed')).toBeDefined()
  })
})
