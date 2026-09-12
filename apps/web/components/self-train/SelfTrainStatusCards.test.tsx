// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, within } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
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

import { SelfTrainStatusCards } from './SelfTrainStatusCards'

describe('SelfTrainStatusCards', () => {
  afterEach(() => cleanup())

  it('renders title labels', () => {
    render(<SelfTrainStatusCards />)
    expect(screen.getByText('Status')).toBeTruthy()
    expect(screen.getByText('PID')).toBeTruthy()
    expect(screen.getByText('Exit code')).toBeTruthy()
    expect(screen.getByText('History lines')).toBeTruthy()
  })

  it('shows "Not started" when no status is provided', () => {
    render(<SelfTrainStatusCards />)
    expect(screen.getByText('Not started')).toBeTruthy()
  })

  it('shows "Running" when status is running', () => {
    render(<SelfTrainStatusCards status="running" />)
    expect(screen.getByText('Running')).toBeTruthy()
  })

  it('shows "Exited" when status is exited', () => {
    render(<SelfTrainStatusCards status="exited" />)
    expect(screen.getByText('Exited')).toBeTruthy()
  })

  it('shows loading state when loading is true', () => {
    render(<SelfTrainStatusCards loading />)
    const cards = screen.getAllByTestId('card')
    expect(within(cards[0]).getByText('...')).toBeTruthy()
  })

  it('renders pid and returncode when provided', () => {
    render(<SelfTrainStatusCards pid={12345} returncode={42} historyCount={5} />)
    expect(screen.getByText('12345')).toBeTruthy()
    expect(screen.getByText('42')).toBeTruthy()
  })

  it('renders history count', () => {
    render(<SelfTrainStatusCards historyCount={42} />)
    expect(screen.getByText('42')).toBeTruthy()
  })

  it('renders four cards', () => {
    const { container } = render(<SelfTrainStatusCards />)
    const cards = container.querySelectorAll('[data-testid="card"]')
    expect(cards).toHaveLength(4)
  })
})
