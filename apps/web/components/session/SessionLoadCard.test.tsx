// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  Button: ({ children, ...props }: any) => <button data-testid="button" {...props}>{children}</button>,
  Input: (props: any) => <input data-testid="input" {...props} />,
  Label: ({ children, ...props }: any) => <label data-testid="label" {...props}>{children}</label>,

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

import { SessionLoadCard } from './SessionLoadCard'

const defaultProps = {
  sessionId: '',
  onSessionIdChange: vi.fn(),
  onInspect: vi.fn(),
  onRegenerate: vi.fn(),
}

describe('SessionLoadCard', () => {
  afterEach(() => cleanup())

  it('renders title "Load Session"', () => {
    render(<SessionLoadCard {...defaultProps} />)
    expect(screen.getByText('Load Session')).toBeTruthy()
  })

  it('renders session ID input', () => {
    render(<SessionLoadCard {...defaultProps} />)
    const input = screen.getByTestId('input')
    expect(input).toBeTruthy()
    expect(input).toHaveAttribute('placeholder', 'Enter session ID...')
  })

  it('renders Inspect button', () => {
    render(<SessionLoadCard {...defaultProps} />)
    expect(screen.getByText('Inspect')).toBeTruthy()
  })

  it('renders Regenerate button', () => {
    render(<SessionLoadCard {...defaultProps} />)
    expect(screen.getByText('Regenerate Last Response')).toBeTruthy()
  })

  it('disables Inspect when loading is true', () => {
    render(<SessionLoadCard {...defaultProps} loading />)
    const buttons = screen.getAllByTestId('button')
    expect(buttons[0]).toBeDisabled()
  })

  it('shows "Loading..." text when loading', () => {
    render(<SessionLoadCard {...defaultProps} loading />)
    expect(screen.getByText('Loading...')).toBeTruthy()
  })

  it('shows "Regenerating..." text when regenerating', () => {
    render(<SessionLoadCard {...defaultProps} regenerating />)
    expect(screen.getByText('Regenerating...')).toBeTruthy()
  })

  it('disables buttons when sessionId is empty', () => {
    render(<SessionLoadCard {...defaultProps} sessionId="" />)
    const buttons = screen.getAllByTestId('button')
    buttons.forEach(btn => expect(btn).toBeDisabled())
  })
})
