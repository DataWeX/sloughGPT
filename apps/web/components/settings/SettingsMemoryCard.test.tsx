// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardDescription: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  CardFooter: ({ children }: any) => <div>{children}</div>,
  Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
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

import { SettingsMemoryCard } from './SettingsMemoryCard'

afterEach(() => cleanup())

describe('SettingsMemoryCard', () => {
  it('renders title and description', () => {
    render(<SettingsMemoryCard customContext="" onChange={() => {}} />)
    expect(screen.getByText('Memory')).toBeTruthy()
    expect(screen.getByText('Custom instructions included with every prompt')).toBeTruthy()
  })

  it('renders textarea with custom context', () => {
    render(<SettingsMemoryCard customContext="You are helpful" onChange={() => {}} />)
    expect(screen.getByDisplayValue('You are helpful')).toBeTruthy()
  })

  it('renders textarea with empty value', () => {
    render(<SettingsMemoryCard customContext="" onChange={() => {}} />)
    expect(screen.getByDisplayValue('')).toBeTruthy()
  })

  it('calls onChange when textarea updated', () => {
    const onChange = vi.fn()
    render(<SettingsMemoryCard customContext="" onChange={onChange} />)
    fireEvent.change(screen.getByRole('textbox', { name: /custom instructions/i }), { target: { value: 'New text' } })
    expect(onChange).toHaveBeenCalledWith('New text')
  })

  it('has correct aria-label on textarea', () => {
    render(<SettingsMemoryCard customContext="" onChange={() => {}} />)
    expect(screen.getByRole('textbox', { name: /custom instructions/i })).toBeTruthy()
  })
})
