// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,

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

import { AgentTemplateCard } from './AgentTemplateCard'

afterEach(() => cleanup())

const mockTemplates = [
  { name: 'Code Assistant', desc: 'Helps write and review code', instructions: 'Be concise', tools: ['read_file', 'write_file'] },
  { name: 'Research Bot', desc: 'Searches and summarizes info', instructions: 'Be thorough', tools: ['web_search'] },
]

describe('AgentTemplateCard', () => {
  it('renders title', () => {
    render(<AgentTemplateCard templates={[]} onSelect={vi.fn()} />)
    expect(screen.getByText('Agent Templates')).toBeTruthy()
  })

  it('shows empty state', () => {
    render(<AgentTemplateCard templates={[]} onSelect={vi.fn()} />)
    expect(screen.getByText('No templates available.')).toBeTruthy()
  })

  it('renders templates with name and description', () => {
    render(<AgentTemplateCard templates={mockTemplates} onSelect={vi.fn()} />)
    expect(screen.getByText('Code Assistant')).toBeTruthy()
    expect(screen.getByText('Helps write and review code')).toBeTruthy()
    expect(screen.getByText('Research Bot')).toBeTruthy()
    expect(screen.getByText('Searches and summarizes info')).toBeTruthy()
  })

  it('renders tool badges', () => {
    render(<AgentTemplateCard templates={mockTemplates} onSelect={vi.fn()} />)
    expect(screen.getByText('read_file')).toBeTruthy()
    expect(screen.getByText('write_file')).toBeTruthy()
    expect(screen.getByText('web_search')).toBeTruthy()
  })

  it('calls onSelect when button clicked', () => {
    const onSelect = vi.fn()
    render(<AgentTemplateCard templates={mockTemplates} onSelect={onSelect} />)
    const buttons = screen.getAllByRole('button').filter(b => b.textContent === 'Select')
    fireEvent.click(buttons[0])
    expect(onSelect).toHaveBeenCalledWith(mockTemplates[0])
  })
})
