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
  Input: ({ ...props }: any) => <input {...props} />,

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

import { AgentExecutionForm } from './AgentExecutionForm'

afterEach(() => cleanup())

const defaultProps = {
  agentName: 'Researcher',
  prompt: '',
  result: null,
  toolsUsed: [],
  running: false,
  errors: {},
  onPromptChange: vi.fn(),
  onExecute: vi.fn(),
  onClose: vi.fn(),
}

describe('AgentExecutionForm', () => {
  it('renders the form container', () => {
    render(<AgentExecutionForm {...defaultProps} />)
    expect(screen.getByTestId('agent-execution-form')).toBeTruthy()
  })

  it('renders Execute and Close buttons', () => {
    render(<AgentExecutionForm {...defaultProps} />)
    expect(screen.getByText('Execute')).toBeTruthy()
    expect(screen.getByText('Close')).toBeTruthy()
  })

  it('calls onExecute when Execute clicked', () => {
    const onExecute = vi.fn()
    render(<AgentExecutionForm {...defaultProps} prompt="hello" onExecute={onExecute} />)
    fireEvent.click(screen.getByText('Execute'))
    expect(onExecute).toHaveBeenCalled()
  })

  it('calls onClose when Close clicked', () => {
    const onClose = vi.fn()
    render(<AgentExecutionForm {...defaultProps} onClose={onClose} />)
    fireEvent.click(screen.getByText('Close'))
    expect(onClose).toHaveBeenCalled()
  })

  it('disables Execute when prompt is empty', () => {
    render(<AgentExecutionForm {...defaultProps} prompt="" />)
    expect(screen.getByText('Execute').hasAttribute('disabled')).toBe(true)
  })

  it('shows Running... when executing', () => {
    render(<AgentExecutionForm {...defaultProps} running={true} prompt="hi" />)
    expect(screen.getByText('Running...')).toBeTruthy()
    expect(screen.getByText('Running...').hasAttribute('disabled')).toBe(true)
  })

  it('displays result when provided', () => {
    render(<AgentExecutionForm {...defaultProps} result="Done!" />)
    expect(screen.getByText('Response')).toBeTruthy()
    expect(screen.getByText('Done!')).toBeTruthy()
  })

  it('displays tools used', () => {
    render(
      <AgentExecutionForm
        {...defaultProps}
        toolsUsed={[{ tool: 'web_search', result: null }]}
        result="Ok"
      />,
    )
    expect(screen.getByText('web search')).toBeTruthy()
  })

  it('shows validation error', () => {
    render(<AgentExecutionForm {...defaultProps} errors={{ prompt: 'Prompt required' }} />)
    expect(screen.getByText('Prompt required')).toBeTruthy()
    expect(screen.getByRole('alert')).toBeTruthy()
  })
})
