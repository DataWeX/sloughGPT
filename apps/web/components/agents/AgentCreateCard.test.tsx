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
  Checkbox: ({ checked, onCheckedChange, ...props }: any) => (
    <input
      type="checkbox"
      checked={!!checked}
      onChange={() => onCheckedChange?.(!checked)}
      {...props}
    />
  ),

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

import { AgentCreateCard } from './AgentCreateCard'

afterEach(() => cleanup())

describe('AgentCreateCard', () => {
  const defaultProps = {
    name: '',
    instructions: '',
    tools: [],
    availableTools: ['read_file', 'write_file', 'web_search'],
    loading: false,
    onNameChange: vi.fn(),
    onInstructionsChange: vi.fn(),
    onToolsChange: vi.fn(),
    onCreate: vi.fn(),
  }

  it('renders title and form elements', () => {
    render(<AgentCreateCard {...defaultProps} />)
    expect(screen.getByText('Create Agent')).toBeTruthy()
    expect(screen.getByTestId('agent-name')).toBeTruthy()
    expect(screen.getByTestId('agent-instructions')).toBeTruthy()
    expect(screen.getByText('read_file')).toBeTruthy()
    expect(screen.getByText('write_file')).toBeTruthy()
    expect(screen.getByText('web_search')).toBeTruthy()
  })

  it('disables create when name empty', () => {
    render(<AgentCreateCard {...defaultProps} />)
    expect(screen.getByTestId('agent-create-btn').hasAttribute('disabled')).toBe(true)
  })

  it('enables create when name entered', () => {
    render(<AgentCreateCard {...defaultProps} name="My Agent" />)
    expect(screen.getByTestId('agent-create-btn').hasAttribute('disabled')).toBe(false)
  })

  it('calls onNameChange on input', () => {
    const onNameChange = vi.fn()
    render(<AgentCreateCard {...defaultProps} onNameChange={onNameChange} />)
    fireEvent.change(screen.getByTestId('agent-name'), { target: { value: 'Test' } })
    expect(onNameChange).toHaveBeenCalledWith('Test')
  })

  it('calls onInstructionsChange on textarea', () => {
    const onInstructionsChange = vi.fn()
    render(<AgentCreateCard {...defaultProps} onInstructionsChange={onInstructionsChange} />)
    fireEvent.change(screen.getByTestId('agent-instructions'), { target: { value: 'Be helpful' } })
    expect(onInstructionsChange).toHaveBeenCalledWith('Be helpful')
  })

  it('calls onCreate when button clicked', () => {
    const onCreate = vi.fn()
    render(<AgentCreateCard {...defaultProps} name="Agent" onCreate={onCreate} />)
    fireEvent.click(screen.getByTestId('agent-create-btn'))
    expect(onCreate).toHaveBeenCalled()
  })

  it('shows loading state', () => {
    render(<AgentCreateCard {...defaultProps} name="Agent" loading={true} />)
    expect(screen.getByText('Creating...')).toBeTruthy()
    expect(screen.getByTestId('agent-create-btn').hasAttribute('disabled')).toBe(true)
  })

  it('calls onToolsChange when checkbox toggled', () => {
    const onToolsChange = vi.fn()
    render(<AgentCreateCard {...defaultProps} onToolsChange={onToolsChange} />)
    const checkboxes = screen.getAllByRole('checkbox')
    fireEvent.click(checkboxes[0])
    expect(onToolsChange).toHaveBeenCalledWith(['read_file'])
  })
})
