/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { WorkspaceGeneralCard } from './WorkspaceGeneralCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Input: (props: any) => <input data-testid="input" {...props} />,

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

describe('WorkspaceGeneralCard', () => {
  it('renders the title', () => {
    render(
      <WorkspaceGeneralCard name="" description="" onNameChange={vi.fn()} onDescriptionChange={vi.fn()} />,
    )
    expect(screen.getByText('General')).toBeDefined()
  })

  it('renders Name label and input', () => {
    render(
      <WorkspaceGeneralCard name="Test" description="" onNameChange={vi.fn()} onDescriptionChange={vi.fn()} />,
    )
    expect(screen.getByText('Name')).toBeDefined()
    expect(screen.getByDisplayValue('Test')).toBeDefined()
  })

  it('renders Description label and textarea', () => {
    render(
      <WorkspaceGeneralCard name="" description="My desc" onNameChange={vi.fn()} onDescriptionChange={vi.fn()} />,
    )
    expect(screen.getByText('Description')).toBeDefined()
  })

  it('calls onNameChange when name input changes', () => {
    const onNameChange = vi.fn()
    render(
      <WorkspaceGeneralCard name="" description="" onNameChange={onNameChange} onDescriptionChange={vi.fn()} />,
    )
    const inputs = screen.getAllByTestId('input')
    fireEvent.change(inputs[0], { target: { value: 'New Name' } })
    expect(onNameChange).toHaveBeenCalledWith('New Name')
  })

  it('calls onDescriptionChange when textarea changes', () => {
    const onDescriptionChange = vi.fn()
    render(
      <WorkspaceGeneralCard name="" description="" onNameChange={vi.fn()} onDescriptionChange={onDescriptionChange} />,
    )
    const textarea = screen.getByPlaceholderText('Describe this workspace')
    fireEvent.change(textarea, { target: { value: 'New desc' } })
    expect(onDescriptionChange).toHaveBeenCalledWith('New desc')
  })

  it('displays current name value', () => {
    render(
      <WorkspaceGeneralCard name="My Workspace" description="" onNameChange={vi.fn()} onDescriptionChange={vi.fn()} />,
    )
    expect(screen.getByDisplayValue('My Workspace')).toBeDefined()
  })
})
