/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { WorkspaceCreateCard } from './WorkspaceCreateCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => <button onClick={onClick} disabled={disabled} {...props}>{children}</button>,
  Input: (props: any) => <input data-testid="ws-input" {...props} />,

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

describe('WorkspaceCreateCard', () => {
  const defaultProps = {
    name: '',
    description: '',
    onNameChange: vi.fn(),
    onDescriptionChange: vi.fn(),
    onCreate: vi.fn(),
  }

  it('renders the title', () => {
    render(<WorkspaceCreateCard {...defaultProps} />)
    expect(screen.getByText('Create Workspace')).toBeDefined()
  })

  it('renders name and description inputs', () => {
    render(<WorkspaceCreateCard {...defaultProps} />)
    expect(screen.getAllByTestId('ws-input')).toHaveLength(2)
  })

  it('renders the Create button', () => {
    render(<WorkspaceCreateCard {...defaultProps} />)
    expect(screen.getByText('Create')).toBeDefined()
  })

  it('calls onCreate when Create is clicked', () => {
    const onCreate = vi.fn()
    render(<WorkspaceCreateCard {...defaultProps} name="My WS" onCreate={onCreate} />)
    fireEvent.click(screen.getByText('Create'))
    expect(onCreate).toHaveBeenCalledOnce()
  })

  it('calls onNameChange when name input changes', () => {
    const onNameChange = vi.fn()
    render(<WorkspaceCreateCard {...defaultProps} onNameChange={onNameChange} />)
    fireEvent.change(screen.getAllByTestId('ws-input')[0], { target: { value: 'New WS' } })
    expect(onNameChange).toHaveBeenCalledWith('New WS')
  })

  it('calls onDescriptionChange when description input changes', () => {
    const onDescriptionChange = vi.fn()
    render(<WorkspaceCreateCard {...defaultProps} onDescriptionChange={onDescriptionChange} />)
    fireEvent.change(screen.getAllByTestId('ws-input')[1], { target: { value: 'A desc' } })
    expect(onDescriptionChange).toHaveBeenCalledWith('A desc')
  })

  it('disables Create button when name is empty', () => {
    render(<WorkspaceCreateCard {...defaultProps} name="" />)
    expect(screen.getByText('Create')).toHaveProperty('disabled', true)
  })

  it('shows creating state', () => {
    render(<WorkspaceCreateCard {...defaultProps} name="Test" creating />)
    expect(screen.getByText('Creating...')).toBeDefined()
  })
})
