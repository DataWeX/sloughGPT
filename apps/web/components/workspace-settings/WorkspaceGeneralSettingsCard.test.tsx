/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { WorkspaceGeneralSettingsCard } from './WorkspaceGeneralSettingsCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => <button onClick={onClick} disabled={disabled} {...props}>{children}</button>,
  Input: (props: any) => <input data-testid="settings-input" {...props} />,

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

describe('WorkspaceGeneralSettingsCard', () => {
  const defaultProps = {
    name: 'My Workspace',
    description: 'A test workspace',
    onNameChange: vi.fn(),
    onDescriptionChange: vi.fn(),
  }

  it('renders the title', () => {
    render(<WorkspaceGeneralSettingsCard {...defaultProps} />)
    expect(screen.getByText('General')).toBeDefined()
  })

  it('renders name input with value', () => {
    render(<WorkspaceGeneralSettingsCard {...defaultProps} />)
    expect((screen.getByTestId('settings-input') as HTMLInputElement).value).toBe('My Workspace')
  })

  it('renders description textarea', () => {
    render(<WorkspaceGeneralSettingsCard {...defaultProps} />)
    expect(screen.getByDisplayValue('A test workspace')).toBeDefined()
  })

  it('calls onNameChange when name input changes', () => {
    const onNameChange = vi.fn()
    render(<WorkspaceGeneralSettingsCard {...defaultProps} onNameChange={onNameChange} />)
    fireEvent.change(screen.getByTestId('settings-input'), { target: { value: 'New Name' } })
    expect(onNameChange).toHaveBeenCalledWith('New Name')
  })

  it('calls onDescriptionChange when description changes', () => {
    const onDescriptionChange = vi.fn()
    render(<WorkspaceGeneralSettingsCard {...defaultProps} onDescriptionChange={onDescriptionChange} />)
    fireEvent.change(screen.getByDisplayValue('A test workspace'), { target: { value: 'New desc' } })
    expect(onDescriptionChange).toHaveBeenCalledWith('New desc')
  })

  it('renders Save button when onSave provided', () => {
    render(<WorkspaceGeneralSettingsCard {...defaultProps} onSave={vi.fn()} />)
    expect(screen.getByText('Save')).toBeDefined()
  })

  it('shows unsaved changes text when hasChanges is true', () => {
    render(<WorkspaceGeneralSettingsCard {...defaultProps} onSave={vi.fn()} hasChanges />)
    expect(screen.getByText('Unsaved changes')).toBeDefined()
  })

  it('disables Save when no changes', () => {
    render(<WorkspaceGeneralSettingsCard {...defaultProps} onSave={vi.fn()} hasChanges={false} />)
    expect(screen.getByText('Save')).toHaveProperty('disabled', true)
  })
})
