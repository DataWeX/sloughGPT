/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { WorkspaceLimitsCard } from './WorkspaceLimitsCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => <button onClick={onClick} disabled={disabled} {...props}>{children}</button>,
  Input: (props: any) => <input data-testid="limits-input" {...props} />,

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

describe('WorkspaceLimitsCard', () => {
  const defaultProps = {
    defaultModel: 'llama-3.2-3b',
    dataRetentionDays: 90,
    maxMembers: 50,
    allowSharing: true,
    onDefaultModelChange: vi.fn(),
    onDataRetentionChange: vi.fn(),
    onMaxMembersChange: vi.fn(),
    onAllowSharingChange: vi.fn(),
  }

  it('renders the Defaults title', () => {
    render(<WorkspaceLimitsCard {...defaultProps} />)
    expect(screen.getByText('Defaults')).toBeDefined()
  })

  it('renders the Limits title', () => {
    render(<WorkspaceLimitsCard {...defaultProps} />)
    expect(screen.getByText('Limits')).toBeDefined()
  })

  it('renders the Sharing title', () => {
    render(<WorkspaceLimitsCard {...defaultProps} />)
    expect(screen.getByText('Sharing')).toBeDefined()
  })

  it('renders default model input', () => {
    render(<WorkspaceLimitsCard {...defaultProps} />)
    expect(screen.getByDisplayValue('llama-3.2-3b')).toBeDefined()
  })

  it('renders data retention input', () => {
    render(<WorkspaceLimitsCard {...defaultProps} />)
    const inputs = screen.getAllByTestId('limits-input')
    expect(inputs.some((el) => (el as HTMLInputElement).value === '90')).toBe(true)
  })

  it('renders allow sharing checkbox', () => {
    render(<WorkspaceLimitsCard {...defaultProps} />)
    expect(screen.getByText('Allow data sharing')).toBeDefined()
  })

  it('calls onAllowSharingChange when checkbox toggled', () => {
    const onAllowSharingChange = vi.fn()
    render(<WorkspaceLimitsCard {...defaultProps} onAllowSharingChange={onAllowSharingChange} />)
    fireEvent.click(screen.getByText('Allow data sharing').closest('label')!)
    expect(onAllowSharingChange).toHaveBeenCalledOnce()
  })

  it('renders Save button when onSave provided', () => {
    render(<WorkspaceLimitsCard {...defaultProps} onSave={vi.fn()} />)
    expect(screen.getByText('Save Settings')).toBeDefined()
  })

  it('shows unsaved changes when hasChanges is true', () => {
    render(<WorkspaceLimitsCard {...defaultProps} onSave={vi.fn()} hasChanges />)
    expect(screen.getByText('You have unsaved changes')).toBeDefined()
  })
})
