/// <reference types="vitest" />
// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { VMTrainingConfigForm } from './VMTrainingConfigForm'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, ...props }: any) => (
    <button onClick={onClick} {...props}>{children}</button>
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

describe('VMTrainingConfigForm', () => {
  const defaultConfig = {
    dataset: 'shakespeare',
    epochs: 1,
    lr: 0.001,
    batch_size: 32,
    n_layer: 4,
    n_head: 4,
    embed_dim: 128,
  }

  const defaultProps = {
    config: defaultConfig,
    onConfigChange: vi.fn(),
    datasetNames: ['shakespeare', 'openwebtext'],
    role: 'user',
    onRoleChange: vi.fn(),
    hints: [],
    onReset: vi.fn(),
    onLoadSample: vi.fn(),
    onLaunch: vi.fn(),
    launchedJob: null,
    onDismissJob: vi.fn(),
  }

  it('renders the title', () => {
    render(<VMTrainingConfigForm {...defaultProps} />)
    expect(screen.getByText('Training launch')).toBeDefined()
  })

  it('renders Reset config button', () => {
    render(<VMTrainingConfigForm {...defaultProps} />)
    expect(screen.getByText('Reset config')).toBeDefined()
  })

  it('calls onReset when Reset config is clicked', () => {
    const onReset = vi.fn()
    render(<VMTrainingConfigForm {...defaultProps} onReset={onReset} />)
    fireEvent.click(screen.getByText('Reset config'))
    expect(onReset).toHaveBeenCalledOnce()
  })

  it('renders dataset selector with available datasets', () => {
    render(<VMTrainingConfigForm {...defaultProps} />)
    expect(screen.getByText('shakespeare')).toBeDefined()
    expect(screen.getByText('openwebtext')).toBeDefined()
    expect(screen.getByText('Custom…')).toBeDefined()
  })

  it('renders all config field labels', () => {
    render(<VMTrainingConfigForm {...defaultProps} />)
    expect(screen.getByText('Epochs')).toBeDefined()
    expect(screen.getByText('Learning rate')).toBeDefined()
    expect(screen.getByText('Batch size')).toBeDefined()
    expect(screen.getByText('Layers')).toBeDefined()
    expect(screen.getByText('Heads')).toBeDefined()
    expect(screen.getByText('Embed size')).toBeDefined()
  })

  it('renders Launch training and Load sample buttons', () => {
    render(<VMTrainingConfigForm {...defaultProps} />)
    expect(screen.getByText('Launch training')).toBeDefined()
    expect(screen.getByText('Load sample')).toBeDefined()
  })

  it('calls onLaunch when Launch training is clicked', () => {
    const onLaunch = vi.fn()
    render(<VMTrainingConfigForm {...defaultProps} onLaunch={onLaunch} />)
    fireEvent.click(screen.getByText('Launch training'))
    expect(onLaunch).toHaveBeenCalledOnce()
  })

  it('calls onLoadSample when Load sample is clicked', () => {
    const onLoadSample = vi.fn()
    render(<VMTrainingConfigForm {...defaultProps} onLoadSample={onLoadSample} />)
    fireEvent.click(screen.getByText('Load sample'))
    expect(onLoadSample).toHaveBeenCalledOnce()
  })

  it('shows role warning when role is user', () => {
    render(<VMTrainingConfigForm {...defaultProps} role="user" />)
    expect(screen.getByText(/Training is denied for the user role/)).toBeDefined()
    expect(screen.getByText('Switch to admin')).toBeDefined()
  })

  it('does not show role warning when role is admin', () => {
    render(<VMTrainingConfigForm {...defaultProps} role="admin" />)
    expect(screen.queryByText('Switch to admin')).toBeNull()
  })

  it('calls onRoleChange when Switch to admin is clicked', () => {
    const onRoleChange = vi.fn()
    render(<VMTrainingConfigForm {...defaultProps} role="user" onRoleChange={onRoleChange} />)
    fireEvent.click(screen.getByText('Switch to admin'))
    expect(onRoleChange).toHaveBeenCalledWith('admin')
  })

  it('renders hints when provided', () => {
    const hints = [{ label: 'Epochs', message: 'using default 1' }]
    render(<VMTrainingConfigForm {...defaultProps} hints={hints} />)
    expect(screen.getByText(/using default 1/)).toBeDefined()
  })

  it('shows launched job notification', () => {
    render(<VMTrainingConfigForm {...defaultProps} launchedJob={5} />)
    expect(screen.getByText(/Launched training job #5/)).toBeDefined()
    expect(screen.getByText('Dismiss')).toBeDefined()
  })

  it('calls onDismissJob when Dismiss is clicked', () => {
    const onDismissJob = vi.fn()
    render(<VMTrainingConfigForm {...defaultProps} launchedJob={5} onDismissJob={onDismissJob} />)
    fireEvent.click(screen.getByText('Dismiss'))
    expect(onDismissJob).toHaveBeenCalledOnce()
  })
})
