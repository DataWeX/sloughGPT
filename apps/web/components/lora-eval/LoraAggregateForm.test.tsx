import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { LoraAggregateForm } from './LoraAggregateForm'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button data-testid="button" onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: ({ onChange, ...props }: any) => <input data-testid="input" onChange={onChange} {...props} />,

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

afterEach(() => cleanup())

describe('LoraAggregateForm', () => {
  const defaultProps = {
    topK: 10,
    minFeedback: 5,
    aggregating: false,
    onTopKChange: vi.fn(),
    onMinFeedbackChange: vi.fn(),
    onAggregate: vi.fn(),
  }

  it('renders the form title', () => {
    render(<LoraAggregateForm {...defaultProps} />)
    expect(screen.getByText('Aggregate Adapters')).toBeDefined()
  })

  it('renders Top K and Min Feedback inputs', () => {
    render(<LoraAggregateForm {...defaultProps} />)
    const inputs = screen.getAllByTestId('input')
    expect(inputs.length).toBeGreaterThanOrEqual(2)
  })

  it('renders the Aggregate button', () => {
    render(<LoraAggregateForm {...defaultProps} />)
    expect(screen.getByText('Aggregate')).toBeDefined()
  })

  it('calls onAggregate when Aggregate is clicked', () => {
    const onAggregate = vi.fn()
    render(<LoraAggregateForm {...defaultProps} onAggregate={onAggregate} />)
    fireEvent.click(screen.getByText('Aggregate'))
    expect(onAggregate).toHaveBeenCalledOnce()
  })

  it('shows Aggregating... when aggregating', () => {
    render(<LoraAggregateForm {...defaultProps} aggregating={true} />)
    expect(screen.getByText('Aggregating...')).toBeDefined()
  })

  it('disables Aggregate button when aggregating', () => {
    render(<LoraAggregateForm {...defaultProps} aggregating={true} />)
    const btn = screen.getByText('Aggregating...').closest('button')
    expect(btn?.disabled).toBe(true)
  })
})
