import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { KbAddEntryForm } from './KbAddEntryForm'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button data-testid="button" onClick={onClick} disabled={disabled} {...props}>{children}</button>
  ),
  Input: ({ onChange, ...props }: any) => <input data-testid="input" onChange={onChange} {...props} />,
  Label: ({ children, ...props }: any) => <label data-testid="label" {...props}>{children}</label>,
  Textarea: ({ onChange, ...props }: any) => <textarea data-testid="textarea" onChange={onChange} {...props} />,
  Slider: () => <div data-testid="slider" />,

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

afterEach(() => cleanup())

describe('KbAddEntryForm', () => {
  const defaultProps = {
    content: '',
    topic: 'general',
    importance: 0.7,
    loading: false,
    suggestResult: null,
    onContentChange: vi.fn(),
    onTopicChange: vi.fn(),
    onImportanceChange: vi.fn(),
    onAdd: vi.fn(),
    onSuggest: vi.fn(),
  }

  it('renders the form title', () => {
    render(<KbAddEntryForm {...defaultProps} />)
    expect(screen.getByText('Add Knowledge Entry')).toBeDefined()
  })

  it('renders content and topic inputs', () => {
    render(<KbAddEntryForm {...defaultProps} />)
    expect(screen.getAllByTestId('textarea').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByTestId('input').length).toBeGreaterThanOrEqual(1)
  })

  it('renders Add Entry and Suggest Topic buttons', () => {
    render(<KbAddEntryForm {...defaultProps} />)
    expect(screen.getByText('Add Entry')).toBeDefined()
    expect(screen.getByText('Suggest Topic')).toBeDefined()
  })

  it('calls onAdd when Add Entry is clicked', () => {
    const onAdd = vi.fn()
    render(<KbAddEntryForm {...defaultProps} content="test" onAdd={onAdd} />)
    fireEvent.click(screen.getByText('Add Entry'))
    expect(onAdd).toHaveBeenCalledOnce()
  })

  it('disables Add Entry when content is empty', () => {
    render(<KbAddEntryForm {...defaultProps} />)
    const addBtn = screen.getByText('Add Entry')
    expect(addBtn.closest('button')?.disabled).toBe(true)
  })

  it('shows suggested topic when provided', () => {
    render(<KbAddEntryForm {...defaultProps} suggestResult="science" />)
    expect(screen.getByText('Suggested: science')).toBeDefined()
  })

  it('shows Adding... when loading', () => {
    render(<KbAddEntryForm {...defaultProps} content="test" loading={true} />)
    expect(screen.getByText('Adding...')).toBeDefined()
  })
})
