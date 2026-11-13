/// <reference types="vitest" />
// @vitest-environment jsdom
import { render, screen, cleanup } from '@testing-library/react'
import { afterEach, describe, it, expect, vi } from 'vitest'
import { VMEditor } from './VMEditor'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div data-testid="card-header">{children}</div>,
  CardTitle: ({ children }: any) => <div data-testid="card-title">{children}</div>,
  CardContent: ({ children }: any) => <div data-testid="card-content">{children}</div>,
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button onClick={onClick} disabled={disabled} {...props}>{children}</button>
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

afterEach(() => cleanup())

describe('VMEditor', () => {
  const defaultProps = {
    source: 'MOV EAX, 42\nHLT',
    onSourceChange: vi.fn(),
    onRun: vi.fn(),
    onStep: vi.fn(),
    onClear: vi.fn(),
    running: false,
    hasResult: false,
  }

  it('renders the card title', () => {
    render(<VMEditor {...defaultProps} />)
    expect(screen.getByText('Assembly Source')).toBeDefined()
  })

  it('renders the source code textarea', () => {
    render(<VMEditor {...defaultProps} />)
    const textarea = screen.getByLabelText('Assembly source code')
    expect(textarea).toBeDefined()
    expect((textarea as HTMLTextAreaElement).value).toBe('MOV EAX, 42\nHLT')
  })

  it('renders Run and Step buttons', () => {
    render(<VMEditor {...defaultProps} />)
    expect(screen.getByText('Run')).toBeDefined()
    expect(screen.getByText('Step')).toBeDefined()
  })

  it('disables buttons when running', () => {
    render(<VMEditor {...defaultProps} running={true} />)
    expect((screen.getByText('Running...') as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByText('Step') as HTMLButtonElement).disabled).toBe(true)
  })

  it('shows line numbers for source code', () => {
    render(<VMEditor {...defaultProps} />)
    expect(screen.getByText('1')).toBeDefined()
    expect(screen.getByText('2')).toBeDefined()
  })

  it('disables Clear when no result', () => {
    render(<VMEditor {...defaultProps} hasResult={false} />)
    expect((screen.getByText('Clear') as HTMLButtonElement).disabled).toBe(true)
  })

  it('calls onRun when Run is clicked', () => {
    const onRun = vi.fn()
    render(<VMEditor {...defaultProps} onRun={onRun} />)
    screen.getByText('Run').click()
    expect(onRun).toHaveBeenCalledOnce()
  })
})
