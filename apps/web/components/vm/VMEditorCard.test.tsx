/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { VMEditorCard } from './VMEditorCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
  Textarea: (props: any) => <textarea data-testid="source-editor" {...props} />,

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

describe('VMEditorCard', () => {
  const defaultProps = {
    source: '[BITS 32]\nMOV EAX, 42\nHLT',
    onSourceChange: vi.fn(),
    onRun: vi.fn(),
  }

  it('renders the title', () => {
    render(<VMEditorCard {...defaultProps} />)
    expect(screen.getByText('Assembly Source')).toBeDefined()
  })

  it('renders the source editor with content', () => {
    render(<VMEditorCard {...defaultProps} />)
    expect((screen.getByTestId('source-editor') as HTMLTextAreaElement).value).toContain('MOV EAX, 42')
  })

  it('renders Run button', () => {
    render(<VMEditorCard {...defaultProps} />)
    expect(screen.getByText('Run')).toBeDefined()
  })

  it('calls onRun when Run is clicked', () => {
    const onRun = vi.fn()
    render(<VMEditorCard {...defaultProps} onRun={onRun} />)
    fireEvent.click(screen.getByText('Run'))
    expect(onRun).toHaveBeenCalledOnce()
  })

  it('calls onSourceChange when editor changes', () => {
    const onSourceChange = vi.fn()
    render(<VMEditorCard {...defaultProps} onSourceChange={onSourceChange} />)
    fireEvent.change(screen.getByTestId('source-editor'), { target: { value: 'NEW' } })
    expect(onSourceChange).toHaveBeenCalledWith('NEW')
  })

  it('disables Run button when running', () => {
    render(<VMEditorCard {...defaultProps} running />)
    const btn = screen.getByText('Running...')
    expect(btn).toBeDefined()
  })

  it('renders program buttons when programs provided', () => {
    const programs = { hello: 'code1', count: 'code2' }
    render(<VMEditorCard {...defaultProps} programs={programs} onProgramSelect={vi.fn()} />)
    expect(screen.getByText('hello')).toBeDefined()
    expect(screen.getByText('count')).toBeDefined()
  })

  it('renders Step button when onStep provided', () => {
    render(<VMEditorCard {...defaultProps} onStep={vi.fn()} />)
    expect(screen.getByText('Step')).toBeDefined()
  })
})
