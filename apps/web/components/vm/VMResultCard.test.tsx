/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { VMResultCard } from './VMResultCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  Badge: ({ children, ...props }: any) => <span data-testid="badge" {...props}>{children}</span>,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
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

describe('VMResultCard', () => {
  const mockResult = {
    success: true,
    exit_code: 0,
    steps_executed: 42,
    elapsed_ms: 1.5,
    status: 'halted',
    registers: [
      { name: 'EAX', value: 42, hex: '0x0000002A' },
      { name: 'EBX', value: 0, hex: '0x00000000' },
    ],
    eip_hex: '0x00001005',
  }

  it('renders the title', () => {
    render(<VMResultCard result={mockResult} />)
    expect(screen.getByText('Result')).toBeDefined()
  })

  it('renders nothing when result is null', () => {
    const { container } = render(<VMResultCard result={null} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders status badge', () => {
    render(<VMResultCard result={mockResult} />)
    expect(screen.getByText('halted')).toBeDefined()
  })

  it('renders exit code', () => {
    render(<VMResultCard result={mockResult} />)
    expect(screen.getByText('0x0')).toBeDefined()
  })

  it('renders steps executed', () => {
    render(<VMResultCard result={mockResult} />)
    expect(screen.getByText('42')).toBeDefined()
  })

  it('renders elapsed time', () => {
    render(<VMResultCard result={mockResult} />)
    expect(screen.getByText('1.5ms')).toBeDefined()
  })

  it('renders registers', () => {
    render(<VMResultCard result={mockResult} />)
    expect(screen.getByText('EAX')).toBeDefined()
    expect(screen.getByText('0x0000002A')).toBeDefined()
  })

  it('renders error message when present', () => {
    const errorResult = { ...mockResult, error: 'Division by zero', success: false }
    render(<VMResultCard result={errorResult} />)
    expect(screen.getByText('Division by zero')).toBeDefined()
  })

  it('renders EIP', () => {
    render(<VMResultCard result={mockResult} />)
    expect(screen.getByText('0x00001005')).toBeDefined()
  })
})
