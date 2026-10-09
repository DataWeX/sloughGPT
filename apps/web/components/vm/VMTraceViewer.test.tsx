/// <reference types="vitest" />
// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { VMTraceViewer } from './VMTraceViewer'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,

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

describe('VMTraceViewer', () => {
  const trace = [
    { step: 0, eip: '0x00001000', opcode: 'MOV', operands: 'EAX, 42' },
    { step: 1, eip: '0x00001003', opcode: 'HLT', operands: '' },
  ]

  it('renders nothing when trace is empty', () => {
    const { container } = render(<VMTraceViewer trace={[]} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders the title with trace length', () => {
    render(<VMTraceViewer trace={trace} />)
    expect(screen.getByText('Execution Trace (first 2 steps)')).toBeDefined()
  })

  it('renders table headers', () => {
    render(<VMTraceViewer trace={trace} />)
    expect(screen.getByText('#')).toBeDefined()
    expect(screen.getByText('EIP')).toBeDefined()
    expect(screen.getByText('Opcode')).toBeDefined()
    expect(screen.getByText('Operands')).toBeDefined()
  })

  it('renders trace rows', () => {
    render(<VMTraceViewer trace={trace} />)
    expect(screen.getByText('0x00001000')).toBeDefined()
    expect(screen.getByText('MOV')).toBeDefined()
    expect(screen.getByText('EAX, 42')).toBeDefined()
    expect(screen.getByText('0x00001003')).toBeDefined()
    expect(screen.getByText('HLT')).toBeDefined()
  })

  it('renders step numbers', () => {
    render(<VMTraceViewer trace={trace} />)
    expect(screen.getByText('0')).toBeDefined()
    expect(screen.getByText('1')).toBeDefined()
  })

  it('renders multiple trace entries', () => {
    const longTrace = [
      { step: 0, eip: '0x1000', opcode: 'MOV', operands: 'EAX, 1' },
      { step: 1, eip: '0x1003', opcode: 'ADD', operands: 'EAX, 2' },
      { step: 2, eip: '0x1006', opcode: 'HLT', operands: '' },
    ]
    render(<VMTraceViewer trace={longTrace} />)
    expect(screen.getByText('Execution Trace (first 3 steps)')).toBeDefined()
    expect(screen.getByText('ADD')).toBeDefined()
  })
})
