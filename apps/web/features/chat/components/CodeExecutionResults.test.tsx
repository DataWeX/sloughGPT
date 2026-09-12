import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  IconX: (props: any) => <span data-testid="icon-x" {...props} />,
  IconCheck: (props: any) => <span data-testid="icon-check" {...props} />,
  IconDownload: (props: any) => <span data-testid="icon-download" {...props} />,

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

import { CodeExecutionResults } from './CodeExecutionResults'

afterEach(cleanup)
beforeEach(() => {
  Object.defineProperty(navigator, 'clipboard', {
    value: {
      writeText: vi.fn().mockResolvedValue(undefined),
    },
    writable: true,
  })
})

const mockExecutions = [
  {
    id: '1',
    code: 'print("Hello")',
    language: 'python',
    output: 'Hello',
    exitCode: 0,
    duration: 150,
    timestamp: new Date(),
  },
  {
    id: '2',
    code: 'console.log("World")',
    language: 'javascript',
    output: 'World',
    exitCode: 0,
    duration: 50,
    timestamp: new Date(),
  },
  {
    id: '3',
    code: 'invalid code',
    language: 'python',
    output: '',
    error: 'SyntaxError: invalid syntax',
    exitCode: 1,
    duration: 10,
    timestamp: new Date(),
  },
]

describe('CodeExecutionResults', () => {
  it('renders empty state', () => {
    render(<CodeExecutionResults executions={[]} />)
    expect(screen.getByText('No code executions')).toBeInTheDocument()
  })

  it('renders execution count', () => {
    render(<CodeExecutionResults executions={mockExecutions} />)
    const pythons = screen.getAllByText('python')
    expect(pythons.length).toBe(2)
    expect(screen.getByText('javascript')).toBeInTheDocument()
  })

  it('shows success icon for exit code 0', () => {
    render(<CodeExecutionResults executions={[mockExecutions[0]]} />)
    expect(screen.getByTestId('icon-check')).toBeInTheDocument()
  })

  it('shows error icon for non-zero exit code', () => {
    render(<CodeExecutionResults executions={[mockExecutions[2]]} />)
    expect(screen.getByTestId('icon-x')).toBeInTheDocument()
  })

  it('expands execution on click', () => {
    render(<CodeExecutionResults executions={[mockExecutions[0]]} />)
    fireEvent.click(screen.getByText('python'))
    expect(screen.getByText('print("Hello")')).toBeInTheDocument()
    expect(screen.getByText('Output')).toBeInTheDocument()
  })

  it('shows error panel for failed execution', () => {
    render(<CodeExecutionResults executions={[mockExecutions[2]]} />)
    fireEvent.click(screen.getByText('python'))
    expect(screen.getByText('Error')).toBeInTheDocument()
    expect(screen.getByText('SyntaxError: invalid syntax')).toBeInTheDocument()
  })

  it('calls onRerun when rerun clicked', () => {
    const onRerun = vi.fn()
    render(<CodeExecutionResults executions={[mockExecutions[0]]} onRerun={onRerun} />)
    fireEvent.click(screen.getByText('python'))
    fireEvent.click(screen.getByText('Rerun'))
    expect(onRerun).toHaveBeenCalledWith('1')
  })

  it('copies code to clipboard', async () => {
    render(<CodeExecutionResults executions={[mockExecutions[0]]} />)
    fireEvent.click(screen.getByText('python'))
    const copyBtn = screen.getAllByRole('button')[0]
    fireEvent.click(copyBtn)
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('print("Hello")')
  })

  it('collapses expanded execution', () => {
    render(<CodeExecutionResults executions={[mockExecutions[0]]} />)
    fireEvent.click(screen.getByText('python'))
    expect(screen.getByText('print("Hello")')).toBeInTheDocument()
    fireEvent.click(screen.getByText('python'))
    expect(screen.queryByText('print("Hello")')).not.toBeInTheDocument()
  })

  it('shows duration', () => {
    render(<CodeExecutionResults executions={[mockExecutions[0]]} />)
    expect(screen.getByText('150ms')).toBeInTheDocument()
  })
})