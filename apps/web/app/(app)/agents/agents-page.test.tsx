import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('class-variance-authority', () => ({ cva: () => () => '' }))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), back: vi.fn(), prefetch: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => '/agents',
}))

vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => { const C = () => <span data-testid={`icon-${name}`}>{name}</span>; C.displayName = `Icon${name}`; return C }
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    Button: ({ children, onClick, variant, size, className, disabled, 'aria-label': ariaLabel }: any) => (
      <button onClick={onClick} className={className} disabled={disabled} aria-label={ariaLabel} data-variant={variant}>{children}</button>
    ),
    Input: ({ value, onChange, className, placeholder }: any) => <input value={value} onChange={onChange} className={className} placeholder={placeholder} />,
    EmptyCard: ({ message }: any) => <div>{message}</div>,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    StatCard: ({ label, value }: any) => <div><span>{label}</span><span>{String(value)}</span></div>,
    IconRefresh: iconMock('refresh'), IconPlus: iconMock('plus'), IconTrash: iconMock('trash'), IconClock: iconMock('clock'),
    AlertDialog: ({ children, open, onOpenChange }: any) => open ? <div data-testid="alert-dialog">{children}</div> : null,
    AlertDialogContent: ({ children }: any) => <div>{children}</div>,
    AlertDialogHeader: ({ children }: any) => <div>{children}</div>,
    AlertDialogTitle: ({ children }: any) => <div>{children}</div>,
    AlertDialogDescription: ({ children }: any) => <div>{children}</div>,
    AlertDialogFooter: ({ children }: any) => <div>{children}</div>,
    AlertDialogCancel: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    AlertDialogAction: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  
    Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
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
}
})

vi.mock('@/components/icons/NavIcons', () => {
  const iconMock = (name: string) => { const C = () => <span data-testid={`icon-${name}`}>{name}</span>; C.displayName = `Icon${name}`; return C }
  return { IconPlus: iconMock('plus'), IconTrash: iconMock('trash'), IconClock: iconMock('clock') }
})

const { mockList, mockListRuns, mockAddToast } = vi.hoisted(() => ({
  mockList: vi.fn(), mockListRuns: vi.fn(), mockAddToast: vi.fn(),
}))

vi.mock('@/lib/agents-controller', () => ({
  agentsController: { list: mockList, listRuns: mockListRuns, create: vi.fn(), update: vi.fn(), delete: vi.fn(), execute: vi.fn(), orchestrate: vi.fn() },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

import AgentsPage from './page'

const RUN_FIXTURE = {
  id: 'run_1',
  goal: 'Research transformers',
  context: '',
  status: 'completed',
  started_at: '2026-08-01T12:00:00Z',
  finished_at: '2026-08-01T12:01:00Z',
  tasks: [
    { id: 't1', description: 'Gather papers', agent: 'researcher', status: 'completed', result_preview: 'notes', depends_on: [] },
    { id: 't2', description: 'Write summary', agent: 'writer', status: 'completed', result_preview: 'summary', depends_on: ['t1'] },
  ],
  completed_count: 2,
  failed_count: 0,
  response: 'Here is the summary.',
  error: '',
  logs: ['[2026-08-01T12:00:00Z] Started: Research transformers', '[2026-08-01T12:01:00Z] Completed'],
}

describe('AgentsPage Run History', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue([])
    mockListRuns.mockResolvedValue({ runs: [], count: 0 })
  })

  afterEach(() => cleanup())

  it('shows empty run history state', async () => {
    render(<AgentsPage />)
    await waitFor(() => { expect(screen.getByText('Run History')).toBeTruthy() })
    await waitFor(() => { expect(screen.getByText(/No runs yet/)).toBeTruthy() })
  })

  it('lists runs with status and goal', async () => {
    mockListRuns.mockResolvedValue({ runs: [RUN_FIXTURE], count: 1 })
    render(<AgentsPage />)
    await waitFor(() => { expect(screen.getByText('Research transformers')).toBeTruthy() })
    expect(mockListRuns).toHaveBeenCalledWith(20)
    expect(screen.getByText('2/2 tasks')).toBeTruthy()
  })

  it('expands a run to show tasks, result, and logs', async () => {
    mockListRuns.mockResolvedValue({ runs: [RUN_FIXTURE], count: 1 })
    render(<AgentsPage />)
    await waitFor(() => { expect(screen.getByText('Research transformers')).toBeTruthy() })
    screen.getByText('Research transformers').click()
    await waitFor(() => { expect(screen.getByText('Gather papers')).toBeTruthy() })
    expect(screen.getByText('Write summary')).toBeTruthy()
    expect(screen.getByText('Here is the summary.')).toBeTruthy()
    expect(screen.getByText(/Started: Research transformers/)).toBeTruthy()
  })

  it('shows error message for failed runs', async () => {
    mockListRuns.mockResolvedValue({
      runs: [{ ...RUN_FIXTURE, id: 'run_2', status: 'failed', error: 'LLM failed', response: '', tasks: [] }],
      count: 1,
    })
    render(<AgentsPage />)
    await waitFor(() => { expect(screen.getByText('Research transformers')).toBeTruthy() })
    screen.getByText('Research transformers').click()
    await waitFor(() => { expect(screen.getByText('LLM failed')).toBeTruthy() })
  })

  it('handles run history fetch failure gracefully', async () => {
    mockListRuns.mockRejectedValue(new Error('500'))
    render(<AgentsPage />)
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Could not load run history', 'error') })
  })

  it('calls listRuns with default limit on mount', async () => {
    render(<AgentsPage />)
    await waitFor(() => { expect(mockListRuns).toHaveBeenCalledWith(20) })
  })
})
