// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

const mocks = vi.hoisted(() => ({
  mockStatus: vi.fn(),
  mockStart: vi.fn(),
  mockStop: vi.fn(),
  mockTrigger: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/workflow-controller', () => ({
  workflowController: {
    status: mocks.mockStatus,
    start: mocks.mockStart,
    stop: mocks.mockStop,
    trigger: mocks.mockTrigger,
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: any) => selector({ addToast: mocks.mockAddToast }),
}))

vi.mock('@/components/workflow/WorkflowHealthCard', () => ({
  WorkflowHealthCard: ({ status }: any) => <div data-testid="health-card" />,
}))

vi.mock('@/components/workflow/WorkflowPipeline', () => ({
  WorkflowPipeline: () => <div data-testid="pipeline" />,
}))

vi.mock('@sloughgpt/strui', () => {
  const StatCard = ({ label, value }: any) => (
    <div>
      <span>{label}</span>
      <span>{value}</span>
    </div>
  )
  const KpiGrid = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...args: any[]) => args.filter(Boolean).join(' '),
    Card: ({ children }: any) => <div>{children}</div>,
    ActionCard: ({ title, actions, children, ...p }: any) => <div data-testid="action-card" {...p}>{title}{actions}{children}</div>,
    CardHeader: ({ children }: any) => <div>{children}</div>,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    CardContent: ({ children }: any) => <div>{children}</div>,
    Button: ({ children, onClick, disabled, ...props }: any) => (
      <button onClick={onClick} disabled={disabled} {...props}>{children}</button>
    ),
    StatCard,
    KpiGrid,
    IconRefresh: () => <span data-testid="icon-refresh" />,
  
Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
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

import { WorkflowSection } from './WorkflowSection'

const baseStatus = {
  running: false,
  stats: {
    feedback_recorded: 100,
    auto_train_steps: 5,
    workflow_runs: 2,
    dpo_train_steps: 1,
  },
  config: {
    aggregate_interval_minutes: 60,
    prune_interval_minutes: 30,
    export_interval_hours: 24,
    health_check_interval_seconds: 300,
  },
}

describe('WorkflowSection', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders loading KPIs initially', () => {
    mocks.mockStatus.mockReturnValue(new Promise(() => {}))
    render(<WorkflowSection />)
    expect(mocks.mockStatus).toHaveBeenCalled()
  })

  it('renders status and stat KPIs from the loaded status', async () => {
    mocks.mockStatus.mockResolvedValue(baseStatus)
    render(<WorkflowSection />)
    expect(await screen.findByText('Feedback Pipeline')).toBeDefined()
    expect((await screen.findAllByText('Stopped')).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Feedback Recorded').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('100').length).toBeGreaterThanOrEqual(1)
  })

  it('shows Running state and Stop button when the workflow is active', async () => {
    mocks.mockStatus.mockResolvedValue({ ...baseStatus, running: true })
    render(<WorkflowSection />)
    expect((await screen.findAllByText('Running')).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByRole('button', { name: 'Stop' }).length).toBeGreaterThanOrEqual(1)
  })

  it('starts the workflow when stopped', async () => {
    mocks.mockStatus.mockResolvedValue(baseStatus)
    mocks.mockStart.mockResolvedValue({ status: 'ok' })
    render(<WorkflowSection />)
    await screen.findByText('Feedback Pipeline')
    fireEvent.click(screen.getAllByRole('button', { name: 'Start' })[0])
    expect(mocks.mockStart).toHaveBeenCalled()
  })

  it('stops the workflow when running', async () => {
    mocks.mockStatus.mockResolvedValue({ ...baseStatus, running: true })
    mocks.mockStop.mockResolvedValue({ status: 'ok' })
    render(<WorkflowSection />)
    await screen.findAllByText('Running')
    fireEvent.click(screen.getAllByRole('button', { name: 'Stop' })[0])
    expect(mocks.mockStop).toHaveBeenCalled()
  })

  it('shows a toast when status loading fails', async () => {
    mocks.mockStatus.mockRejectedValue(new Error('down'))
    render(<WorkflowSection />)
    expect(await screen.findByText('Feedback Pipeline')).toBeDefined()
    expect(mocks.mockAddToast).toHaveBeenCalledWith('Could not load workflow status', 'error')
  })

  it('runs a manual trigger and shows the result message', async () => {
    mocks.mockStatus.mockResolvedValue(baseStatus)
    mocks.mockTrigger.mockResolvedValue({ status: 'done' })
    render(<WorkflowSection />)
    await screen.findByText('Feedback Pipeline')
    fireEvent.click(screen.getAllByRole('button', { name: 'Aggregate' })[0])
    expect(await screen.findByText('aggregate: done')).toBeDefined()
  })

  it('shows the failure message when a trigger fails', async () => {
    mocks.mockStatus.mockResolvedValue(baseStatus)
    mocks.mockTrigger.mockRejectedValue(new Error('boom'))
    render(<WorkflowSection />)
    await screen.findByText('Feedback Pipeline')
    fireEvent.click(screen.getAllByRole('button', { name: 'Prune' })[0])
    expect(await screen.findByText('prune failed')).toBeDefined()
  })

  it('renders health card and pipeline sections', async () => {
    mocks.mockStatus.mockResolvedValue(baseStatus)
    render(<WorkflowSection />)
    expect(await screen.findByTestId('health-card')).toBeDefined()
    expect(screen.getByTestId('pipeline')).toBeDefined()
  })

  it('renders configuration cards when config is present', async () => {
    mocks.mockStatus.mockResolvedValue(baseStatus)
    render(<WorkflowSection />)
    expect(await screen.findByText('Configuration')).toBeDefined()
    expect(screen.getByText('60 min')).toBeDefined()
    expect(screen.getByText('30 min')).toBeDefined()
    expect(screen.getByText('24 hr')).toBeDefined()
    expect(screen.getByText('300s')).toBeDefined()
  })

  it('renders stats detail card with DPO train steps', async () => {
    mocks.mockStatus.mockResolvedValue(baseStatus)
    render(<WorkflowSection />)
    expect(await screen.findByText('Stats')).toBeDefined()
    expect(screen.getByText('DPO Train Steps')).toBeDefined()
  })
})