// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

vi.mock('@/lib/registry-controller', () => ({
  registryController: { list: vi.fn(), stats: vi.fn(), best: vi.fn() },
}))
vi.mock('@/components/registry/RegistryHealthCard', () => ({
  RegistryHealthCard: () => <div data-testid="registry-health" />,
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (s: any) => s({ addToast: vi.fn() }),
}))
vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Card: ({ children, ...p }: any) => <div data-testid="card" {...p}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...p }: any) => <div data-testid="card-title" {...p}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...p }: any) => <button onClick={onClick} disabled={disabled} {...p}>{children}</button>,
  StatCard: ({ label, value, ...p }: any) => <div data-testid="stat-card" data-label={label}>{String(value)}</div>,
  KpiGrid: ({ children, ...p }: any) => <div data-testid="kpi-grid" {...p}>{children}</div>,
  ActionCard: ({ title, actions, children, ...p }: any) => <div data-testid="action-card" {...p}>{title}{actions}{children}</div>,
  SearchInput: ({ value, onChange, ...p }: any) => <input value={value} onChange={e => onChange(e.target.value)} {...p} />,
  Skeleton: (p: any) => <div data-testid="skeleton" {...p} />,
  IconRefresh: () => <span>↻</span>,

    Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
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
}))

import RegistryContent from './RegistryContent'
import { registryController } from '@/lib/registry-controller'

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(registryController.list).mockResolvedValue([
    { model_id: 'gpt2', status: 'loaded', metrics: {} },
    { model_id: 'qwen', status: 'failed', metrics: { error: 'Timeout' } },
  ])
  vi.mocked(registryController.stats).mockResolvedValue({ total_models: 2, loaded_models: 1, failed_models: 1, circuit_breaker_open: false })
  vi.mocked(registryController.best).mockResolvedValue({ model_id: 'gpt2' })
})

afterEach(() => cleanup())

describe('RegistryContent', () => {
  it('calls registry APIs on mount', async () => {
    render(<RegistryContent />)
    await waitFor(() => {
      expect(registryController.list).toHaveBeenCalled()
      expect(registryController.stats).toHaveBeenCalled()
      expect(registryController.best).toHaveBeenCalled()
    })
  })

  it('displays KPI stats', async () => {
    render(<RegistryContent />)
    await waitFor(() => {
      expect(screen.getAllByTestId('stat-card').length).toBeGreaterThanOrEqual(4)
    })
  })
})
