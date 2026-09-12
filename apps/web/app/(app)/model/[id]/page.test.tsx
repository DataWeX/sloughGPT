import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...args: any[]) => args.join(' '),
    Button: ({ children, onClick, disabled, variant, size, className, 'aria-label': ariaLabel }: any) => (
      <button onClick={onClick} disabled={disabled} data-variant={variant} aria-label={ariaLabel} className={className}>{children}</button>
    ),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    Skeleton: () => <div data-testid="skeleton" />,
    Badge: ({ label, variant, size, className, children }: any) => <span data-variant={variant} className={className}>{label || children}</span>,
    StatCard: ({ label, value }: any) => <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span></div>,
    KpiGrid: ({ children, columns }: any) => <div data-columns={columns}>{children}</div>,
    KeyValueList: ({ items }: any) => <div>{items?.map((item: any, i: number) => <div key={i} data-testid={`kv-${item.label}`}><span>{item.label}</span><span>{item.value}</span></div>)}</div>,
    SettingsRow: ({ title, control, children }: any) => <div><span>{title}</span>{control || children}</div>,
    Slider: ({ value, onValueChange, min, max, step }: any) => (
      <input type="range" value={value?.[0]} min={min} max={max} step={step}
        onChange={e => onValueChange?.([Number(e.target.value)])} />
    ),
    Breadcrumbs: ({ items, className }: any) => <nav aria-label="Breadcrumb" className={className}>{items?.map((item: any, i: number) => <span key={i}>{item.label}</span>)}</nav>,
    IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
    IconTrash: () => <span data-testid="icon-trash">trash</span>,
  
    Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
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
}
})

const mocks = vi.hoisted(() => ({
  push: vi.fn(),
  addToast: vi.fn(),
  list: vi.fn(),
  getHealth: vi.fn(),
  load: vi.fn(),
  unloadModel: vi.fn(),
  listFineTuned: vi.fn(),
  loadFineTuned: vi.fn(),
  run: vi.fn(),
  get: vi.fn(),
  update: vi.fn(),
  apiGet: vi.fn(),
}))

const stableRouter = { push: vi.fn() }
vi.mock('next/navigation', () => ({ useParams: () => ({ id: 'gpt2' }), useRouter: () => stableRouter, useSearchParams: () => new URLSearchParams(), usePathname: () => '/model/gpt2' }))
vi.mock('@/lib/toast-store', () => ({ useToastStore: (sel: any) => sel({ addToast: mocks.addToast }) }))
vi.mock('@/lib/model-controller', () => ({
  modelController: {
    list: mocks.list,
    getHealth: mocks.getHealth,
    load: mocks.load,
    unloadModel: mocks.unloadModel,
  },
}))
vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    listFineTuned: mocks.listFineTuned,
    loadFineTuned: mocks.loadFineTuned,
  },
}))
vi.mock('@/lib/benchmark-controller', () => ({
  benchmarkController: { run: mocks.run },
}))
vi.mock('@/lib/generation-config-controller', () => ({
  generationConfigController: { get: mocks.get, update: mocks.update },
}))
vi.mock('@/lib/error-utils', () => ({
  extractErrorMessage: (e: any, fallback: string) => e instanceof Error ? e.message : fallback,
}))
vi.mock('@/lib/http-client', () => ({
  apiGet: mocks.apiGet,
}))
vi.mock('@/lib/dev-log', () => ({
  logger: { debug: vi.fn() },
}))
vi.mock('@/components/model/QuantizeCard', () => ({
  QuantizeCard: ({ isLoaded, modelId }: any) => <div data-testid="quantize-card" data-loaded={isLoaded}>Quantize</div>,
}))

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ title, children, loading, loadingContent, loadingCards }: any) => (
    <div className="sl-page mx-auto max-w-4xl">
      <h1>{loading ? '...' : title}</h1>
      {loading ? (loadingContent ?? (loadingCards ? <div data-testid="skeleton" /> : null)) : children}
    </div>
  ),
}))

import Page from './page'

const SAMPLE_MODEL: any = {
  model_id: 'gpt2', id: 'gpt2', name: 'GPT-2', source: 'huggingface', description: 'Small model',
  size_gb: 0.5, params: '124M', type: 'text-generation', tags: ['gpt', 'small'],
  status: 'ready',
}

beforeEach(() => {
  vi.clearAllMocks()
  mocks.list.mockResolvedValue([SAMPLE_MODEL])
  mocks.getHealth.mockResolvedValue({ model_loaded: false, device: null })
  mocks.listFineTuned.mockResolvedValue([])
  mocks.apiGet.mockImplementation((url: string) => {
    if (url.includes('/registry/models/')) return Promise.resolve(SAMPLE_MODEL)
    return Promise.resolve({ logs: [] })
  })
  mocks.get.mockResolvedValue({ temperature: 0.7, max_new_tokens: 256, top_p: 1.0, top_k: 50 })
  mocks.update.mockResolvedValue({})
})

afterEach(() => cleanup())

describe('ModelDetailPage', () => {
  it('renders loading state', async () => {
    mocks.list.mockImplementation(() => new Promise(() => {}))
    render(<Page />)
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(1)
  })

  it('shows model name and breadcrumbs', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getAllByText(/GPT.?2/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows status card with ready badge', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('ready')).toBeTruthy()
    })
  })

  it('shows unloaded status for non-ready model', async () => {
    mocks.apiGet.mockImplementation((url: string) => {
      if (url.includes('/registry/models/')) return Promise.resolve({ ...SAMPLE_MODEL, status: 'unloaded' })
      return Promise.resolve({ logs: [] })
    })
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('unloaded')).toBeTruthy()
    })
  })

  it('shows Quick Actions card', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Quick Actions')).toBeTruthy()
    })
  })

  it('shows Latency Distribution card', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Latency Distribution')).toBeTruthy()
    })
  })

  it('shows metrics grid with zero values', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Total Requests')).toBeTruthy()
    })
    expect(screen.getByText('Total Tokens')).toBeTruthy()
    expect(screen.getByText('Avg Latency')).toBeTruthy()
    expect(screen.getByText('Tokens/sec')).toBeTruthy()
  })

  it('shows Request Queue card', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Request Queue')).toBeTruthy()
    })
  })

  it('shows Circuit Breaker card', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Circuit Breaker')).toBeTruthy()
    })
  })

  it('navigates to training queue', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Quick Actions')).toBeTruthy()
    })
    fireEvent.click(screen.getByText('Training Queue'))
    expect(stableRouter.push).toHaveBeenCalledWith('/training/queue')
  })

  it('navigates to HuggingFace', async () => {
    const openSpy = vi.spyOn(window, 'open')
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('HuggingFace')).toBeTruthy()
    })
    fireEvent.click(screen.getByText('HuggingFace'))
    expect(openSpy).toHaveBeenCalledWith('https://huggingface.co/gpt2', '_blank')
    openSpy.mockRestore()
  })

  it('calls apiGet on mount', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(mocks.apiGet).toHaveBeenCalledWith('/registry/models/gpt2')
    })
  })

  it('shows error toast on load failure', async () => {
    mocks.apiGet.mockRejectedValue(new Error('network'))
    render(<Page />)
    await waitFor(() => {
      expect(mocks.addToast).toHaveBeenCalledWith('Failed to load model details', 'error')
    })
  })

  it('shows placeholder when model data is empty', async () => {
    mocks.apiGet.mockImplementation((url: string) => {
      if (url.includes('/registry/models/')) return Promise.resolve(null)
      return Promise.resolve({ logs: [] })
    })
    render(<Page />)
    await waitFor(() => {
      expect(mocks.apiGet).toHaveBeenCalled()
    })
  })
})
