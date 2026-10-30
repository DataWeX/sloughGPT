import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor, act } from '@testing-library/react'

const mockHealthState = vi.fn()
const mockGetCurrent = vi.fn()
const mockGetTraitWeights = vi.fn()

vi.mock('@/hooks/useLiveStatus', () => ({
  useLiveStatus: () => {
    const h = mockHealthState()
    if (h === 'offline')
      return {
        connectionStatus: 'offline',
        health: null,
        healthLegacy: 'offline',
        lastUpdate: null,
        failureCount: 1,
        connected: false,
        live: false,
      }
    if (h === 'error')
      return {
        connectionStatus: 'error',
        health: null,
        healthLegacy: null,
        lastUpdate: null,
        failureCount: 6,
        connected: false,
        live: false,
      }
    if (h === 'stale')
      return {
        connectionStatus: 'connecting',
        health: {
          model_loaded: true,
          model_type: 'gpt2',
          inference_count: 42,
          health_score: 85,
          health_status: 'healthy',
          health_summary: 'Healthy',
          tokens_per_sec: 15,
          is_inferencing: false,
          cpu_percent: 30,
          memory_percent: 40,
          uptime_seconds: 100,
          request_count: 50,
          error_count: 0,
          soul: 'friendly',
        },
        healthLegacy: null,
        lastUpdate: Date.now(),
        failureCount: 1,
        connected: false,
        live: false,
      }
    if (h === null)
      return {
        connectionStatus: 'connecting',
        health: null,
        healthLegacy: null,
        lastUpdate: null,
        failureCount: 0,
        connected: false,
        live: false,
      }
    return {
      connectionStatus: 'connected',
      health: h,
      healthLegacy: h,
      lastUpdate: Date.now(),
      failureCount: 0,
      connected: true,
      live: true,
    }
  },
  liveStatusStore: {
    getState: vi.fn(() => ({
      connectionStatus: 'connected',
      health: null,
      healthLegacy: null,
      lastUpdate: null,
      failureCount: 0,
    })),
    subscribe: vi.fn(() => vi.fn()),
  },
}))

vi.mock('@/lib/souls-controller', () => ({
  soulsController: {
    getCurrent: (...a: any[]) => mockGetCurrent(...a),
    getTraitWeights: (...a: any[]) => mockGetTraitWeights(...a),
  },
}))

vi.mock('@/components/souls/PersonalitySummary', () => ({
  deriveArchetype: vi.fn(() => ({ label: 'The Optimist' })),
}))

const mockGetUnseenCount = vi.fn()

vi.mock('@/components/WhatsNewDialog', () => ({
  getUnseenCount: () => Promise.resolve(mockGetUnseenCount()),
}))

const mockHealthSummary = {
  score: 85,
  summary: 'Healthy',
  tokens_per_sec: 15,
  model_loaded: true,
  model_type: 'gpt2',
  soul: 'friendly',
  uptime_seconds: 100,
  request_count: 50,
  error_count: 0,
  cpu_percent: 30,
  memory_percent: 40,
}

const mockClearFailures = vi.fn()

const { useApiMonitor: _useApiMonitor, setHealthSummaryData } = vi.hoisted(() => {
  let hc: typeof mockHealthSummary | null = null
  return {
    useApiMonitor: (selector: (s: any) => any) =>
      selector({
        healthSummary: hc,
        recentFailures: [],
        failureCount: 0,
        lastOffline: null,
        clearFailures: mockClearFailures,
      }),
    setHealthSummaryData: (v: typeof mockHealthSummary | null) => {
      hc = v
    },
  }
})

vi.mock('@/lib/api-monitor-store', () => ({
  useApiMonitor: _useApiMonitor,
}))

vi.mock('@/lib/config', () => ({
  PUBLIC_API_URL: 'http://localhost:8000',
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
  Tooltip: ({ children }: any) => <div>{children}</div>,
  TooltipTrigger: ({ children }: any) => <div>{children}</div>,
  TooltipContent: ({ children }: any) => <div>{children}</div>,
  IconMenu: (props: any) => <span />,
  IconGrid: (props: any) => <span />,
  IconRefresh: (props: any) => <span />,
  IconX: (props: any) => <span />,

  Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
  Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
  Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
  ActionCard: ({ title, children }: any) => (
    <div data-testid="action-card">
      <h3>{title}</h3>
      {children}
    </div>
  ),
  Tabs: ({ children }: any) => <div>{children}</div>,
  TabsList: ({ children }: any) => <div>{children}</div>,
  TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  TabsContent: ({ children }: any) => <div>{children}</div>,
  Textarea: ({ value, onChange, ...props }: any) => (
    <textarea value={value} onChange={onChange} {...props} />
  ),
  Separator: () => <hr />,
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

const { probeBackendAndRevalidateMock } = vi.hoisted(() => ({
  probeBackendAndRevalidateMock: vi.fn(),
}))
vi.mock('@/lib/backend-retry', () => ({
  probeBackendAndRevalidate: probeBackendAndRevalidateMock,
}))

import { StatusBar, ERROR_BADGE_TTL_MS } from './StatusBar'
import { useErrorStore } from '@/lib/error-store'

describe('StatusBar', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setHealthSummaryData(null)
    useErrorStore.getState().clearErrors()
    mockGetUnseenCount.mockReturnValue(0)
    mockHealthState.mockReturnValue({
      model_loaded: true,
      model_type: 'gpt2',
      inference_count: 42,
      health_score: 85,
      health_status: 'healthy',
      health_summary: 'Healthy',
      tokens_per_sec: 15,
      is_inferencing: false,
      cpu_percent: 30,
      memory_percent: 40,
      uptime_seconds: 100,
      request_count: 50,
      error_count: 0,
      soul: 'friendly',
    })
    mockGetCurrent.mockResolvedValue({ name: 'friendly', description: 'Friendly soul' })
    mockGetTraitWeights.mockResolvedValue({
      personality: { warmth: 0.7 },
      cognition: {},
      emotion: {},
    })
  })

  afterEach(cleanup)

  it('renders status text', () => {
    render(<StatusBar />)
    expect(screen.getByText('Healthy')).toBeDefined()
  })

  it('renders archetype badge when soul and weights available', async () => {
    render(<StatusBar />)
    const badge = await screen.findAllByText('The Optimist')
    expect(badge.length).toBeGreaterThanOrEqual(1)
  })

  it('renders tokens per second from live health', async () => {
    mockHealthState.mockReturnValue({
      model_loaded: true,
      model_type: 'gpt2',
      inference_count: 42,
      health_score: 85,
      health_status: 'healthy',
      health_summary: 'Healthy',
      tokens_per_sec: 15,
      is_inferencing: false,
      cpu_percent: 30,
      memory_percent: 40,
      uptime_seconds: 100,
      request_count: 50,
      error_count: 0,
      soul: 'friendly',
    })
    render(<StatusBar />)
    const tps = await screen.findAllByText('15 tok/s')
    expect(tps.length).toBeGreaterThanOrEqual(1)
  })

  it('renders inference count when no tps', async () => {
    mockHealthState.mockReturnValue({
      model_loaded: true,
      model_type: 'gpt2',
      inference_count: 42,
      health_score: 85,
      health_status: 'healthy',
      health_summary: 'Healthy',
      tokens_per_sec: 0,
      is_inferencing: false,
      cpu_percent: 30,
      memory_percent: 40,
      uptime_seconds: 100,
      request_count: 50,
      error_count: 0,
      soul: 'friendly',
    })
    render(<StatusBar />)
    const count = await screen.findAllByText('42 reqs')
    expect(count.length).toBeGreaterThanOrEqual(1)
  })

  it('shows offline status', () => {
    mockHealthState.mockReturnValue('offline')
    render(<StatusBar />)
    expect(screen.getByText(/Offline/)).toBeDefined()
  })

  it('shows the recoverable not-responding status when reload protection engaged', () => {
    mockHealthState.mockReturnValue('error')
    render(<StatusBar />)
    // 'error' must not fall through to the healthy branch ("No model").
    expect(screen.getByText(/Not responding — retrying/)).toBeDefined()
    expect(screen.queryByText('No model')).toBeNull()
  })

  it('does not re-fetch soul on health-only ticks', async () => {
    const { rerender } = render(<StatusBar />)
    await screen.findAllByText('The Optimist')
    expect(mockGetCurrent).toHaveBeenCalledTimes(1)
    rerender(<StatusBar />)
    rerender(<StatusBar />)
    await waitFor(() => {
      expect(mockGetCurrent).toHaveBeenCalledTimes(1)
    })
  })

  it('re-fetches soul when the active soul changes', async () => {
    const { rerender } = render(<StatusBar />)
    await screen.findAllByText('The Optimist')
    mockHealthState.mockReturnValue({
      model_loaded: true,
      model_type: 'gpt2',
      inference_count: 42,
      health_score: 85,
      health_status: 'healthy',
      health_summary: 'Healthy',
      tokens_per_sec: 15,
      is_inferencing: false,
      cpu_percent: 30,
      memory_percent: 40,
      uptime_seconds: 100,
      request_count: 50,
      error_count: 0,
      soul: 'curious',
    })
    mockGetCurrent.mockResolvedValue({ name: 'curious', description: 'Curious soul' })
    rerender(<StatusBar />)
    await waitFor(() => {
      expect(mockGetCurrent).toHaveBeenCalledTimes(2)
    })
  })

  it('links to monitoring page', () => {
    render(<StatusBar />)
    const link = screen.getByRole('link')
    expect(link.getAttribute('href')).toBe('/settings')
  })

  it('dispatches toggle-whatsnew event when whats-new button is clicked', () => {
    const listener = vi.fn()
    window.addEventListener('toggle-whatsnew', listener)
    render(<StatusBar />)
    const btn = screen.getByLabelText("What's new")
    fireEvent.click(btn)
    expect(listener).toHaveBeenCalled()
    window.removeEventListener('toggle-whatsnew', listener)
  })

  it('shows unseen count badge when there are new features', async () => {
    mockGetUnseenCount.mockReturnValue(3)
    render(<StatusBar />)
    await waitFor(() => {
      expect(screen.getByText('3')).toBeDefined()
    })
  })

  it('caps unseen count badge at 9+', async () => {
    mockGetUnseenCount.mockReturnValue(15)
    render(<StatusBar />)
    await waitFor(() => {
      expect(screen.getByText('9+')).toBeDefined()
    })
  })

  it('says Reconnecting with the last known model instead of stale healthy text', async () => {
    mockHealthState.mockReturnValue('stale')
    render(<StatusBar />)
    // Flush the souls-controller promise inside act — otherwise it resolves
    // after the assertions and React warns about an update outside act.
    await act(async () => {})
    // Before the fix the else-branch printed health_summary for any status
    // with health present — the bar kept claiming "Healthy" through a drop.
    expect(screen.getByText(/Reconnecting/)).toBeDefined()
    expect(screen.queryByText('Healthy')).toBeNull()
  })

  it('offline Retry probes the backend through the shared helper (never reloads)', async () => {
    mockHealthState.mockReturnValue('offline')
    render(<StatusBar />)
    await act(async () => {})
    // The remedy lives in the details panel — expand, then retry.
    fireEvent.click(screen.getByRole('button', { name: 'Expand details' }))
    fireEvent.click(screen.getByRole('button', { name: 'Retry connection' }))
    expect(probeBackendAndRevalidateMock).toHaveBeenCalledTimes(1)
  })

  it('errs badge counts only errors inside the recency window', async () => {
    // The store has no setState on the hook — seed through addError with a
    // fake clock: one error 20× deduped an hour ago, one fresh failure.
    // Capture real now FIRST: under fake timers Date.now() reads the rewound
    // clock, so computing offsets from it would age both errors past the TTL.
    const realNow = Date.now()
    vi.useFakeTimers()
    try {
      vi.setSystemTime(realNow - ERROR_BADGE_TTL_MS - 60_000)
      for (let i = 0; i < 20; i++) {
        useErrorStore.getState().addError(new Error('old failure from an hour ago'), {
          source: 'test',
        })
      }
      vi.setSystemTime(realNow - 60_000)
      useErrorStore.getState().addError(new Error('fresh failure just now'), { source: 'test' })
    } finally {
      vi.useRealTimers()
    }
    render(<StatusBar />)
    await act(async () => {})
    // Before the fix the badge summed every undismissed error forever —
    // "live · 21 errs" with nothing recent and no way to clear it. Now only
    // the fresh failure counts; the 20×-deduped hour-old one is TTL'd out.
    expect(screen.getByText('1 err')).toBeDefined()
    expect(screen.queryByText(/21/)).toBeNull()
  })

  it('clicking the errs badge clears the stale backlog', async () => {
    useErrorStore.getState().addError(new Error('boom'), { source: 'test' })
    render(<StatusBar />)
    await act(async () => {})
    expect(screen.getByText('1 err')).toBeDefined()
    fireEvent.click(screen.getByRole('button', { name: /clear error count/i }))
    expect(screen.queryByRole('button', { name: /clear error count/i })).toBeNull()
  })
})
