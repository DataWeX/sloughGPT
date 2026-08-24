import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act, cleanup } from '@testing-library/react'
import { systemController } from '@/lib/system-controller'
import { trainingController } from '@/lib/training-controller'
import { knowledgeController } from '@/lib/knowledge-controller'
import { benchmarkController } from '@/lib/benchmark-controller'
import { multimodalController } from '@/lib/controllers'

const { mockDetailedHealth, mockUseLiveStatus } = vi.hoisted(() => ({
  mockDetailedHealth: {
    status: 'healthy', uptime_seconds: 3600, timestamp: new Date().toISOString(),
    request_count: 100, error_count: 2, avg_latency_ms: 45, p95_latency_ms: 120,
    requests_per_minute: 5, path_latencies: [], recent_errors: [],
    inference_count: 50, total_tokens: 10000, tokens_per_sec: 25, avg_tokens_per_request: 200,
    health_score: { score: 85, status: 'healthy' }, status_message: 'OK',
    model_metrics: [], model_events: [], health_history: [], memory_history: [], rate_violations: [],
    system: { cpu_percent: 45.2, memory_percent: 62.1, memory_available_mb: 8192 },
    model_loaded: true, model_loading: false, model_type: 'gpt2', soul: 'default',
    inference: { is_inferencing: false, inference_count: 50 },
    kv_sessions: { enabled: true, active_sessions: 3, cached_tokens: 120 },
    training_pool: { active_jobs: 0, max_workers: 2, total_tracked: 0 },
  },
  mockUseLiveStatus: { health: null as Record<string, unknown> | null, connectionStatus: 'disconnected' as string },
}))

vi.mock('@/lib/system-controller', () => ({
  systemController: {
    getDetailedHealth: vi.fn().mockResolvedValue(mockDetailedHealth),
    getMetrics: vi.fn().mockResolvedValue({ cpu_percent: 45, memory_percent: 62, memory_used_gb: 8, memory_total_gb: 16 }),
    getDisk: vi.fn().mockResolvedValue({ total_gb: 500, used_gb: 275, free_gb: 225, percent: 55 }),
    getInfo: vi.fn().mockResolvedValue({ platform: 'linux', platform_release: '5.15', platform_version: '#1 SMP', architecture: 'x86_64', processor: 'Intel', cpu_count: 8 }),
    getExecutorStatus: vi.fn().mockResolvedValue({ initialized: true, active_jobs: 0, max_workers: 2, total_tracked: 0, jobs: [] }),
    getInferencePoolStatus: vi.fn().mockResolvedValue(null),
    getProcessGuardStatus: vi.fn().mockResolvedValue(null),
  },
}))

vi.mock('@/lib/benchmark-controller', () => ({
  benchmarkController: {
    quality: vi.fn().mockResolvedValue(null),
    stats: vi.fn().mockResolvedValue(null),
  },
}))

vi.mock('@/lib/knowledge-controller', () => ({
  knowledgeController: {
    stats: vi.fn().mockResolvedValue(null),
    getAdapterStatus: vi.fn().mockResolvedValue(null),
    list: vi.fn().mockResolvedValue([]),
  },
}))

vi.mock('@/lib/controllers', () => ({
  multimodalController: {
    getDPOStatus: vi.fn().mockResolvedValue(null),
    getStatus: vi.fn().mockResolvedValue(null),
    getCapabilities: vi.fn().mockResolvedValue(null),
  },
}))

vi.mock('@/lib/training-controller', () => ({
  trainingController: {
    list: vi.fn().mockResolvedValue([]),
    getRecoveryStats: vi.fn().mockResolvedValue(null),
    getAutoTrainStatus: vi.fn().mockResolvedValue(null),
    getStatus: vi.fn().mockResolvedValue(null),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => ({ addToast: vi.fn() }),
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
}))

vi.mock('@/lib/format-bytes', () => ({
  todayDateString: () => '2026-08-07',
  getJsonItem: vi.fn().mockReturnValue([]),
}))

vi.mock('@/lib/error-utils', () => ({
  extractErrorMessage: vi.fn().mockReturnValue('Error'),
}))

vi.mock('@/components/monitoring/StatusCard', () => ({ StatusCard: () => <div data-testid="status-card" /> }))
vi.mock('@/components/monitoring/ResourceCard', () => ({ ResourceCard: () => <div data-testid="resource-card" /> }))
vi.mock('@/components/monitoring/ProcessCard', () => ({ ProcessCard: () => <div data-testid="process-card" /> }))
vi.mock('@/components/monitoring/TrafficCard', () => ({ TrafficCard: () => <div data-testid="traffic-card" /> }))
vi.mock('@/components/monitoring/LatencyCard', () => ({ LatencyCard: () => <div data-testid="latency-card" /> }))
vi.mock('@/components/monitoring/PathLatenciesCard', () => ({ PathLatenciesCard: () => <div data-testid="path-latencies-card" /> }))
vi.mock('@/components/monitoring/ServerErrorsCard', () => ({ ServerErrorsCard: () => <div data-testid="server-errors-card" /> }))
vi.mock('@/components/monitoring/RateViolationsCard', () => ({ RateViolationsCard: () => <div data-testid="rate-violations-card" /> }))
vi.mock('@/components/monitoring/AlertPanel', () => ({ AlertPanel: () => <div data-testid="alert-panel" /> }))
vi.mock('@/components/monitoring/QualityCard', () => ({ QualityCard: () => <div data-testid="quality-card" /> }))
vi.mock('@/components/monitoring/ModelMetricsCard', () => ({ ModelMetricsCard: () => <div data-testid="model-metrics-card" /> }))
vi.mock('@/components/monitoring/ModelEventsCard', () => ({ ModelEventsCard: () => <div data-testid="model-events-card" /> }))
vi.mock('@/components/monitoring/DiagnosticsCard', () => ({ DiagnosticsCard: () => <div data-testid="diagnostics-card" /> }))
vi.mock('@/components/monitoring/KnowledgeCard', () => ({ KnowledgeCard: () => <div data-testid="knowledge-card" /> }))
vi.mock('@/components/monitoring/AutoTrainCard', () => ({ AutoTrainCard: () => <div data-testid="autotrain-card" /> }))
vi.mock('@/components/monitoring/FeedbackCard', () => ({ FeedbackCard: () => <div data-testid="feedback-card" /> }))
vi.mock('@/components/monitoring/ExecutorPool', () => ({ ExecutorPool: () => <div data-testid="executor-pool" /> }))
vi.mock('@/components/monitoring/KvCacheCard', () => ({ KvCacheCard: () => <div data-testid="kv-cache-card" /> }))
vi.mock('@/components/monitoring/SystemInfoCards', () => ({
  GpuCard: () => <div data-testid="gpu-card" />,
  DiskCard: () => <div data-testid="disk-card" />,
  ServerInfoCard: () => <div data-testid="server-info-card" />,
}))
vi.mock('@/components/monitoring/TrainingHistory', () => ({ TrainingHistory: () => <div data-testid="training-history" /> }))
vi.mock('@/components/ActivityTicker', () => ({
  ActivityTicker: () => <div data-testid="activity-ticker" />,
  ErrorList: () => <div data-testid="error-list" />,
}))
vi.mock('@/components/OutputCard', () => ({ OutputCard: () => <div data-testid="output-card" /> }))
vi.mock('@/components/monitoring/SystemChart', () => ({ SystemChart: () => <div data-testid="system-chart" /> }))
vi.mock('@/components/monitoring/TrendChart', () => ({ TrendChart: () => <div data-testid="trend-chart" /> }))
vi.mock('@/components/monitoring/WorkflowCard', () => ({ WorkflowCard: () => <div data-testid="workflow-card" /> }))
vi.mock('@/hooks/useLiveStatus', () => ({
  useLiveStatus: () => mockUseLiveStatus,
}))

import MonitoringPage from './page'

describe('MonitoringPage — initial load flow', () => {
  afterEach(() => { cleanup() })

  it('renders without crashing', async () => {
    render(<MonitoringPage />)
    expect(document.body).toBeTruthy()
    await act(async () => {})
  })

  it('renders page header', async () => {
    render(<MonitoringPage />)
    expect(screen.getAllByText(/system health/i).length).toBeGreaterThanOrEqual(1)
    await act(async () => {})
  })

  it('shows System Health title', async () => {
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/system health/i).length).toBeGreaterThanOrEqual(1)
    })
  })
})

describe('MonitoringPage — auto-refresh flow', () => {
  beforeEach(() => { vi.clearAllMocks(); mockUseLiveStatus.connectionStatus = 'disconnected'; mockUseLiveStatus.health = null })
  afterEach(() => { cleanup() })

  it('auto-refresh toggle present', async () => {
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/auto/i).length).toBeGreaterThanOrEqual(1)
    })
    await act(async () => {})
  })
})

describe('MonitoringPage — refresh flow', () => {
  beforeEach(() => { vi.clearAllMocks(); mockUseLiveStatus.connectionStatus = 'disconnected'; mockUseLiveStatus.health = null })
  afterEach(() => { cleanup() })

  it('refresh button calls fetchAll', async () => {
    const { systemController } = await import('@/lib/system-controller')
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/refresh/i).length).toBeGreaterThanOrEqual(1)
    })
    const refreshBtn = screen.getAllByText(/refresh/i)[0]
    await act(async () => { fireEvent.click(refreshBtn) })
    expect(systemController.getDetailedHealth).toHaveBeenCalled()
    await act(async () => {})
  })
})

describe('MonitoringPage — essential cards always visible', () => {
  beforeEach(() => { vi.clearAllMocks(); mockUseLiveStatus.connectionStatus = 'disconnected'; mockUseLiveStatus.health = null })
  afterEach(() => { cleanup() })

  it('renders status card', async () => {
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(screen.getByTestId('status-card')).toBeTruthy()
    })
    await act(async () => {})
  })

  it('renders resource card', async () => {
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(screen.getByTestId('resource-card')).toBeTruthy()
    })
    await act(async () => {})
  })

  it('renders alert panel', async () => {
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(screen.getByTestId('alert-panel')).toBeTruthy()
    })
    await act(async () => {})
  })

  it('renders server errors card', async () => {
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(screen.getByTestId('server-errors-card')).toBeTruthy()
    })
    await act(async () => {})
  })

  it('renders trend chart', async () => {
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(screen.getByTestId('trend-chart')).toBeTruthy()
    })
    await act(async () => {})
  })
})

describe('MonitoringPage — collapsed sections exist', () => {
  beforeEach(() => { vi.clearAllMocks(); mockUseLiveStatus.connectionStatus = 'disconnected'; mockUseLiveStatus.health = null })
  afterEach(() => { cleanup() })

  it('has Diagnostics section header', async () => {
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/diagnostics/i).length).toBeGreaterThanOrEqual(1)
    })
    await act(async () => {})
  })

  it('has Training & Quality section header', async () => {
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/training/i).length).toBeGreaterThanOrEqual(1)
    })
    await act(async () => {})
  })

  it('has System Info section header', async () => {
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/system info/i).length).toBeGreaterThanOrEqual(1)
    })
    await act(async () => {})
  })

  it('has Server Output section header', async () => {
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/service output/i).length).toBeGreaterThanOrEqual(1)
    })
    await act(async () => {})
  })
})

describe('MonitoringPage — error handling flow', () => {
  beforeEach(() => { vi.clearAllMocks(); mockUseLiveStatus.connectionStatus = 'disconnected'; mockUseLiveStatus.health = null })
  afterEach(() => { cleanup() })

  it('handles health fetch failure gracefully', async () => {
    const { systemController } = await import('@/lib/system-controller')
    vi.mocked(systemController.getDetailedHealth).mockRejectedValue(new Error('Network error'))
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/system health/i).length).toBeGreaterThanOrEqual(1)
    })
    await act(async () => {})
  })

  it('handles metrics fetch failure gracefully', async () => {
    const { systemController } = await import('@/lib/system-controller')
    vi.mocked(systemController.getMetrics).mockRejectedValue(new Error('Metrics error'))
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/system health/i).length).toBeGreaterThanOrEqual(1)
    })
    await act(async () => {})
  })
})

describe('MonitoringPage — data loading flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockUseLiveStatus.connectionStatus = 'disconnected'
    mockUseLiveStatus.health = null
    const sc = vi.mocked(systemController)
    sc.getDetailedHealth.mockResolvedValue(mockDetailedHealth)
    sc.getMetrics.mockResolvedValue({ cpu_percent: 45, memory_percent: 62, memory_used_gb: 8, memory_total_gb: 16 })
    sc.getInfo.mockResolvedValue({ platform: 'linux', platform_release: '5.15', platform_version: '#1 SMP', architecture: 'x86_64', processor: 'Intel', cpu_count: 8 })
    sc.getDisk.mockResolvedValue({ total_gb: 500, used_gb: 275, free_gb: 225, percent: 55 })
    sc.getExecutorStatus.mockResolvedValue({ initialized: true, active_jobs: 0, max_workers: 2, total_tracked: 0, jobs: [] })
    sc.getProcessGuardStatus.mockResolvedValue({ enabled: false, active: false, pid: null, crash_count: 0, restart_count: 0, last_error: null, worker_state: null, pending_restart: false } as any)
    sc.getInferencePoolStatus.mockResolvedValue({ initialized: false } as any)
    vi.mocked(trainingController).list.mockResolvedValue([])
    vi.mocked(trainingController).getAutoTrainStatus.mockResolvedValue({ running: false } as any)
    vi.mocked(knowledgeController).stats.mockResolvedValue({ total_facts: 0 } as any)
    vi.mocked(knowledgeController).getAdapterStatus.mockResolvedValue({ has_adapter: false } as any)
    vi.mocked(benchmarkController).quality.mockResolvedValue({ count: 0, adapters: [] } as any)
    vi.mocked(benchmarkController).stats.mockResolvedValue({ total_runs: 0 } as any)
    vi.mocked(multimodalController).getDPOStatus.mockResolvedValue({ status: 'idle', accepted_count: 0, rejected_count: 0 } as any)
    vi.mocked(multimodalController).getStatus.mockResolvedValue({ enabled: false } as any)
  })
  afterEach(() => { cleanup() })

  it('calls systemController on mount', async () => {
    const { systemController } = await import('@/lib/system-controller')
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(systemController.getDetailedHealth).toHaveBeenCalled()
    })
    await act(async () => {})
  })

  it('calls getMetrics on mount', async () => {
    const { systemController } = await import('@/lib/system-controller')
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(systemController.getMetrics).toHaveBeenCalled()
    })
    await act(async () => {})
  })

  it('calls getDisk on mount', async () => {
    const { systemController } = await import('@/lib/system-controller')
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(systemController.getDisk).toHaveBeenCalled()
    })
    await act(async () => {})
  })

  it('calls getInfo on mount', async () => {
    const { systemController } = await import('@/lib/system-controller')
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(systemController.getInfo).toHaveBeenCalled()
    })
    await act(async () => {})
  })

  it('loads training jobs', async () => {
    const { trainingController } = await import('@/lib/training-controller')
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(trainingController.list).toHaveBeenCalled()
    })
    await act(async () => {})
  })
})

describe('MonitoringPage — connection-status gating', () => {
  beforeEach(() => { vi.clearAllMocks(); mockUseLiveStatus.connectionStatus = 'disconnected'; mockUseLiveStatus.health = null })
  afterEach(() => { cleanup() })

  it('does not re-poll when connectionStatus is disconnected', async () => {
    const { systemController } = await import('@/lib/system-controller')
    vi.mocked(systemController.getDetailedHealth).mockClear()
    render(<MonitoringPage />)
    await waitFor(() => { expect(systemController.getDetailedHealth).toHaveBeenCalled() })
    const countAfterInitial = vi.mocked(systemController.getDetailedHealth).mock.calls.length
    await act(async () => { await new Promise(r => setTimeout(r, 12000)) })
    expect(vi.mocked(systemController.getDetailedHealth).mock.calls.length).toBe(countAfterInitial)
  })
})

describe('MonitoringPage — fetchAll failure handling', () => {
  beforeEach(() => { vi.clearAllMocks(); mockUseLiveStatus.connectionStatus = 'disconnected'; mockUseLiveStatus.health = null })
  afterEach(() => { cleanup() })

  it('handles individual endpoint failure gracefully', async () => {
    const { systemController } = await import('@/lib/system-controller')
    vi.mocked(systemController.getDetailedHealth).mockRejectedValue(new Error('Network down'))
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(systemController.getDetailedHealth).toHaveBeenCalled()
    })
    await act(async () => {})
  })

  it('renders page structure even when all endpoints fail', async () => {
    const { systemController } = await import('@/lib/system-controller')
    vi.mocked(systemController.getDetailedHealth).mockRejectedValue(new Error('fail'))
    vi.mocked(systemController.getMetrics).mockRejectedValue(new Error('fail'))
    vi.mocked(systemController.getInfo).mockRejectedValue(new Error('fail'))
    vi.mocked(systemController.getDisk).mockRejectedValue(new Error('fail'))
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/system health/i).length).toBeGreaterThanOrEqual(1)
    })
    await act(async () => {})
  })

  it('keeps previously fetched data on partial failure', async () => {
    const { systemController } = await import('@/lib/system-controller')
    type Health = Awaited<ReturnType<typeof systemController.getDetailedHealth>>
    vi.mocked(systemController.getDetailedHealth).mockResolvedValue(mockDetailedHealth as unknown as Health)
    render(<MonitoringPage />)
    await waitFor(() => {
      expect(screen.getByTestId('status-card')).toBeTruthy()
    })
    vi.mocked(systemController.getDetailedHealth).mockRejectedValue(new Error('fail'))
    await act(async () => { fireEvent.click(screen.getAllByText(/refresh/i)[0]) })
    await waitFor(() => {
      expect(screen.getByTestId('status-card')).toBeTruthy()
      expect(screen.getAllByText(/error/i).length).toBeGreaterThanOrEqual(1)
    })
  })
})
