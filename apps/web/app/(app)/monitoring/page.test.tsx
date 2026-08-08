import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, act } from '@testing-library/react'

vi.mock('@/lib/system-controller', () => ({
  systemController: {
    getDetailedHealth: vi.fn().mockResolvedValue(null),
    getMetrics: vi.fn().mockResolvedValue(null),
    getDisk: vi.fn().mockResolvedValue(null),
    getInfo: vi.fn().mockResolvedValue(null),
    getExecutorStatus: vi.fn().mockResolvedValue(null),
    getInferencePoolStatus: vi.fn().mockResolvedValue(null),
    getProcessGuardStatus: vi.fn().mockResolvedValue(null),
  },
}))

vi.mock('@/lib/benchmark-controller', () => ({
  benchmarkController: { getMetrics: vi.fn().mockResolvedValue(null) },
}))

vi.mock('@/lib/knowledge-controller', () => ({
  knowledgeController: { getStats: vi.fn().mockResolvedValue(null), list: vi.fn().mockResolvedValue([]) },
}))

vi.mock('@/lib/controllers', () => ({
  multimodalController: { getCapabilities: vi.fn().mockResolvedValue(null) },
}))

vi.mock('@/lib/training-controller', () => ({
  trainingController: {
    getTrainingJobs: vi.fn().mockResolvedValue([]),
    getRecoveryStats: vi.fn().mockResolvedValue(null),
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

import MonitoringPage from './page'

describe('MonitoringPage', () => {
  beforeEach(() => { vi.clearAllMocks() })
  afterEach(() => { cleanup() })

  it('renders without crashing', async () => {
    render(<MonitoringPage />)
    expect(document.body).toBeTruthy()
    await act(async () => {})
  })

  it('renders page header', async () => {
    render(<MonitoringPage />)
    expect(screen.getAllByText(/system health|monitoring/i).length).toBeGreaterThanOrEqual(1)
    await act(async () => {})
  })
})
