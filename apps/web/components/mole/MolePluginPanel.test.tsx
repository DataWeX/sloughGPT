import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'

vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
}))
vi.mock('@/hooks/useLiveStatus', () => ({
  useLiveStatus: () => ({ health: null, connectionStatus: 'connected', connected: true }),
}))
vi.mock('@/lib/dev-log', () => ({
  logger: { debug: vi.fn(), info: vi.fn(), warning: vi.fn(), error: vi.fn() },
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (m: string) => void }) => unknown) =>
    selector({ addToast: vi.fn() }),
}))

import { MolePluginPanel } from './MolePluginPanel'
import { apiGet } from '@/lib/http-client'
import type { MoleReportResponse } from './types'

const get = vi.mocked(apiGet)

const REPORT = {
  schema_version: 1,
  ts: Math.floor(Date.now() / 1000) - 30,
  targets: { web: 'http://localhost:5173', api: 'http://localhost:8000' },
  probes: [
    { name: 'http', ok: true, error: '' },
    { name: 'sse', ok: true, error: '' },
    { name: 'journey', ok: false, error: 'no report — run with sweep' },
  ],
  findings: [
    {
      source: 'http',
      check: 'api.health_score',
      severity: 'warn',
      score: 70,
      message: 'Health score degraded (62)',
      detail: 'recent_errors: 3',
      component: 'api',
    },
    {
      source: 'sse',
      check: 'stream.frames',
      severity: 'ok',
      score: 100,
      message: 'SSE cadence nominal (3 frames)',
      component: 'api',
    },
  ],
  summary: { total: 2, by_severity: { ok: 1, info: 0, warn: 1, critical: 0 } },
  overall: 'warn',
}

describe('MolePluginPanel', () => {
  afterEach(cleanup)

  beforeEach(() => {
    get.mockReset()
  })

  it('renders nothing while the plugin list has not resolved', () => {
    get.mockReturnValue(new Promise(() => {}) as never)
    const { container } = render(<MolePluginPanel plugin={null} />)
    expect(container.firstChild).toBeNull()
  })

  it('shows the disabled empty-state when the plugin is disabled', () => {
    render(<MolePluginPanel plugin={{ enabled: false }} />)
    expect(screen.getByText('Mole plugin disabled')).toBeInTheDocument()
    expect(get).not.toHaveBeenCalled()
  })

  it('shows the empty state when enabled but no report exists', async () => {
    get.mockResolvedValue({
      report: null,
      path: '/home/user/.cache/slog-doctor/findings-report.json',
      age_s: null,
    } as MoleReportResponse)
    render(<MolePluginPanel plugin={{ enabled: true }} />)
    expect(await screen.findByText('No report yet — run a check')).toBeInTheDocument()
    expect(screen.getByTestId('run-mole-button')).toBeInTheDocument()
  })

  it('renders summary, findings and the live component strip when enabled', async () => {
    get.mockResolvedValue({ report: REPORT, path: '/tmp/report.json', age_s: 30 })
    render(<MolePluginPanel plugin={{ enabled: true }} />)
    expect(await screen.findByTestId('mole-summary')).toBeInTheDocument()
    expect(screen.getByTestId('mole-overall')).toHaveTextContent('Overall Warning')
    expect(screen.getByTestId('findings-list')).toBeInTheDocument()
    expect(screen.getByText('Health score degraded (62)')).toBeInTheDocument()
    expect(screen.getByText('Live components')).toBeInTheDocument()
  })

  it('keeps finding detail collapsed by default', async () => {
    get.mockResolvedValue({ report: REPORT, path: '/tmp/report.json', age_s: 30 })
    const { container } = render(<MolePluginPanel plugin={{ enabled: true }} />)
    await screen.findByTestId('findings-list')
    const disclosures = container.querySelectorAll('details')
    expect(disclosures.length).toBeGreaterThanOrEqual(1)
    for (const details of disclosures) {
      expect(details.open).toBe(false)
    }
  })

  it('shows an error state when the report cannot be loaded', async () => {
    get.mockRejectedValue(new Error('Connection unavailable — server may be starting up'))
    render(<MolePluginPanel plugin={{ enabled: true }} />)
    await waitFor(() => {
      const alerts = screen.getAllByRole('alert')
      expect(alerts.length).toBeGreaterThan(0)
      expect(alerts.some((a) => a.textContent?.includes('Connection unavailable'))).toBe(true)
    })
  })

  it('loads the report only when the plugin is enabled', async () => {
    get.mockResolvedValue({ report: null, path: '/tmp/report.json', age_s: null })
    render(<MolePluginPanel plugin={{ enabled: false }} />)
    await new Promise((r) => setTimeout(r, 0))
    expect(get).not.toHaveBeenCalled()
  })
})
