// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'

vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (m: string, t?: string) => void }) => unknown) =>
    selector({ addToast: vi.fn() }),
}))
vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k, locale: 'en', setLocale: vi.fn() }),
  LocaleProvider: ({ children }: { children: React.ReactNode }) => children,
}))
vi.mock('@/lib/dev-log', () => ({
  logger: { warning: vi.fn(), info: vi.fn(), error: vi.fn() },
}))
vi.mock('@/hooks/useLiveStatus', () => ({
  useLiveStatus: () => ({
    health: { model_loaded: true, model_type: 'qwen', health_score: 83 },
    connectionStatus: 'connected',
  }),
}))

import { apiGet, apiPost } from '@/lib/http-client'
import { OverlayDock } from '@/components/overlay/OverlayDock'
import { useMoleDockWidget } from './MoleDock'
import type { MoleReport, MoleReportResponse } from './types'

const get = vi.mocked(apiGet)
const post = vi.mocked(apiPost)

function reportFixture(over: Partial<MoleReport> = {}): MoleReport {
  return {
    schema_version: 1,
    ts: 1,
    targets: {},
    probes: [{ name: 'http', ok: true }],
    findings: [
      {
        source: 'http',
        check: 'api.health_score',
        severity: 'ok',
        score: 92,
        message: 'API health score 92 — healthy',
        component: 'api',
      },
    ],
    summary: { total: 3, by_severity: { ok: 2, info: 0, warn: 0, critical: 1 } },
    overall: 'critical',
    ...over,
  } as MoleReport
}

/** Renders the real hook through the real dock — the shape AppLayout uses. */
function Harness() {
  const widget = useMoleDockWidget()
  return <OverlayDock widgets={[widget]} />
}

afterEach(cleanup)

describe('MoleDock (site doctor dock tenant)', () => {
  beforeEach(() => {
    get.mockReset()
    post.mockReset()
  })

  it('loads GET /mole/report once on mount', async () => {
    get.mockResolvedValue({ report: reportFixture(), path: '', age_s: 0 } as MoleReportResponse)
    render(<Harness />)
    await waitFor(() => expect(get).toHaveBeenCalledTimes(1))
    expect(get.mock.calls[0][0]).toBe('/mole/report')
  })

  it('badges the worst actionable severity count on the operator-tagged button', async () => {
    get.mockResolvedValue({ report: reportFixture(), path: '', age_s: 0 } as MoleReportResponse)
    render(<Harness />)
    const badge = await screen.findByTestId('overlay-badge-mole')
    expect(badge).toHaveTextContent('1') // critical: 1
    const button = screen.getByTestId('overlay-widget-mole')
    expect(button).toHaveAttribute('aria-label', 'nav.mole')
    expect(button.closest('[data-role]')).toHaveAttribute('data-role', 'operator')
  })

  it('stays badgeless and default-ringed while there is no report yet', async () => {
    get.mockResolvedValue({ report: null, path: '', age_s: null } as MoleReportResponse)
    render(<Harness />)
    await waitFor(() => expect(get).toHaveBeenCalled())
    expect(screen.queryByTestId('overlay-badge-mole')).toBeNull()
    expect(screen.getByTestId('overlay-widget-mole').parentElement).toHaveAttribute(
      'data-role',
      'operator',
    )
  })

  it('panel shows summary + findings + run action for a loaded report', async () => {
    get.mockResolvedValue({ report: reportFixture(), path: '', age_s: 0 } as MoleReportResponse)
    render(<Harness />)
    fireEvent.click(await screen.findByTestId('overlay-widget-mole'))

    expect(await screen.findByTestId('mole-summary')).toBeDefined()
    expect(screen.getByTestId('findings-list')).toBeDefined()
    expect(screen.getByTestId('run-mole-button')).toBeDefined()
    expect(screen.getByTestId('mole-dock-panel').closest('[role="dialog"]')).toHaveAttribute(
      'aria-label',
      'nav.mole',
    )
  })

  it('refreshes the report after a successful run', async () => {
    get.mockResolvedValue({ report: reportFixture(), path: '', age_s: 0 } as MoleReportResponse)
    post.mockResolvedValue({ report: reportFixture() })
    render(<Harness />)
    fireEvent.click(await screen.findByTestId('overlay-widget-mole'))
    fireEvent.click(await screen.findByTestId('run-mole-button'))
    await waitFor(() => expect(get).toHaveBeenCalledTimes(2))
    expect(post.mock.calls[0][0]).toBe('/mole/run')
  })

  it('shows the empty state when no report exists', async () => {
    get.mockResolvedValue({ report: null, path: '', age_s: null } as MoleReportResponse)
    render(<Harness />)
    fireEvent.click(await screen.findByTestId('overlay-widget-mole'))
    expect(await screen.findByTestId('mole-dock-empty')).toHaveTextContent('No report yet')
    expect(screen.queryByTestId('mole-summary')).toBeNull()
  })

  it('surfaces a load failure as an alert and does not fake an empty report', async () => {
    get.mockRejectedValue(new Error('offline'))
    render(<Harness />)
    fireEvent.click(await screen.findByTestId('overlay-widget-mole'))
    const alert = await screen.findByTestId('mole-dock-error')
    expect(alert).toHaveAttribute('role', 'alert')
    expect(alert).toHaveTextContent('offline')
    expect(screen.queryByTestId('mole-dock-empty')).toBeNull()
  })
})
