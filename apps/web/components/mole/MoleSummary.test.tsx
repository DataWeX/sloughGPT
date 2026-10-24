// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'

import { MoleSummary } from './MoleSummary'
import type { MoleFinding, MoleReport } from './types'

afterEach(cleanup)

const FINDINGS: MoleFinding[] = [
  {
    source: 'http',
    check: 'api.health_score',
    severity: 'critical',
    score: 0,
    message: 'API unreachable',
    detail: 'GET /health → timed out',
    component: 'api',
  },
  {
    source: 'journey',
    check: 'journey.flows',
    severity: 'warn',
    score: 60,
    message: '1/4 journeys failed: Sign up',
    component: 'journeys',
  },
]

function makeReport(overrides: Partial<MoleReport> = {}): MoleReport {
  return {
    schema_version: 1,
    ts: Math.floor(Date.now() / 1000) - 120,
    targets: { web: 'http://localhost:5173', api: 'http://localhost:8000' },
    probes: [
      { name: 'http', ok: false, error: 'boom' },
      { name: 'sse', ok: true },
      { name: 'journey', ok: true },
    ],
    findings: FINDINGS,
    summary: {
      total: 4,
      by_severity: { ok: 1, info: 1, warn: 1, critical: 1 },
    },
    overall: 'critical',
    ...overrides,
  }
}

describe('MoleSummary', () => {
  it('renders the overall severity pill', () => {
    render(<MoleSummary report={makeReport()} />)
    expect(screen.getByTestId('mole-overall')).toHaveTextContent('Overall Critical')
  })

  it('renders severity counts', () => {
    render(<MoleSummary report={makeReport()} />)
    expect(screen.getByTestId('severity-count-critical')).toHaveTextContent('1Critical')
    expect(screen.getByTestId('severity-count-warn')).toHaveTextContent('1Warning')
    expect(screen.getByTestId('severity-count-info')).toHaveTextContent('1Info')
    expect(screen.getByTestId('severity-count-ok')).toHaveTextContent('1OK')
  })

  it('renders probe and finding totals', () => {
    render(<MoleSummary report={makeReport()} />)
    expect(screen.getByText(/4 findings/)).toBeInTheDocument()
    expect(screen.getByText(/2\/3 probes ok/)).toBeInTheDocument()
  })

  it('renders report age from the report timestamp', () => {
    render(<MoleSummary report={makeReport()} />)
    expect(screen.getByText(/Checked 2m ago/)).toBeInTheDocument()
  })

  it('falls back to age_s when the report has no timestamp', () => {
    render(<MoleSummary report={makeReport({ ts: 0 })} ageS={30} />)
    expect(screen.getByText(/Checked 30s ago/)).toBeInTheDocument()
  })

  it('renders info/warn pill tones without raw colors', () => {
    const { container } = render(<MoleSummary report={makeReport({ overall: 'warn' })} />)
    expect(screen.getByTestId('mole-overall')).toHaveTextContent('Overall Warning')
    expect(container.innerHTML).not.toMatch(/#[0-9a-fA-F]{3,8}\b/)
    expect(container.innerHTML).not.toMatch(/rgba?\(/)
  })
})
