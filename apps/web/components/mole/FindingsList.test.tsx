// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import { FindingsList } from './FindingsList'
import type { MoleFinding } from './types'

afterEach(cleanup)

const FINDINGS: MoleFinding[] = [
  {
    source: 'http',
    check: 'api.reachable',
    severity: 'critical',
    score: 0,
    message: 'API unreachable',
    detail: 'GET http://localhost:8000/health → timed out',
    component: 'api',
  },
  {
    source: 'sse',
    check: 'stream.frames',
    severity: 'warn',
    score: 60,
    message: 'No frames from /health/stream',
    component: 'api',
  },
  {
    source: 'journey',
    check: 'journey.report',
    severity: 'info',
    score: 90,
    message: 'Journey report missing — no sweep has been recorded',
    component: 'journeys',
  },
  {
    source: 'http',
    check: 'api.errors',
    severity: 'ok',
    score: 100,
    message: 'Error volume nominal',
    component: 'api',
  },
]

describe('FindingsList', () => {
  it('groups findings worst-first', () => {
    const { container } = render(<FindingsList findings={FINDINGS} />)
    const headings = Array.from(container.querySelectorAll('section h3')).map(
      (h) => h.textContent ?? '',
    )
    expect(headings[0]).toMatch(/Critical/)
    expect(headings[1]).toMatch(/Warning/)
    expect(headings[2]).toMatch(/Info/)
    expect(headings[3]).toMatch(/OK/)
  })

  it('renders severity marker, component tag, check and message per row', () => {
    render(<FindingsList findings={FINDINGS} />)
    const rows = screen.getAllByTestId('finding-row')
    expect(rows).toHaveLength(4)
    expect(screen.getByText('api.reachable')).toBeInTheDocument()
    expect(screen.getByText('API unreachable')).toBeInTheDocument()
    expect(screen.getByText('journeys')).toBeInTheDocument()
  })

  it('keeps detail collapsed behind a disclosure by default', () => {
    const { container } = render(<FindingsList findings={FINDINGS} />)
    const disclosures = container.querySelectorAll('details')
    expect(disclosures.length).toBeGreaterThanOrEqual(1)
    for (const details of disclosures) {
      expect(details.open).toBe(false)
    }
  })

  it('expands the detail when the disclosure is activated', () => {
    const { container } = render(<FindingsList findings={FINDINGS} />)
    const details = container.querySelector('details')!
    const summary = details.querySelector('summary')!
    fireEvent.click(summary)
    expect(details.open).toBe(true)
    expect(details.textContent).toContain('GET http://localhost:8000/health')
  })

  it('labels each disclosure for screen readers', () => {
    const { container } = render(<FindingsList findings={FINDINGS} />)
    const details = container.querySelector('details')!
    expect(details.getAttribute('aria-label')).toContain('api.reachable')
    expect(details.getAttribute('aria-label')).toContain('Critical')
  })

  it('renders rows without detail as plain labelled rows', () => {
    render(
      <FindingsList
        findings={[{ ...FINDINGS[0], detail: undefined, component: undefined }]}
      />,
    )
    const row = screen.getByTestId('finding-row')
    expect(row.getAttribute('aria-label')).toContain('api.reachable')
    expect(row.querySelectorAll('details')).toHaveLength(0)
  })

  it('shows an empty state when there are no findings', () => {
    render(<FindingsList findings={[]} />)
    expect(screen.getByTestId('findings-empty')).toHaveTextContent('No findings recorded.')
  })
})
