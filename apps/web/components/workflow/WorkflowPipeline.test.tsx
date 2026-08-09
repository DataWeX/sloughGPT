// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { WorkflowPipeline } from './WorkflowPipeline'
import type { WorkflowStatus } from '@/lib/workflow-controller'

beforeEach(() => { vi.clearAllMocks() })
afterEach(() => { cleanup() })

const runningStatus: WorkflowStatus = {
  running: true,
  config: {
    aggregate_interval_minutes: 30,
    prune_interval_minutes: 60,
    export_interval_hours: 24,
    health_check_interval_seconds: 30,
  },
  stats: {
    feedback_records: 150,
    adapters_count: 3,
    last_aggregate: new Date(Date.now() - 600000).toISOString(),
    last_prune: new Date(Date.now() - 3600000).toISOString(),
  },
}

const stoppedStatus: WorkflowStatus = {
  running: false,
  config: runningStatus.config,
  stats: runningStatus.stats,
}

describe('WorkflowPipeline', () => {
  it('returns null when no status', () => {
    const { container } = render(<WorkflowPipeline status={null} />)
    expect(container.querySelector('[data-testid="workflow-pipeline"]')).toBeNull()
  })

  it('returns null when no config', () => {
    const { container } = render(<WorkflowPipeline status={{ running: false }} />)
    expect(container.querySelector('[data-testid="workflow-pipeline"]')).toBeNull()
  })

  it('renders pipeline card when status has config', () => {
    render(<WorkflowPipeline status={runningStatus} />)
    expect(screen.getAllByTestId('workflow-pipeline').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Pipeline').length).toBeGreaterThanOrEqual(1)
  })

  it('shows all 5 pipeline steps', () => {
    render(<WorkflowPipeline status={runningStatus} />)
    expect(screen.getAllByText('Feedback').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Aggregate').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Train').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Prune').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Export').length).toBeGreaterThanOrEqual(1)
  })

  it('shows step numbers 1-5', () => {
    render(<WorkflowPipeline status={runningStatus} />)
    expect(screen.getAllByText('1').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('5').length).toBeGreaterThanOrEqual(1)
  })

  it('shows interval labels', () => {
    render(<WorkflowPipeline status={runningStatus} />)
    expect(screen.getAllByText('continuous').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('30m').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('60m').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('24h').length).toBeGreaterThanOrEqual(1)
  })

  it('shows feedback record count', () => {
    render(<WorkflowPipeline status={runningStatus} />)
    expect(screen.getAllByText('150 feedback records').length).toBeGreaterThanOrEqual(1)
  })

  it('shows Active and Idle legend', () => {
    render(<WorkflowPipeline status={runningStatus} />)
    expect(screen.getAllByText('Active').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Idle').length).toBeGreaterThanOrEqual(1)
  })

  it('shows last run times for steps with data', () => {
    render(<WorkflowPipeline status={runningStatus} />)
    expect(screen.getAllByText(/ago/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows never for steps without last run', () => {
    const statusNoRuns: WorkflowStatus = {
      running: true,
      config: runningStatus.config,
      stats: { feedback_records: 0, adapters_count: 0 },
    }
    render(<WorkflowPipeline status={statusNoRuns} />)
    expect(screen.getAllByText('never').length).toBeGreaterThanOrEqual(1)
  })
})
