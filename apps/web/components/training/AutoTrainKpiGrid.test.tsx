import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { AutoTrainKpiGrid } from './AutoTrainKpiGrid'

afterEach(() => cleanup())

describe('AutoTrainKpiGrid', () => {
  it('shows loading skeletons', () => {
    const { container } = render(<AutoTrainKpiGrid checkpointCount={0} completedCount={0} trainingRunning={false} loss={null} loading={true} />)
    expect(container.querySelectorAll('[class*="animate-pulse"]').length).toBeGreaterThanOrEqual(1)
  })
  it('shows stats when loaded', () => {
    render(<AutoTrainKpiGrid checkpointCount={5} completedCount={3} trainingRunning={true} loss={0.4231} loading={false} />)
    expect(screen.getAllByText('5').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Training').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('0.4231').length).toBeGreaterThanOrEqual(1)
  })
  it('shows Idle when not training', () => {
    render(<AutoTrainKpiGrid checkpointCount={0} completedCount={0} trainingRunning={false} loss={null} loading={false} />)
    expect(screen.getAllByText('Idle').length).toBeGreaterThanOrEqual(1)
  })
  it('shows -- when loss is null', () => {
    render(<AutoTrainKpiGrid checkpointCount={0} completedCount={0} trainingRunning={false} loss={null} loading={false} />)
    expect(screen.getAllByText('--').length).toBeGreaterThanOrEqual(1)
  })
  it('shows labels', () => {
    render(<AutoTrainKpiGrid checkpointCount={0} completedCount={0} trainingRunning={false} loss={null} loading={false} />)
    expect(screen.getAllByText('Checkpoints').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Trained').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Status').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Current loss').length).toBeGreaterThanOrEqual(1)
  })
})
