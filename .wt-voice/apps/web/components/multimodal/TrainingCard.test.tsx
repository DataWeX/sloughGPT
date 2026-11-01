import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  Line: () => <div data-testid="line" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
}))

import TrainingCard from './TrainingCard'

describe('TrainingCard', () => {
  afterEach(cleanup)

  const base = {
    report: { images_learned: 10, vocab_size: 256, unique_captions: 8, diversity_ratio: 0.75, accuracy_history: [0.5, 0.7, 0.8], caption_history: ['a cat', 'a dog'], replay_buffer_size: 128, trained: false, mean_accuracy: 0.7, last_accuracy: 0.8 },
    trainStatus: null,
  }

  it('renders KPI stats', () => {
    render(<TrainingCard {...base} />)
    expect(screen.getByText('10')).toBeDefined()
    expect(screen.getByText('256 words')).toBeDefined()
    expect(screen.getByText('8')).toBeDefined()
    expect(screen.getByText('75%')).toBeDefined()
  })

  it('renders accuracy chart when history available', () => {
    render(<TrainingCard {...base} />)
    expect(screen.getByTestId('line-chart')).toBeDefined()
  })

  it('renders caption history', () => {
    render(<TrainingCard {...base} />)
    expect(screen.getByText('a cat')).toBeDefined()
    expect(screen.getByText('a dog')).toBeDefined()
  })

  it('renders training status progress', () => {
    const status = { running: true, job_id: 'j1', total: 10, completed: 5, errors: 0, progress_pct: 50, current_caption: 'learning...', current_image: '', started_at: null, finished_at: null }
    render(<TrainingCard {...base} trainStatus={status} />)
    expect(screen.getByText(/5\/10/)).toBeDefined()
    const pctEls = screen.getAllByText('50%')
    expect(pctEls.length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText(/learning\.\.\./)).toBeDefined()
  })

  it('skips chart when accuracy_history has only 1 entry', () => {
    const { container } = render(<TrainingCard report={{ ...base.report, accuracy_history: [0.5], caption_history: [] }} />)
    expect(container.querySelector('[data-testid="line-chart"]')).toBeNull()
  })

  it('skips chart when accuracy_history is empty', () => {
    const { container } = render(<TrainingCard report={{ ...base.report, accuracy_history: [], caption_history: [] }} />)
    expect(container.querySelector('[data-testid="line-chart"]')).toBeNull()
  })

  it('renders without trainStatus', () => {
    const { container } = render(<TrainingCard {...base} trainStatus={null} />)
    expect(screen.getByText('10')).toBeDefined()
    expect(container.querySelector('[class*="progress"]')).toBeNull()
  })

  it('hides progress bar when total is 0', () => {
    const status = { running: true, job_id: 'j1', total: 0, completed: 0, errors: 0, progress_pct: 0, current_caption: '', current_image: '', started_at: null, finished_at: null }
    const { container } = render(<TrainingCard {...base} trainStatus={status} />)
    expect(container.querySelector('[role="progressbar"]')).toBeNull()
  })

  it('slices caption_history to last 6 in reverse order', () => {
    const captions = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8']
    render(<TrainingCard report={{ ...base.report, caption_history: captions }} />)
    expect(screen.getByText('c8')).toBeDefined()
    expect(screen.getByText('c3')).toBeDefined()
    expect(screen.queryByText('c1')).toBeNull()
  })

  it('renders diversity as percentage', () => {
    render(<TrainingCard {...base} />)
    expect(screen.getByText('75%')).toBeDefined()
  })
})
