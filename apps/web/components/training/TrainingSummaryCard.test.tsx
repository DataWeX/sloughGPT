// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import type { Checkpoint } from '@/lib/souls-controller'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children }: any) => <div>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
}))

import { TrainingSummaryCard } from './TrainingSummaryCard'

const cp = (overrides: Partial<Checkpoint>) => ({ name: 'cp', soul: 'assistant', ...overrides })

describe('TrainingSummaryCard', () => {
  afterEach(cleanup)

  it('shows no-data state for empty checkpoints', () => {
    const { container } = render(<TrainingSummaryCard checkpoints={[]} />)
    expect(container.innerHTML).toBe('')
  })

  it('shows total checkpoints count', () => {
    render(<TrainingSummaryCard checkpoints={[
      cp({ name: 'cp1', loss: 0.5, model_type: 'gpt2' }),
      cp({ name: 'cp2', loss: 0.3, model_type: 'gpt2' }),
    ]} />)
    expect(screen.getByText('Total checkpoints')).toBeDefined()
    expect(screen.getByText('2')).toBeDefined()
  })

  it('shows best loss', () => {
    render(<TrainingSummaryCard checkpoints={[
      cp({ name: 'cp1', loss: 0.5 }),
      cp({ name: 'cp2', loss: 0.3 }),
    ]} />)
    expect(screen.getByText('Best loss')).toBeDefined()
    expect(screen.getByText('0.3000')).toBeDefined()
  })

  it('shows avg loss', () => {
    render(<TrainingSummaryCard checkpoints={[
      cp({ name: 'cp1', loss: 0.6 }),
      cp({ name: 'cp2', loss: 0.4 }),
    ]} />)
    expect(screen.getByText('Avg loss')).toBeDefined()
    expect(screen.getByText('0.5000')).toBeDefined()
  })

  it('shows loss spread when multiple checkpoints', () => {
    render(<TrainingSummaryCard checkpoints={[
      cp({ name: 'cp1', loss: 0.2 }),
      cp({ name: 'cp2', loss: 0.8 }),
    ]} />)
    expect(screen.getByText('Loss spread')).toBeDefined()
    expect(screen.getByText('0.6000')).toBeDefined()
  })

  it('shows total training time', () => {
    render(<TrainingSummaryCard checkpoints={[
      cp({ name: 'cp1', training_duration_s: 120 }),
      cp({ name: 'cp2', training_duration_s: 60 }),
    ]} />)
    expect(screen.getByText('Total training time')).toBeDefined()
    expect(screen.getByText('3m 0s')).toBeDefined()
  })

  it('shows fastest run', () => {
    render(<TrainingSummaryCard checkpoints={[
      cp({ name: 'cp1', training_duration_s: 120 }),
      cp({ name: 'cp2', training_duration_s: 30 }),
    ]} />)
    expect(screen.getByText('Fastest run')).toBeDefined()
    expect(screen.getByText('30s')).toBeDefined()
  })

  it('shows max vocab size', () => {
    render(<TrainingSummaryCard checkpoints={[
      cp({ name: 'cp1', vocab_size: 256 }),
      cp({ name: 'cp2', vocab_size: 512 }),
    ]} />)
    expect(screen.getByText('Max vocab size')).toBeDefined()
    expect(screen.getByText('512')).toBeDefined()
  })

  it('shows top model type when multiple types', () => {
    render(<TrainingSummaryCard checkpoints={[
      cp({ name: 'cp1', model_type: 'gpt2' }),
      cp({ name: 'cp2', model_type: 'gpt2' }),
      cp({ name: 'cp3', model_type: 'qwen' }),
    ]} />)
    expect(screen.getByText('Top model type')).toBeDefined()
    expect(screen.getByText('gpt2 (2)')).toBeDefined()
  })

  it('formats duration in hours', () => {
    render(<TrainingSummaryCard checkpoints={[
      cp({ name: 'cp1', training_duration_s: 3660 }),
    ]} />)
    expect(screen.getAllByText('1h 1m').length).toBeGreaterThanOrEqual(1)
  })
})
