import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

const mockPush = vi.fn()
vi.mock('next/navigation', () => ({ useRouter: () => ({ push: mockPush }) }))

import { TrainingHistory } from './TrainingHistory'

const jobs = [
  { id: 'job-1', name: 'distill-a', status: 'completed', loss: 1.5, epochs_completed: 3 },
  { id: 'job-2', name: 'distill-b', status: 'running', loss: 2.25, epochs_completed: 1 },
  { id: 'job-3', name: '', status: 'failed' },
] as any[]

describe('TrainingHistory', () => {
  afterEach(cleanup)

  it('shows empty state when there are no jobs', () => {
    render(<TrainingHistory jobs={[]} />)
    expect(screen.getByText('No training jobs yet')).toBeDefined()
  })

  it('renders job name or id', () => {
    render(<TrainingHistory jobs={jobs} />)
    expect(screen.getByText('distill-a')).toBeDefined()
    expect(screen.getByText('job-3')).toBeDefined()
  })

  it('renders status badges', () => {
    render(<TrainingHistory jobs={jobs} />)
    expect(screen.getByText('completed')).toBeDefined()
    expect(screen.getByText('running')).toBeDefined()
    expect(screen.getByText('failed')).toBeDefined()
  })

  it('renders loss with three decimals', () => {
    render(<TrainingHistory jobs={jobs} />)
    expect(screen.getByText('1.500')).toBeDefined()
    expect(screen.getByText('2.250')).toBeDefined()
  })

  it('renders completed epochs', () => {
    render(<TrainingHistory jobs={jobs} />)
    expect(screen.getByText('ep3')).toBeDefined()
  })

  it('does not render loss or epochs when absent', () => {
    render(<TrainingHistory jobs={[jobs[2]]} />)
    expect(screen.queryByText(/^\d+\.\d{3}$/)).toBeNull()
    expect(screen.queryByText(/^ep/)).toBeNull()
  })

  it('shows only first 6 jobs and a +N more line', () => {
    const many = Array.from({ length: 8 }, (_, i) => ({ id: `job-${i}`, name: `name-${i}`, status: 'queued' })) as any[]
    render(<TrainingHistory jobs={many} />)
    expect(screen.getByText('name-5')).toBeDefined()
    expect(screen.queryByText('name-6')).toBeNull()
    expect(screen.getByText('+2 more')).toBeDefined()
  })
})