// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
  Skeleton: (props: any) => <div data-testid="skeleton" {...props} />,
}))

import { CloudTrainingJobList } from './CloudTrainingJobList'
import type { CloudJob } from './CloudTrainingJobList'

afterEach(() => cleanup())

const mockJobs: CloudJob[] = [
  { job_id: 'job-1', provider: 'aws', status: 'completed', progress: 100 },
  { job_id: 'job-2', provider: 'gcp', status: 'running', progress: 50 },
  { job_id: 'job-3', provider: 'local', status: 'failed', progress: 20, error: 'OOM' },
]

describe('CloudTrainingJobList', () => {
  it('renders the card title', () => {
    render(<CloudTrainingJobList />)
    expect(screen.getAllByText('Training Jobs').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no jobs', () => {
    render(<CloudTrainingJobList />)
    expect(screen.getByText('No training jobs yet.')).toBeTruthy()
  })

  it('shows skeleton when loading', () => {
    render(<CloudTrainingJobList loading />)
    expect(screen.getByTestId('skeleton')).toBeTruthy()
  })

  it('renders jobs when provided', () => {
    render(<CloudTrainingJobList jobs={mockJobs} />)
    expect(screen.getByText('job-1')).toBeTruthy()
    expect(screen.getByText('job-2')).toBeTruthy()
    expect(screen.getByText('job-3')).toBeTruthy()
  })

  it('renders correct number of badges', () => {
    render(<CloudTrainingJobList jobs={mockJobs} />)
    const badges = screen.getAllByText(/completed|running|failed/)
    expect(badges.length).toBe(3)
  })

  it('renders correct status text in badges', () => {
    render(<CloudTrainingJobList jobs={mockJobs} />)
    expect(screen.getByText('completed')).toBeTruthy()
    expect(screen.getByText('running')).toBeTruthy()
    expect(screen.getByText('failed')).toBeTruthy()
  })

  it('renders provider names', () => {
    render(<CloudTrainingJobList jobs={mockJobs} />)
    expect(screen.getByText('aws')).toBeTruthy()
    expect(screen.getByText('gcp')).toBeTruthy()
    expect(screen.getByText('local')).toBeTruthy()
  })
})
