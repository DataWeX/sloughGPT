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
}))

import { AutoTrainHistoryCard } from './AutoTrainHistoryCard'

afterEach(() => cleanup())

describe('AutoTrainHistoryCard', () => {
  it('shows empty state', () => {
    render(<AutoTrainHistoryCard lastTrain={null} />)
    expect(screen.getByText('Last Training Run')).toBeTruthy()
    expect(screen.getByText('No training runs yet.')).toBeTruthy()
  })

  it('shows training run details', () => {
    render(<AutoTrainHistoryCard lastTrain={{
      started_at: '2026-09-09T10:00:00Z',
      completed_at: '2026-09-09T10:05:00Z',
      pairs_used: 25,
      checkpoint: 'lora-v3',
    }} />)
    expect(screen.getByText('Started')).toBeTruthy()
    expect(screen.getByText('Completed')).toBeTruthy()
    expect(screen.getByText('25')).toBeTruthy()
    expect(screen.getByText('lora-v3')).toBeTruthy()
  })

  it('shows in-progress state', () => {
    render(<AutoTrainHistoryCard lastTrain={{
      started_at: '2026-09-09T10:00:00Z',
      completed_at: null,
      pairs_used: 10,
      checkpoint: 'lora-v4',
    }} />)
    expect(screen.getByText('In progress...')).toBeTruthy()
  })

  it('shows duration', () => {
    render(<AutoTrainHistoryCard lastTrain={{
      started_at: '2026-09-09T10:00:00Z',
      completed_at: '2026-09-09T10:05:00Z',
      pairs_used: 15,
      checkpoint: 'lora-5',
    }} />)
    expect(screen.getByText(/Duration:/)).toBeTruthy()
  })
})
