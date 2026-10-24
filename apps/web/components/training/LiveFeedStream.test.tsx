// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, act } from '@testing-library/react'
import React from 'react'
import { LiveFeedStream } from './LiveFeedStream'
import type { SSEEnvelope, SSEStreamOptions } from '@/lib/sse-client'

vi.mock('@sloughgpt/strui', () => ({
  Badge: ({ children, className }: any) => <span className={className}>{children}</span>,
  StatusDot: ({ tone, pulse }: any) => (
    <span data-testid="status-dot" data-tone={tone} data-pulse={pulse ? 'yes' : 'no'} />
  ),
}))

let capturedOptions: SSEStreamOptions | null = null
let startMock: ReturnType<typeof vi.fn>
let stopMock: ReturnType<typeof vi.fn>

vi.mock('@/lib/sse-client', () => ({
  createSSEStream: (options: SSEStreamOptions) => {
    capturedOptions = options
    startMock = vi.fn()
    stopMock = vi.fn()
    return { start: startMock, stop: stopMock, connected: false }
  },
}))

function frame(data: Record<string, unknown>): SSEEnvelope {
  return {
    stream: 'training-live',
    phase: 'LIVE',
    status: 'working',
    data,
    meta: {},
    message: '',
  }
}

function snapshot(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    sources: [
      {
        name: 'api-conversations',
        spec: 'feed:api-conversations',
        total: 12,
        delta: 0,
        last_updated: null,
      },
    ],
    totals: { 'api-conversations': 12 },
    jobs: { active: [], count: 0 },
    at: new Date().toISOString(),
    changed: false,
    ...overrides,
  }
}

function emit(data: Record<string, unknown>): void {
  act(() => {
    capturedOptions?.onEvent(frame(data))
  })
}

function openStream(): void {
  act(() => {
    capturedOptions?.onOpen?.()
  })
}

beforeEach(() => {
  capturedOptions = null
  startMock = undefined as unknown as ReturnType<typeof vi.fn>
  stopMock = undefined as unknown as ReturnType<typeof vi.fn>
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('LiveFeedStream', () => {
  it('starts the live stream on mount and stops it on unmount', () => {
    const { unmount } = render(<LiveFeedStream />)
    expect(capturedOptions?.url).toBe('/training/live-stream')
    expect(startMock).toHaveBeenCalledTimes(1)
    unmount()
    expect(stopMock).toHaveBeenCalledTimes(1)
  })

  it('shows the empty state before any frame arrives', () => {
    render(<LiveFeedStream />)
    expect(screen.getByText(/No activity yet/)).toBeInTheDocument()
    expect(screen.getByText(/Loading feed totals/)).toBeInTheDocument()
  })

  it('renders feed totals with the trainable spec from the baseline frame', () => {
    render(<LiveFeedStream />)
    openStream()
    emit(snapshot())
    expect(screen.getByText('feed:api-conversations')).toBeInTheDocument()
    expect(screen.getByText('12')).toBeInTheDocument()
    expect(screen.queryByText(/Loading feed totals/)).not.toBeInTheDocument()
  })

  it('appends an arrival entry when a feed total grows', () => {
    render(<LiveFeedStream />)
    openStream()
    emit(snapshot())
    emit(
      snapshot({
        sources: [
          {
            name: 'api-conversations',
            spec: 'feed:api-conversations',
            total: 14,
            delta: 2,
            last_updated: null,
          },
        ],
        totals: { 'api-conversations': 14 },
        changed: true,
      }),
    )
    expect(
      screen.getByText('feed:api-conversations received 2 new records'),
    ).toBeInTheDocument()
    expect(screen.getByText('+2')).toBeInTheDocument()
  })

  it('does not append log entries on unchanged frames', () => {
    render(<LiveFeedStream />)
    openStream()
    emit(snapshot())
    emit(snapshot())
    expect(screen.queryByText(/received/)).not.toBeInTheDocument()
    expect(screen.getAllByRole('listitem').length).toBeLessThanOrEqual(2) // totals li only
  })

  it('shows active jobs and their transitions', () => {
    render(<LiveFeedStream />)
    openStream()
    emit(snapshot())
    emit(
      snapshot({
        jobs: {
          active: [
            { id: 'j1', name: 'Live job', status: 'running', dataset: 'feed:feedback', progress: 40 },
          ],
          count: 1,
        },
        changed: true,
      }),
    )
    expect(screen.getByText('Live job')).toBeInTheDocument()
    expect(screen.getByText(/running 40%/)).toBeInTheDocument()
    expect(screen.getByText(/Live job started on feed:feedback/)).toBeInTheDocument()
    expect(screen.getByText('1 job running')).toBeInTheDocument()
  })

  it('ignores envelopes from other streams', () => {
    render(<LiveFeedStream />)
    act(() => {
      capturedOptions?.onEvent({
        stream: 'health',
        phase: 'HEALTH',
        status: 'working',
        data: { bogus: true },
        meta: {},
        message: '',
      })
    })
    expect(screen.queryByText(/Loading feed totals/)).toBeInTheDocument()
  })

  it('reflects connection state on the status dot', () => {
    render(<LiveFeedStream />)
    expect(screen.getByTestId('status-dot').dataset.tone).toBe('muted')
    openStream()
    expect(screen.getByTestId('status-dot').dataset.tone).toBe('primary')
    expect(screen.getByText('Live — following feeds')).toBeInTheDocument()
    act(() => {
      capturedOptions?.onError?.(new Error('Failed to fetch'))
    })
    expect(screen.getByTestId('status-dot').dataset.tone).toBe('warning')
    expect(screen.getByText('Reconnecting…')).toBeInTheDocument()
  })
})
