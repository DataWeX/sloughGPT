import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'

vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
  apiPut: vi.fn(),
  apiDelete: vi.fn(),
}))
vi.mock('@/hooks/useLiveStatus', () => ({
  useLiveStatus: () => ({ health: null, connectionStatus: 'connected', connected: true }),
}))
vi.mock('@/lib/dev-log', () => ({
  logger: { debug: vi.fn(), info: vi.fn(), warning: vi.fn(), error: vi.fn() },
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (m: string) => void }) => unknown) =>
    selector({ addToast: vi.fn() }),
}))

import CalendarPage from './page'
import { apiGet } from '@/lib/http-client'

const get = vi.mocked(apiGet)

function makeEvent(overrides: Partial<Record<string, string>> = {}) {
  const today = new Date().toISOString().split('T')[0]
  return {
    id: 'event-1-abc123',
    title: 'Standup',
    description: 'daily sync',
    date: today,
    start_time: '09:00',
    end_time: '10:00',
    color: 'primary',
    created_at: '2026-10-04T10:00:00Z',
    updated_at: '2026-10-04T10:00:00Z',
    ...overrides,
  }
}

describe('CalendarPage', () => {
  afterEach(cleanup)

  beforeEach(() => {
    get.mockReset()
  })

  it('fetches events through http-client with an ISO date and renders them', async () => {
    get.mockResolvedValue({ events: [makeEvent()] } as never)
    render(<CalendarPage />)

    expect(await screen.findByText('Standup')).toBeInTheDocument()
    const today = new Date().toISOString().split('T')[0]
    // http-client unwraps the {status, data} envelope before this resolves —
    // a raw fetch() here would read data.events off the envelope and render
    // an empty day even when the API returns events.
    expect(get).toHaveBeenCalledWith(`/api/calendar/events?date=${today}`)
  })

  it('renders the empty day state when the API returns no events', async () => {
    get.mockResolvedValue({ events: [] } as never)
    render(<CalendarPage />)

    expect(await screen.findByText('No events scheduled')).toBeInTheDocument()
  })
})
