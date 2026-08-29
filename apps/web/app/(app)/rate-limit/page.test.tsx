import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

const {
  mockGetStatus, mockCheck, mockAddToast,
} = vi.hoisted(() => ({
  mockGetStatus: vi.fn(), mockCheck: vi.fn(), mockAddToast: vi.fn(),
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...a: any[]) => a.join(' ')),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Button: ({ children, onClick, disabled, size, variant }: any) => (
      <button onClick={onClick} disabled={disabled} data-size={size} data-variant={variant}>{children}</button>
    ),
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    StatCard: ({ label, value, icon }: any) => (
      <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span>{icon}</div>
    ),
  }
})

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/lib/rate-limit-controller', () => ({
  rateLimitController: {
    getStatus: (...a: unknown[]) => mockGetStatus(...a),
    check: (...a: unknown[]) => mockCheck(...a),
  },
}))

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title, subtitle, headerRight }: any) => (
    <div data-testid="page-container"><h1>{title}</h1><p>{subtitle}</p><div>{headerRight}</div>{children}</div>
  ),
}))

import RateLimitPage from './page'

afterEach(cleanup)

beforeEach(() => {
  vi.clearAllMocks()
  mockGetStatus.mockResolvedValue({ enabled: true, requests_per_minute: 60, burst_size: 10 })
})

describe('RateLimitPage', () => {
  it('renders page header', async () => {
    render(<RateLimitPage />)
    expect(screen.getByText('Rate Limiting')).toBeTruthy()
    expect(screen.getByText('Rate limit configuration and status')).toBeTruthy()
  })

  it('fetches status on mount', async () => {
    render(<RateLimitPage />)
    await waitFor(() => {
      expect(mockGetStatus).toHaveBeenCalledTimes(1)
    })
  })

  it('shows active status when enabled', async () => {
    render(<RateLimitPage />)
    await waitFor(() => {
      const stat = screen.getByTestId('stat-Status')
      expect(stat.textContent).toContain('Active')
    })
  })

  it('shows inactive status when disabled', async () => {
    mockGetStatus.mockResolvedValue({ enabled: false, requests_per_minute: 0, burst_size: 0 })
    render(<RateLimitPage />)
    await waitFor(() => {
      const stat = screen.getByTestId('stat-Status')
      expect(stat.textContent).toContain('Inactive')
    })
  })

  it('displays rate limit config values', async () => {
    render(<RateLimitPage />)
    await waitFor(() => {
      const rpm = screen.getByTestId('stat-Requests/min')
      expect(rpm.textContent).toContain('60')
      const burst = screen.getByTestId('stat-Burst Size')
      expect(burst.textContent).toContain('10')
    })
  })

  it('displays JSON configuration', async () => {
    render(<RateLimitPage />)
    await waitFor(() => {
      expect(screen.getByText(/"enabled": true/)).toBeTruthy()
      expect(screen.getByText(/"requests_per_minute": 60/)).toBeTruthy()
    })
  })

  it('check now calls controller', async () => {
    mockCheck.mockResolvedValue({ allowed: true, wait_time: 0 })
    render(<RateLimitPage />)
    await waitFor(() => { expect(screen.getByText('Check Now')).toBeTruthy() })

    await act(async () => { fireEvent.click(screen.getByText('Check Now')) })
    await waitFor(() => {
      expect(mockCheck).toHaveBeenCalledTimes(1)
    })
  })

  it('shows check result - allowed', async () => {
    mockCheck.mockResolvedValue({ allowed: true, wait_time: 0 })
    render(<RateLimitPage />)
    await waitFor(() => { expect(screen.getByText('Check Now')).toBeTruthy() })

    await act(async () => { fireEvent.click(screen.getByText('Check Now')) })
    await waitFor(() => {
      expect(screen.getByText('Allowed: Yes')).toBeTruthy()
    })
  })

  it('shows check result - denied with wait time', async () => {
    mockCheck.mockResolvedValue({ allowed: false, wait_time: 3.5 })
    render(<RateLimitPage />)
    await waitFor(() => { expect(screen.getByText('Check Now')).toBeTruthy() })

    await act(async () => { fireEvent.click(screen.getByText('Check Now')) })
    await waitFor(() => {
      expect(screen.getByText('Allowed: No')).toBeTruthy()
      expect(screen.getByText('Wait: 3.5s')).toBeTruthy()
    })
  })

  it('check error shows toast', async () => {
    mockCheck.mockRejectedValue(new Error('network'))
    render(<RateLimitPage />)
    await waitFor(() => { expect(screen.getByText('Check Now')).toBeTruthy() })

    await act(async () => { fireEvent.click(screen.getByText('Check Now')) })
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Could not check rate limit', 'error')
    })
  })

  it('status fetch error shows toast', async () => {
    mockGetStatus.mockRejectedValue(new Error('network'))
    render(<RateLimitPage />)
    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Could not load rate limit status', 'error')
    })
  })

  it('refresh button calls fetchStatus', async () => {
    render(<RateLimitPage />)
    await waitFor(() => { expect(mockGetStatus).toHaveBeenCalledTimes(1) })

    const refreshBtn = screen.getAllByRole('button').find(b => b.textContent === 'Refresh')
    await act(async () => { fireEvent.click(refreshBtn!) })
    await waitFor(() => { expect(mockGetStatus).toHaveBeenCalledTimes(2) })
  })

  it('shows Checking... while check is in progress', async () => {
    let resolveCheck: (v: any) => void
    mockCheck.mockReturnValue(new Promise(r => { resolveCheck = r }))

    render(<RateLimitPage />)
    await waitFor(() => { expect(screen.getByText('Check Now')).toBeTruthy() })

    await act(async () => { fireEvent.click(screen.getByText('Check Now')) })
    expect(screen.getByText('Checking...')).toBeTruthy()

    await act(async () => { resolveCheck!({ allowed: true, wait_time: 0 }) })
  })
})
