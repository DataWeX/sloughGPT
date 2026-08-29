import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import { TokenBilling } from './TokenBilling'

afterEach(cleanup)

const mockFetch = vi.fn()

beforeEach(() => {
  global.fetch = mockFetch
  mockFetch.mockResolvedValue({
    ok: true,
    json: vi.fn().mockResolvedValue({
      userId: 'u1',
      balance: 4500,
      tier: 'pro',
      dailyUsed: 350,
      dailyLimit: 10000,
      monthlyUsed: 15000,
      monthlyLimit: 300000,
    }),
  })
})

describe('TokenBilling', () => {
  it('renders loading state initially', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue({
        userId: 'u1',
        balance: 4500,
        tier: 'pro',
        dailyUsed: 350,
        dailyLimit: 10000,
        monthlyUsed: 15000,
        monthlyLimit: 300000,
      }),
    })
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue({
        totalRequests: 0,
        totalTokens: 0,
        totalCost: 0,
        byModel: {},
        byDay: {},
      }),
    })
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue({ records: [] }),
    })

    render(<TokenBilling />)
    await waitFor(() => {
      expect(screen.getByText('Token Billing')).toBeInTheDocument()
    })
  })

  it('displays balance after load', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue({
        userId: 'u1',
        balance: 4500,
        tier: 'pro',
        dailyUsed: 350,
        dailyLimit: 10000,
        monthlyUsed: 15000,
        monthlyLimit: 300000,
      }),
    })
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue({
        totalRequests: 42,
        totalTokens: 125000,
        totalCost: 1.25,
        byModel: {},
        byDay: {},
      }),
    })
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue({ records: [] }),
    })

    render(<TokenBilling />)
    expect(await screen.findByText('4.5K')).toBeInTheDocument()
  })

  it('shows tier badge', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue({
        userId: 'u1',
        balance: 4500,
        tier: 'pro',
        dailyUsed: 350,
        dailyLimit: 10000,
        monthlyUsed: 15000,
        monthlyLimit: 300000,
      }),
    })
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue({
        totalRequests: 0,
        totalTokens: 0,
        totalCost: 0,
        byModel: {},
        byDay: {},
      }),
    })
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue({ records: [] }),
    })

    render(<TokenBilling />)
    expect(await screen.findByText('Pro')).toBeInTheDocument()
  })

  it('switches to history tab', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue({
        userId: 'u1',
        balance: 4500,
        tier: 'free',
        dailyUsed: 350,
        dailyLimit: 500,
        monthlyUsed: 15000,
        monthlyLimit: 10000,
      }),
    })
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue({
        totalRequests: 0,
        totalTokens: 0,
        totalCost: 0,
        byModel: {},
        byDay: {},
      }),
    })
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue({ records: [] }),
    })

    render(<TokenBilling />)
    await waitFor(() => {
      fireEvent.click(screen.getByText('history'))
    })
    expect(screen.getByText('No usage yet')).toBeInTheDocument()
  })

  it('switches to pricing tab', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue({
        userId: 'u1',
        balance: 4500,
        tier: 'free',
        dailyUsed: 350,
        dailyLimit: 500,
        monthlyUsed: 15000,
        monthlyLimit: 10000,
      }),
    })
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue({
        totalRequests: 0,
        totalTokens: 0,
        totalCost: 0,
        byModel: {},
        byDay: {},
      }),
    })
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue({ records: [] }),
    })

    render(<TokenBilling />)
    await waitFor(() => {
      fireEvent.click(screen.getByText('pricing'))
    })
    expect(screen.getByText('free')).toBeInTheDocument()
    expect(screen.getByText('pro')).toBeInTheDocument()
    expect(screen.getByText('enterprise')).toBeInTheDocument()
  })

  it('shows current plan highlight', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue({
        userId: 'u1',
        balance: 4500,
        tier: 'pro',
        dailyUsed: 350,
        dailyLimit: 10000,
        monthlyUsed: 15000,
        monthlyLimit: 300000,
      }),
    })
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue({
        totalRequests: 0,
        totalTokens: 0,
        totalCost: 0,
        byModel: {},
        byDay: {},
      }),
    })
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue({ records: [] }),
    })

    render(<TokenBilling />)
    await waitFor(() => {
      fireEvent.click(screen.getByText('pricing'))
    })
    expect(screen.getByText('Current plan')).toBeInTheDocument()
  })
})