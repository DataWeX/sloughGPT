import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  IconX: (props: any) => <span data-testid="icon-x" {...props} />,
  IconRefresh: (props: any) => <span data-testid="icon-refresh" {...props} />,
  IconCheck: (props: any) => <span data-testid="icon-check" {...props} />,
  IconDownload: (props: any) => <span data-testid="icon-download" {...props} />,
}))

const mockApiGet = vi.fn()

vi.mock('@/lib/http-client', () => ({
  apiGet: (...a: any[]) => mockApiGet(...a),
}))

import { TokenBilling } from './TokenBilling'

afterEach(cleanup)

beforeEach(() => {
  mockApiGet.mockReset()
})

describe('TokenBilling', () => {
  it('renders loading state initially', async () => {
    mockApiGet.mockResolvedValueOnce({
      userId: 'u1',
      balance: 4500,
      tier: 'pro',
      dailyUsed: 350,
      dailyLimit: 10000,
      monthlyUsed: 15000,
      monthlyLimit: 300000,
    })
    mockApiGet.mockResolvedValueOnce({
      totalRequests: 0,
      totalTokens: 0,
      totalCost: 0,
      byModel: {},
      byDay: {},
    })
    mockApiGet.mockResolvedValueOnce({ records: [] })

    render(<TokenBilling />)
    await waitFor(() => {
      expect(screen.getByText('Token Billing')).toBeInTheDocument()
    })
  })

  it('displays balance after load', async () => {
    mockApiGet.mockResolvedValueOnce({
      userId: 'u1',
      balance: 4500,
      tier: 'pro',
      dailyUsed: 350,
      dailyLimit: 10000,
      monthlyUsed: 15000,
      monthlyLimit: 300000,
    })
    mockApiGet.mockResolvedValueOnce({
      totalRequests: 42,
      totalTokens: 125000,
      totalCost: 1.25,
      byModel: {},
      byDay: {},
    })
    mockApiGet.mockResolvedValueOnce({ records: [] })

    render(<TokenBilling />)
    expect(await screen.findByText('4.5K')).toBeInTheDocument()
  })

  it('shows tier badge', async () => {
    mockApiGet.mockResolvedValueOnce({
      userId: 'u1',
      balance: 4500,
      tier: 'pro',
      dailyUsed: 350,
      dailyLimit: 10000,
      monthlyUsed: 15000,
      monthlyLimit: 300000,
    })
    mockApiGet.mockResolvedValueOnce({
      totalRequests: 0,
      totalTokens: 0,
      totalCost: 0,
      byModel: {},
      byDay: {},
    })
    mockApiGet.mockResolvedValueOnce({ records: [] })

    render(<TokenBilling />)
    expect(await screen.findByText('Pro')).toBeInTheDocument()
  })

  it('switches to history tab', async () => {
    mockApiGet.mockResolvedValueOnce({
      userId: 'u1',
      balance: 4500,
      tier: 'free',
      dailyUsed: 350,
      dailyLimit: 500,
      monthlyUsed: 15000,
      monthlyLimit: 10000,
    })
    mockApiGet.mockResolvedValueOnce({
      totalRequests: 0,
      totalTokens: 0,
      totalCost: 0,
      byModel: {},
      byDay: {},
    })
    mockApiGet.mockResolvedValueOnce({ records: [] })

    render(<TokenBilling />)
    await waitFor(() => {
      fireEvent.click(screen.getByText('history'))
    })
    expect(screen.getByText('No usage yet')).toBeInTheDocument()
  })

  it('switches to pricing tab', async () => {
    mockApiGet.mockResolvedValueOnce({
      userId: 'u1',
      balance: 4500,
      tier: 'free',
      dailyUsed: 350,
      dailyLimit: 500,
      monthlyUsed: 15000,
      monthlyLimit: 10000,
    })
    mockApiGet.mockResolvedValueOnce({
      totalRequests: 0,
      totalTokens: 0,
      totalCost: 0,
      byModel: {},
      byDay: {},
    })
    mockApiGet.mockResolvedValueOnce({ records: [] })

    render(<TokenBilling />)
    await waitFor(() => {
      fireEvent.click(screen.getByText('pricing'))
    })
    expect(screen.getByText('free')).toBeInTheDocument()
    expect(screen.getByText('pro')).toBeInTheDocument()
    expect(screen.getByText('enterprise')).toBeInTheDocument()
  })

  it('shows current plan highlight', async () => {
    mockApiGet.mockResolvedValueOnce({
      userId: 'u1',
      balance: 4500,
      tier: 'pro',
      dailyUsed: 350,
      dailyLimit: 10000,
      monthlyUsed: 15000,
      monthlyLimit: 300000,
    })
    mockApiGet.mockResolvedValueOnce({
      totalRequests: 0,
      totalTokens: 0,
      totalCost: 0,
      byModel: {},
      byDay: {},
    })
    mockApiGet.mockResolvedValueOnce({ records: [] })

    render(<TokenBilling />)
    await waitFor(() => {
      fireEvent.click(screen.getByText('pricing'))
    })
    expect(screen.getByText('Current plan')).toBeInTheDocument()
  })
})