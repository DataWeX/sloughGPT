import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  IconX: (props: any) => <span data-testid="icon-x" {...props} />,
  IconRefresh: (props: any) => <span data-testid="icon-refresh" {...props} />,
  IconCheck: (props: any) => <span data-testid="icon-check" {...props} />,
  IconDownload: (props: any) => <span data-testid="icon-download" {...props} />,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,
    Separator: () => <hr />,
    Tooltip: ({ children }: any) => <>{children}</>,
    TooltipTrigger: ({ children }: any) => <>{children}</>,
    TooltipContent: ({ children }: any) => <>{children}</>,
    Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
    Avatar: ({ children }: any) => <div>{children}</div>,
    AvatarFallback: ({ children }: any) => <div>{children}</div>,
    ScrollArea: ({ children }: any) => <div>{children}</div>,
    Table: ({ children }: any) => <table>{children}</table>,
    TableBody: ({ children }: any) => <tbody>{children}</tbody>,
    TableRow: ({ children }: any) => <tr>{children}</tr>,
    TableCell: ({ children }: any) => <td>{children}</td>,
    TableHead: ({ children }: any) => <th>{children}</th>,
    TableHeader: ({ children }: any) => <thead>{children}</thead>,
    Collapsible: ({ children }: any) => <div>{children}</div>,
    CollapsibleTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    CollapsibleContent: ({ children }: any) => <div>{children}</div>,
    Toggle: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Command: ({ children }: any) => <div>{children}</div>,
    CommandInput: ({ ...props }: any) => <input {...props} />,
    CommandList: ({ children }: any) => <div>{children}</div>,
    CommandEmpty: ({ children }: any) => <div>{children}</div>,
    CommandGroup: ({ children }: any) => <div>{children}</div>,
    CommandItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
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