import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import { act } from 'react'

// ── strui mock ──
vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => { const C = () => <span data-testid={`icon-${name}`}>{name}</span>; C.displayName = `Icon${name}`; return C }
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    Card: passthrough, CardContent: passthrough,
    CardHeader: ({ children, className }: any) => <div className={className}>{children}</div>,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    Button: ({ children, onClick, variant, size, className, disabled }: any) => (
      <button onClick={onClick} className={className} disabled={disabled} data-variant={variant}>{children}</button>
    ),
    IconRefresh: iconMock('refresh'), IconTrash: iconMock('trash'),
    StatCard: ({ label, value }: any) => <div data-testid="stat-card"><span>{label}</span>: <span>{String(value)}</span></div>,
    KpiGrid: ({ children }: any) => <div data-testid="kpi-grid">{children}</div>,
  }
})

// ── controller / toast / config / fetch mocks ──
const { mockList, mockGetQuality, mockAggregateBest, mockPrune, mockReset, mockAddToast, mockFetch, mockRunEval, mockGetHistory } = vi.hoisted(() => ({
  mockList: vi.fn(), mockGetQuality: vi.fn(), mockAggregateBest: vi.fn(),
  mockPrune: vi.fn(), mockReset: vi.fn(), mockAddToast: vi.fn(), mockFetch: vi.fn(),
  mockRunEval: vi.fn(), mockGetHistory: vi.fn(),
}))

vi.mock('@/lib/user-adapters-controller', () => ({
  userAdaptersController: {
    list: mockList, getQuality: mockGetQuality, aggregateBest: mockAggregateBest,
    prune: mockPrune, reset: mockReset, get: vi.fn(),
  },
}))

vi.mock('@/lib/lora-eval-controller', () => ({
  loraEvalController: { runEval: mockRunEval, getHistory: mockGetHistory },
}))

vi.mock('@/lib/toast-store', () => ({ useToastStore: (sel: any) => sel({ addToast: mockAddToast }) }))
vi.mock('@/lib/config', () => ({ PUBLIC_API_URL: 'http://test-api' }))

import AdaptersPage from './page'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

const mockStats = {
  total_users: 12,
  total_size_bytes: 5242880,
  total_size_mb: 5.0,
  adapter_rank: 8,
  model_dim: 768,
  avg_size_per_user_kb: 3.5,
}

const mockQuality = {
  count: 2,
  adapters: [
    { user_id: 'user-1', rank: 8, alpha: 8, model_dim: 768, created_at: '2026-06-01T00:00:00Z', updated_at: '2026-07-01T00:00:00Z', feedback_count: 5 },
    { user_id: 'user-2', rank: 8, alpha: 8, model_dim: 768, created_at: '2026-05-01T00:00:00Z', updated_at: '2026-06-15T00:00:00Z', feedback_count: 3 },
  ],
}

function mockHistoryFetch() {
  mockFetch.mockImplementation((url: string) => {
    if (url.includes('/lora-eval/run')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'started' }) })
    }
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ data: { results: [{ adapter_path: 'best_aggregated.npz', status: 'completed' }] } }),
    })
  })
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.stubGlobal('fetch', mockFetch)
  mockList.mockResolvedValue(mockStats)
  mockGetQuality.mockResolvedValue(mockQuality)
  mockRunEval.mockResolvedValue({ status: 'started' })
  mockGetHistory.mockResolvedValue([])
  mockHistoryFetch()
})

describe('AdaptersPage', () => {
  it('shows loading initially and calls list + getQuality', () => {
    mockList.mockReturnValue(new Promise(() => {}))
    render(<AdaptersPage />)
    expect(screen.getAllByText('Adapters').length).toBeGreaterThanOrEqual(1)
    expect(mockList).toHaveBeenCalledTimes(1)
    expect(mockGetQuality).toHaveBeenCalledWith(3)
    expect(screen.queryByText('Adapter Stats')).toBeNull()
  })

  it('displays adapter stats after loading', async () => {
    render(<AdaptersPage />)
    await waitFor(() => { expect(screen.getByText('Adapter Stats')).toBeTruthy() })
    expect(screen.getByText('12')).toBeTruthy()
    expect(screen.getByText('5.0 MB')).toBeTruthy()
    expect(screen.getByText('3.5 KB')).toBeTruthy()
  })

  it('lists adapters with feedback and rank', async () => {
    render(<AdaptersPage />)
    await waitFor(() => { expect(screen.getAllByText('user-1').length).toBeGreaterThan(0) })
    expect(screen.getAllByText('user-2').length).toBeGreaterThan(0)
    expect(screen.getAllByText('5 feedback').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('rank 8').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Adapters (2)')).toBeTruthy()
  })

  it('renders the adapter health card when adapters exist', async () => {
    render(<AdaptersPage />)
    await waitFor(() => { expect(screen.getByText('Adapter Health')).toBeTruthy() })
    expect(screen.getByText('Total Feedback')).toBeTruthy()
    expect(screen.getByText('Rank 8 (2)')).toBeTruthy()
    expect(screen.getByText('5 fb')).toBeTruthy()
  })

  it('shows empty state when no adapters exist', async () => {
    mockGetQuality.mockResolvedValue({ count: 0, adapters: [] })
    render(<AdaptersPage />)
    await waitFor(() => { expect(screen.getByText(/No adapters yet/)).toBeTruthy() })
    expect(screen.queryByText('Adapter Health')).toBeNull()
  })

  it('shows error state with retry on fetch failure', async () => {
    mockList.mockRejectedValueOnce(new Error('backend down'))
    render(<AdaptersPage />)
    await waitFor(() => { expect(screen.getByText('backend down')).toBeTruthy() })
    expect(screen.getByText('Retry')).toBeTruthy()
  })

  it('recovers after retry', async () => {
    mockList.mockRejectedValueOnce(new Error('backend down'))
    render(<AdaptersPage />)
    await waitFor(() => { expect(screen.getByText('backend down')).toBeTruthy() })
    await act(async () => { screen.getByText('Retry').click() })
    await waitFor(() => { expect(screen.getByText('Adapter Stats')).toBeTruthy() })
  })

  it('aggregates best adapters and shows verdict', async () => {
    mockAggregateBest.mockResolvedValue({ status: 'ok', user_count: 3, eval: { verdict: 'better' } })
    render(<AdaptersPage />)
    await waitFor(() => { expect(screen.getByText('Aggregate Best')).toBeTruthy() })
    await act(async () => { screen.getByText('Aggregate Best').click() })
    await waitFor(() => { expect(screen.getByText('Aggregated 3 adapters. Verdict: better')).toBeTruthy() })
  })

  it('shows aggregate failure message', async () => {
    mockAggregateBest.mockRejectedValue(new Error('no adapters to aggregate'))
    render(<AdaptersPage />)
    await waitFor(() => { expect(screen.getByText('Aggregate Best')).toBeTruthy() })
    await act(async () => { screen.getByText('Aggregate Best').click() })
    await waitFor(() => { expect(screen.getByText('no adapters to aggregate')).toBeTruthy() })
  })

  it('dismisses the aggregate result message', async () => {
    mockAggregateBest.mockResolvedValue({ status: 'ok', user_count: 3, eval: { verdict: 'better' } })
    render(<AdaptersPage />)
    await waitFor(() => { expect(screen.getByText('Aggregate Best')).toBeTruthy() })
    await act(async () => { screen.getByText('Aggregate Best').click() })
    await waitFor(() => { expect(screen.getByText('Aggregated 3 adapters. Verdict: better')).toBeTruthy() })
    await act(async () => { screen.getByText('Dismiss').click() })
    expect(screen.queryByText('Aggregated 3 adapters. Verdict: better')).toBeNull()
  })

  it('prunes old adapters and refetches', async () => {
    mockPrune.mockResolvedValue({ status: 'ok', deleted_count: 2, deleted_users: ['user-1', 'user-2'] })
    render(<AdaptersPage />)
    await waitFor(() => { expect(screen.getByText('Prune Old')).toBeTruthy() })
    expect(mockList).toHaveBeenCalledTimes(1)
    await act(async () => { screen.getByText('Prune Old').click() })
    await waitFor(() => { expect(mockPrune).toHaveBeenCalled() })
    expect(screen.getByText('Pruned 2 adapters')).toBeTruthy()
    await waitFor(() => { expect(mockList).toHaveBeenCalledTimes(2) })
  })

  it('shows prune failure message', async () => {
    mockPrune.mockRejectedValue(new Error('prune failed'))
    render(<AdaptersPage />)
    await waitFor(() => { expect(screen.getByText('Prune Old')).toBeTruthy() })
    await act(async () => { screen.getByText('Prune Old').click() })
    await waitFor(() => { expect(screen.getByText('prune failed')).toBeTruthy() })
  })

  it('resets an adapter and refetches', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    mockReset.mockResolvedValue({ status: 'ok', user_id: 'user-1', feedback_count: 0 })
    const { container } = render(<AdaptersPage />)
    await waitFor(() => { expect(screen.getAllByText('user-1').length).toBeGreaterThan(0) })
    const delBtns = container.querySelectorAll('button.text-destructive')
    await act(async () => { (delBtns[0] as HTMLElement).click() })
    await waitFor(() => { expect(mockReset).toHaveBeenCalledWith('user-1') })
    vi.mocked(window.confirm).mockRestore()
  })

  it('shows error toast when reset fails', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    mockReset.mockRejectedValue(new Error('boom'))
    const { container } = render(<AdaptersPage />)
    await waitFor(() => { expect(screen.getAllByText('user-1').length).toBeGreaterThan(0) })
    const delBtns = container.querySelectorAll('button.text-destructive')
    await act(async () => { (delBtns[0] as HTMLElement).click() })
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Failed to reset adapter', 'error') })
    vi.mocked(window.confirm).mockRestore()
  })

  it('runs LoRA eval and shows eval history', async () => {
    mockGetHistory.mockResolvedValue([{ adapter_path: 'best_aggregated.npz', verdict: 'better' }])
    render(<AdaptersPage />)
    await waitFor(() => { expect(screen.getByText('Run LoRA Eval')).toBeTruthy() })
    await act(async () => { screen.getByText('Run LoRA Eval').click() })
    await waitFor(() => {
      expect(mockRunEval).toHaveBeenCalledWith('data/user_adapters/best_aggregated.npz')
    })
    expect(mockAddToast).toHaveBeenCalledWith('Evaluation complete', 'success')
    await waitFor(() => { expect(screen.getAllByText('better').length).toBeGreaterThan(0) })
  })

  it('shows error toast when eval fails', async () => {
    mockRunEval.mockRejectedValueOnce(new Error('network'))
    render(<AdaptersPage />)
    await waitFor(() => { expect(screen.getByText('Run LoRA Eval')).toBeTruthy() })
    await act(async () => { screen.getByText('Run LoRA Eval').click() })
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Eval failed', 'error') })
  })

  it('loads eval history on refresh', async () => {
    mockGetHistory.mockResolvedValue([{ adapter_path: 'best_aggregated.npz', verdict: 'better' }])
    render(<AdaptersPage />)
    await waitFor(() => { expect(screen.getByText('Adapter Stats')).toBeTruthy() })
    expect(screen.queryByText('best_aggregated.npz')).toBeNull()
    await act(async () => { screen.getAllByTestId('icon-refresh')[0].click() })
    await waitFor(() => { expect(screen.getByText('best_aggregated.npz')).toBeTruthy() })
  })
})
