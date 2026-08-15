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
    Input: ({ value, onChange, placeholder, onKeyDown }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} onKeyDown={onKeyDown} />
    ),
    IconRefresh: iconMock('refresh'), IconTrash: iconMock('trash'),
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    StatCard: ({ label, value }: any) => <div>{label}: {value}</div>,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
  }
})

// ── controller & toast mocks ──
const { mockList, mockCreate, mockDelete, mockLogMetric, mockLogParam, mockComplete, mockAddToast } = vi.hoisted(() => ({
  mockList: vi.fn(), mockCreate: vi.fn(), mockDelete: vi.fn(),
  mockLogMetric: vi.fn(), mockLogParam: vi.fn(), mockComplete: vi.fn(), mockAddToast: vi.fn(),
}))

vi.mock('@/lib/experiments-controller', () => ({
  experimentsController: {
    list: mockList, create: mockCreate, delete: mockDelete,
    logMetric: mockLogMetric, logParam: mockLogParam, complete: mockComplete,
    getExperimentData: vi.fn().mockResolvedValue({ id: 'x', metrics: [], params: [], status: null }),
  },
}))

vi.mock('@/lib/toast-store', () => ({ useToastStore: (sel: any) => sel({ addToast: mockAddToast }) }))
vi.mock('@/components/experiments/ExperimentDetailsCard', () => ({
  ExperimentDetailsCard: ({ experimentId }: any) => (
    <div data-testid="experiment-details" data-id={experimentId}>Experiment Data</div>
  ),
}))

import ExperimentsPage from './page'

afterEach(() => { cleanup() })
beforeEach(() => {
  vi.clearAllMocks()
  mockList.mockResolvedValue([])
})

const mockExperiment = { id: 'distill-run-3', status: 'running', created_at: '2026-01-01T00:00:00Z' }

describe('ExperimentsPage', () => {
  it('shows loading initially and calls list', () => {
    mockList.mockReturnValue(new Promise(() => {}))
    render(<ExperimentsPage />)
    expect(screen.getAllByText('Experiments').length).toBeGreaterThanOrEqual(1)
    expect(mockList).toHaveBeenCalledTimes(1)
    expect(screen.queryByText('New Experiment')).toBeNull()
  })

  it('displays experiments after loading', async () => {
    mockList.mockResolvedValue([mockExperiment])
    render(<ExperimentsPage />)
    await waitFor(() => { expect(screen.getByText('distill-run-3')).toBeTruthy() })
  })

  it('shows empty state when no experiments exist', async () => {
    mockList.mockResolvedValue([])
    render(<ExperimentsPage />)
    await waitFor(() => { expect(screen.getByText('No experiments yet.')).toBeTruthy() })
  })

  it('shows experiment count in the subtitle', async () => {
    mockList.mockResolvedValue([mockExperiment, { id: 'exp-2' }])
    render(<ExperimentsPage />)
    await waitFor(() => { expect(screen.getByText('2 experiments')).toBeTruthy() })
  })

  it('disables create button when name is empty', async () => {
    mockList.mockResolvedValue([])
    render(<ExperimentsPage />)
    await waitFor(() => { expect(screen.getByText('Create')).toBeTruthy() })
    expect((screen.getByText('Create') as HTMLButtonElement).disabled).toBe(true)
  })

  it('creates an experiment on submit', async () => {
    mockCreate.mockResolvedValue({ id: 'new-exp', name: 'My Exp', created: true })
    mockList.mockResolvedValue([])
    render(<ExperimentsPage />)
    await waitFor(() => { expect(screen.getByText('Create')).toBeTruthy() })
    await act(async () => { fireEvent.change(screen.getByPlaceholderText('Experiment name'), { target: { value: 'My Exp' } }) })
    await act(async () => { screen.getByText('Create').click() })
    await waitFor(() => { expect(mockCreate).toHaveBeenCalledWith('My Exp') })
  })

  it('creates an experiment on Enter key', async () => {
    mockCreate.mockResolvedValue({ id: 'new-exp', name: 'Key Exp', created: true })
    mockList.mockResolvedValue([])
    render(<ExperimentsPage />)
    await waitFor(() => { expect(screen.getByPlaceholderText('Experiment name')).toBeTruthy() })
    await act(async () => { fireEvent.change(screen.getByPlaceholderText('Experiment name'), { target: { value: 'Key Exp' } }) })
    await act(async () => { fireEvent.keyDown(screen.getByPlaceholderText('Experiment name'), { key: 'Enter' }) })
    await waitFor(() => { expect(mockCreate).toHaveBeenCalledWith('Key Exp') })
  })

  it('shows error toast when create fails', async () => {
    mockCreate.mockRejectedValue(new Error('boom'))
    mockList.mockResolvedValue([])
    render(<ExperimentsPage />)
    await waitFor(() => { expect(screen.getByText('Create')).toBeTruthy() })
    await act(async () => { fireEvent.change(screen.getByPlaceholderText('Experiment name'), { target: { value: 'X' } }) })
    await act(async () => { screen.getByText('Create').click() })
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Failed to create experiment', 'error') })
  })

  it('deletes an experiment', async () => {
    mockDelete.mockResolvedValue({ id: 'distill-run-3', deleted: true })
    mockList.mockResolvedValue([mockExperiment])
    const { container } = render(<ExperimentsPage />)
    await waitFor(() => { expect(screen.getByText('distill-run-3')).toBeTruthy() })
    const delBtn = container.querySelector('button.text-destructive') as HTMLElement
    await act(async () => { delBtn.click() })
    await waitFor(() => { expect(mockDelete).toHaveBeenCalledWith('distill-run-3') })
  })

  it('shows error toast when delete fails', async () => {
    mockDelete.mockRejectedValue(new Error('boom'))
    mockList.mockResolvedValue([mockExperiment])
    const { container } = render(<ExperimentsPage />)
    await waitFor(() => { expect(screen.getByText('distill-run-3')).toBeTruthy() })
    const delBtn = container.querySelector('button.text-destructive') as HTMLElement
    await act(async () => { delBtn.click() })
    await waitFor(() => { expect(mockAddToast).toHaveBeenCalledWith('Failed to delete experiment', 'error') })
  })

  it('filters experiments by search query', async () => {
    mockList.mockResolvedValue([{ id: 'distill-run-3' }, { id: 'alpha-run' }])
    render(<ExperimentsPage />)
    await waitFor(() => { expect(screen.getByText('distill-run-3')).toBeTruthy() })
    await act(async () => { fireEvent.change(screen.getByPlaceholderText('Search...'), { target: { value: 'alpha' } }) })
    expect(screen.getByText('alpha-run')).toBeTruthy()
    expect(screen.queryByText('distill-run-3')).toBeNull()
  })

  it('selects an experiment to reveal logging card and details', async () => {
    mockList.mockResolvedValue([mockExperiment])
    render(<ExperimentsPage />)
    await waitFor(() => { expect(screen.getByText('distill-run-3')).toBeTruthy() })
    await act(async () => { screen.getByText('distill-run-3').click() })
    expect(screen.getByText('Log to: distill-run-3')).toBeTruthy()
    expect(screen.getByTestId('experiment-details').getAttribute('data-id')).toBe('distill-run-3')
  })

  it('deselects an experiment on second click', async () => {
    mockList.mockResolvedValue([mockExperiment])
    render(<ExperimentsPage />)
    await waitFor(() => { expect(screen.getByText('distill-run-3')).toBeTruthy() })
    await act(async () => { screen.getByText('distill-run-3').click() })
    expect(screen.getByText('Log to: distill-run-3')).toBeTruthy()
    await act(async () => { screen.getByText('distill-run-3').click() })
    expect(screen.queryByText('Log to: distill-run-3')).toBeNull()
  })

  it('logs a metric to the selected experiment', async () => {
    mockLogMetric.mockResolvedValue({ status: 'logged' })
    mockList.mockResolvedValue([mockExperiment])
    render(<ExperimentsPage />)
    await waitFor(() => { expect(screen.getByText('distill-run-3')).toBeTruthy() })
    await act(async () => { screen.getByText('distill-run-3').click() })
    await act(async () => { fireEvent.change(screen.getByPlaceholderText('Metric name'), { target: { value: 'loss' } }) })
    await act(async () => { fireEvent.change(screen.getAllByPlaceholderText('Value')[0], { target: { value: '0.25' } }) })
    await act(async () => { screen.getByText('Log Metric').click() })
    await waitFor(() => { expect(mockLogMetric).toHaveBeenCalledWith('distill-run-3', 'loss', 0.25) })
    expect(screen.getByText('Logged loss=0.25')).toBeTruthy()
  })

  it('logs a param to the selected experiment', async () => {
    mockLogParam.mockResolvedValue({ status: 'logged' })
    mockList.mockResolvedValue([mockExperiment])
    render(<ExperimentsPage />)
    await waitFor(() => { expect(screen.getByText('distill-run-3')).toBeTruthy() })
    await act(async () => { screen.getByText('distill-run-3').click() })
    await act(async () => { fireEvent.change(screen.getByPlaceholderText('Param name'), { target: { value: 'lr' } }) })
    await act(async () => { fireEvent.change(screen.getAllByPlaceholderText('Value')[1], { target: { value: '1e-4' } }) })
    await act(async () => { screen.getByText('Log Param').click() })
    await waitFor(() => { expect(mockLogParam).toHaveBeenCalledWith('distill-run-3', 'lr', '1e-4') })
    expect(screen.getByText('Logged lr=1e-4')).toBeTruthy()
  })

  it('marks an experiment complete and shows status message', async () => {
    mockComplete.mockResolvedValue({ status: 'completed' })
    mockList.mockResolvedValue([mockExperiment])
    render(<ExperimentsPage />)
    await waitFor(() => { expect(screen.getByText('distill-run-3')).toBeTruthy() })
    await act(async () => { screen.getByText('Done').click() })
    await waitFor(() => { expect(mockComplete).toHaveBeenCalledWith('distill-run-3') })
    expect(screen.getByText('Experiment distill-run-3 marked complete')).toBeTruthy()
  })

  it('dismisses the log status message', async () => {
    mockComplete.mockResolvedValue({ status: 'completed' })
    mockList.mockResolvedValue([mockExperiment])
    render(<ExperimentsPage />)
    await waitFor(() => { expect(screen.getByText('distill-run-3')).toBeTruthy() })
    await act(async () => { screen.getByText('Done').click() })
    await waitFor(() => { expect(screen.getByText('Experiment distill-run-3 marked complete')).toBeTruthy() })
    await act(async () => { screen.getByText('Dismiss').click() })
    expect(screen.queryByText('Experiment distill-run-3 marked complete')).toBeNull()
  })

  it('refetches experiments on refresh click', async () => {
    mockList.mockResolvedValue([mockExperiment])
    render(<ExperimentsPage />)
    await waitFor(() => { expect(screen.getByText('distill-run-3')).toBeTruthy() })
    expect(mockList).toHaveBeenCalledTimes(1)
    await act(async () => { screen.getByTestId('icon-refresh').click() })
    await waitFor(() => { expect(mockList).toHaveBeenCalledTimes(2) })
  })

  it('toggles auto refresh', async () => {
    mockList.mockResolvedValue([mockExperiment])
    render(<ExperimentsPage />)
    await waitFor(() => { expect(screen.getByText('distill-run-3')).toBeTruthy() })
    await act(async () => { screen.getByText('Refresh').click() })
    expect(screen.getByText('Auto')).toBeTruthy()
  })
})
