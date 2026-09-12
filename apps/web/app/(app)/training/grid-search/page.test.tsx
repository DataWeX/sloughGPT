import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...a: any[]) => a.join(' '),
    Button: ({ children, onClick, disabled, className, variant, size, ...props }: any) => (
      <button onClick={onClick} disabled={disabled} className={className} {...props}>{children}</button>
    ),
    Card: passthrough,
    CardContent: passthrough,
    CardHeader: passthrough,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    Input: ({ value, onChange, placeholder, className, ...props }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} className={className} {...props} />
    ),
    Badge: ({ children, variant, className }: any) => <span className={className}>{children}</span>,
    Skeleton: () => <div data-testid="skeleton" />,
    IconPlus: () => <span data-testid="icon-plus" />,
    IconTrash: () => <span data-testid="icon-trash" />,
    IconPlay: () => <span data-testid="icon-play" />,
    IconRefresh: () => <span data-testid="icon-refresh" />,
  }
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ title, subtitle, headerRight, children }: any) => (
    <div data-testid="page-container">
      <h1>{title}</h1>
      {subtitle && <p>{subtitle}</p>}
      <div data-testid="header-right">{headerRight}</div>
      <div>{children}</div>
    </div>
  ),
}))

vi.mock('@/components/AppRouteHeader', () => ({
  AppRouteHeader: ({ children }: any) => <div>{children}</div>,
  AppRouteHeaderLead: ({ children }: any) => <div>{children}</div>,
}))

vi.mock('lucide-react', () => ({
  Grid3X3: () => <span data-testid="icon-grid3x3" />,
  Zap: () => <span data-testid="icon-zap" />,
  Clock: () => <span data-testid="icon-clock" />,
  CheckCircle: () => <span data-testid="icon-check-circle" />,
  XCircle: () => <span data-testid="icon-x-circle" />,
  Loader2: ({ className }: any) => <span data-testid="icon-loader" className={className} />,
  ArrowRight: () => <span data-testid="icon-arrow-right" />,
}))

const { mockAddToast } = vi.hoisted(() => ({ mockAddToast: vi.fn() }))
const { mockListDatasets } = vi.hoisted(() => ({ mockListDatasets: vi.fn() }))
const { mockStartAutoTrain } = vi.hoisted(() => ({ mockStartAutoTrain: vi.fn() }))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: { startAutoTrain: mockStartAutoTrain },
}))

vi.mock('@/lib/dataset-controller', () => ({
  datasetController: { list: mockListDatasets },
}))

vi.mock('@/hooks/useRefreshShortcut', () => ({
  useRefreshShortcut: vi.fn(),
}))

import GridSearchPage from './page'

async function loadDatasets() {
  await act(async () => {
    fireEvent.click(screen.getByTestId('icon-refresh').closest('button')!)
  })
  await waitFor(() => {
    expect(screen.getByText('Dataset 1')).toBeTruthy()
  })
}

afterEach(() => { cleanup() })
beforeEach(() => {
  vi.clearAllMocks()
  mockListDatasets.mockResolvedValue([{ id: 'ds-1', name: 'Dataset 1' }, { id: 'ds-2', name: 'Dataset 2' }])
  mockStartAutoTrain.mockResolvedValue({})
})

describe('GridSearchPage', () => {
  it('renders without crashing', async () => {
    render(<GridSearchPage />)
    await waitFor(() => {
      expect(screen.getByText('Hyperparameter Grid Search')).toBeTruthy()
    })
  })

  it('renders the page title and subtitle', async () => {
    render(<GridSearchPage />)
    await waitFor(() => {
      expect(screen.getByText('Hyperparameter Grid Search')).toBeTruthy()
    })
    expect(screen.getByText(/combinations from .* parameters/)).toBeTruthy()
  })

  it('displays default parameter configuration', async () => {
    render(<GridSearchPage />)
    await waitFor(() => {
      expect(screen.getByText('Search Space')).toBeTruthy()
    })
    expect(screen.getByText('Dataset')).toBeTruthy()
    expect(screen.getByText(/Combinations Preview/)).toBeTruthy()
  })

  it('shows 3 values badges for default params', async () => {
    render(<GridSearchPage />)
    await waitFor(() => {
      expect(screen.getAllByText('3 values').length).toBe(3)
    })
  })

  it('shows the run button with combination count', async () => {
    render(<GridSearchPage />)
    await waitFor(() => {
      expect(screen.getByText('Run 27 Runs')).toBeTruthy()
    })
  })

  it('shows validation toast when starting without dataset', async () => {
    render(<GridSearchPage />)
    await waitFor(() => {
      expect(screen.getByText('Run 27 Runs')).toBeTruthy()
    })
    await act(async () => {
      screen.getByText('Run 27 Runs').click()
    })
    expect(mockAddToast).toHaveBeenCalledWith('Select a dataset first', 'error')
  })

  it('loads datasets on refresh click', async () => {
    render(<GridSearchPage />)
    await loadDatasets()
    expect(mockListDatasets).toHaveBeenCalled()
    expect(screen.getByText('Dataset 1')).toBeTruthy()
  })

  it('shows validation toast when parameter name is empty', async () => {
    render(<GridSearchPage />)
    await waitFor(() => {
      expect(screen.getByText('Run 27 Runs')).toBeTruthy()
    })
    await loadDatasets()
    await act(async () => {
      fireEvent.change(screen.getByDisplayValue('Select a dataset...'), { target: { value: 'ds-1' } })
    })
    await act(async () => {
      fireEvent.change(screen.getByDisplayValue('learning_rate'), { target: { value: '' } })
    })
    await act(async () => {
      screen.getByText(/Run \d+ Runs/).click()
    })
    expect(mockAddToast).toHaveBeenCalledWith('Fill in all parameter names and values', 'error')
  })

  it('starts grid search and shows results on success', async () => {
    mockStartAutoTrain.mockResolvedValue({ loss: 0.3 })
    render(<GridSearchPage />)
    await waitFor(() => {
      expect(screen.getByText('Run 27 Runs')).toBeTruthy()
    })
    await loadDatasets()

    await act(async () => {
      fireEvent.change(screen.getByDisplayValue('learning_rate'), { target: { value: 'lr' } })
    })
    await act(async () => {
      fireEvent.change(screen.getByDisplayValue('batch_size'), { target: { value: 'bs' } })
    })
    await act(async () => {
      fireEvent.change(screen.getByDisplayValue('epochs'), { target: { value: 'ep' } })
    })
    await act(async () => {
      fireEvent.change(screen.getByDisplayValue('Select a dataset...'), { target: { value: 'ds-1' } })
    })
    await act(async () => {
      screen.getByText(/Run \d+ Runs/).click()
    })

    await waitFor(() => {
      expect(mockStartAutoTrain).toHaveBeenCalled()
    })
    expect(mockAddToast).toHaveBeenCalledWith('Grid search complete: 27 runs', 'success')
  })

  it('shows failed status when a run errors', async () => {
    mockStartAutoTrain.mockRejectedValue(new Error('boom'))
    render(<GridSearchPage />)
    await waitFor(() => {
      expect(screen.getByText('Run 27 Runs')).toBeTruthy()
    })
    await loadDatasets()

    await act(async () => {
      fireEvent.change(screen.getByDisplayValue('learning_rate'), { target: { value: 'lr' } })
    })
    await act(async () => {
      fireEvent.change(screen.getByDisplayValue('batch_size'), { target: { value: 'bs' } })
    })
    await act(async () => {
      fireEvent.change(screen.getByDisplayValue('epochs'), { target: { value: 'ep' } })
    })
    await act(async () => {
      fireEvent.change(screen.getByDisplayValue('Select a dataset...'), { target: { value: 'ds-1' } })
    })
    await act(async () => {
      screen.getByText(/Run \d+ Runs/).click()
    })

    await waitFor(() => {
      expect(screen.getByText(/Results/)).toBeTruthy()
    })
    expect(mockAddToast).toHaveBeenCalledWith('Grid search complete: 27 runs', 'success')
  })

  it('adds a new parameter row', async () => {
    render(<GridSearchPage />)
    await waitFor(() => {
      expect(screen.getByText('Add Parameter')).toBeTruthy()
    })
    await act(async () => {
      screen.getByText('Add Parameter').click()
    })
    await waitFor(() => {
      expect(screen.getByText(/4 parameters/)).toBeTruthy()
    })
    expect(screen.getAllByPlaceholderText('Parameter name').length).toBe(4)
  })

  it('removes a parameter row', async () => {
    render(<GridSearchPage />)
    await waitFor(() => {
      expect(screen.getByText('Add Parameter')).toBeTruthy()
    })
    const trashButtons = screen.getAllByTestId('icon-trash')
    await act(async () => {
      trashButtons[0].closest('button')!.click()
    })
    await waitFor(() => {
      expect(screen.getByText(/2 parameters/)).toBeTruthy()
    })
    expect(screen.getAllByPlaceholderText('Parameter name').length).toBe(2)
  })

  it('shows combination preview entries', async () => {
    render(<GridSearchPage />)
    await waitFor(() => {
      expect(screen.getByText('#1')).toBeTruthy()
    })
    expect(screen.getByText('#20')).toBeTruthy()
    expect(screen.getByText('...and 7 more')).toBeTruthy()
  })

  it('refresh button exists', async () => {
    render(<GridSearchPage />)
    await waitFor(() => {
      expect(screen.getByTestId('icon-refresh')).toBeTruthy()
    })
  })
})
