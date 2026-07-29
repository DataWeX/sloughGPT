import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

const {
  mockPush, mockAddToast, mockList, mockDelete,
} = vi.hoisted(() => ({
  mockPush: vi.fn(),
  mockAddToast: vi.fn(),
  mockList: vi.fn(),
  mockDelete: vi.fn(),
}))

vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => { const C = () => <span data-testid={`icon-${name}`}>{name}</span>; C.displayName = `Icon${name}`; return C }
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    Card: ({ children, className, onClick }: any) => <div className={className} onClick={onClick}>{children}</div>, CardContent: ({ children, className }: any) => <div className={className}>{children}</div>,
    CardHeader: passthrough, CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    EmptyCard: ({ message, action }: any) => <div><span>{message}</span>{action}</div>,
    Button: ({ children, onClick, variant, size, className, disabled, 'aria-label': ariaLabel }: any) => (
      <button onClick={onClick} className={className} disabled={disabled} aria-label={ariaLabel} data-variant={variant}>{children}</button>
    ),
    Input: ({ value, onChange, className, placeholder }: any) => (
      <input value={value} onChange={onChange} className={className} placeholder={placeholder} />
    ),
    Skeleton: ({ className }: any) => <div className={className} />,
    IconRefresh: iconMock('refresh'), IconPlus: iconMock('plus'), IconTrash: iconMock('trash'),
    AlertDialog: ({ open, onOpenChange, children }: any) => open ? <div data-testid="alert-dialog">{children}</div> : null,
    AlertDialogContent: ({ children }: any) => <div>{children}</div>,
    AlertDialogHeader: ({ children }: any) => <div>{children}</div>,
    AlertDialogTitle: ({ children }: any) => <div>{children}</div>,
    AlertDialogDescription: ({ children }: any) => <div>{children}</div>,
    AlertDialogFooter: ({ children }: any) => <div>{children}</div>,
    AlertDialogCancel: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    AlertDialogAction: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
  }
})

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: mockPush }),
  useParams: () => ({ id: 'test' }),
}))

vi.mock('@/lib/dataset-controller', () => ({
  datasetController: {
    list: mockList,
    delete: mockDelete,
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel ? sel({ addToast: mockAddToast }) : { addToast: mockAddToast },
}))

import DatasetsPage from './page'

const mockDatasets = [
  { id: 'ds1', name: 'Shakespeare', source: 'local', size: 102400, samples: 500, created_at: '2026-01-15T00:00:00Z' },
  { id: 'ds2', name: 'GitHub Code', source: 'github', size: 2048000, samples: 1200, created_at: '2026-02-20T00:00:00Z' },
  { id: 'ds3', name: 'Wikipedia', source: 'url', size: 51200, samples: 50, created_at: '2026-03-10T00:00:00Z' },
]

const mockDatasetDetail = {
  id: 'test',
  name: 'Test Dataset',
  source: 'local',
  size: 1024,
  samples: 100,
  created_at: '2026-01-01T00:00:00Z',
  tags: ['text'],
}

beforeEach(() => {
  vi.clearAllMocks()
  mockList.mockResolvedValue(mockDatasets)
  mockDelete.mockResolvedValue(undefined)
})

afterEach(cleanup)

describe('DatasetsPage', () => {
  it('renders header with title', async () => {
    render(<DatasetsPage />)
    await waitFor(() => expect(screen.getByText('Datasets')).toBeDefined())
    expect(screen.getByText('Refresh')).toBeDefined()
    expect(screen.getByText('Import')).toBeDefined()
  })

  it('renders dataset cards', async () => {
    render(<DatasetsPage />)
    await waitFor(() => expect(screen.getByText('Shakespeare')).toBeDefined())
    expect(screen.getByText('GitHub Code')).toBeDefined()
    expect(screen.getByText('Wikipedia')).toBeDefined()
  })

  it('shows dataset metadata (source, size, samples)', async () => {
    render(<DatasetsPage />)
    await waitFor(() => {
      expect(screen.getByText('500 samples')).toBeDefined()
      expect(screen.getByText('100.0 KB')).toBeDefined()
    })
  })

  it('filters datasets by search', async () => {
    render(<DatasetsPage />)
    await waitFor(() => expect(screen.getByText('Shakespeare')).toBeDefined())
    const searchInput = screen.getByPlaceholderText('Search datasets...')
    fireEvent.change(searchInput, { target: { value: 'shakes' } })
    await waitFor(() => {
      expect(screen.getByText('Shakespeare')).toBeDefined()
    })
    expect(screen.queryByText('GitHub Code')).toBeNull()
  })

  it('shows empty state when no datasets', async () => {
    mockList.mockResolvedValue([])
    render(<DatasetsPage />)
    await waitFor(() => expect(screen.getByText('No datasets yet.')).toBeDefined())
    expect(screen.getByText('Import Dataset')).toBeDefined()
  })

  it('navigates to dataset detail on card click', async () => {
    render(<DatasetsPage />)
    await waitFor(() => expect(screen.getByText('Shakespeare')).toBeDefined())
    fireEvent.click(screen.getByText('Shakespeare'))
    expect(mockPush).toHaveBeenCalledWith('/dataset/ds1')
  })

  it('calls delete and removes dataset from list', async () => {
    render(<DatasetsPage />)
    await waitFor(() => expect(screen.getByText('Shakespeare')).toBeDefined())
    const deleteButtons = screen.getAllByLabelText(/Delete/)
    await act(async () => { fireEvent.click(deleteButtons[0]) })
    await waitFor(() => { expect(screen.getByTestId('alert-dialog')).toBeTruthy() })
    const dialog = screen.getByTestId('alert-dialog')
    const confirmBtn = dialog.querySelector('button:last-child') as HTMLElement
    await act(async () => { confirmBtn.click() })
    await waitFor(() => {
      expect(mockDelete).toHaveBeenCalledWith('ds1')
      expect(mockAddToast).toHaveBeenCalledWith(expect.stringContaining('Shakespeare'), 'info')
    })
  })

  it('navigates to training on Import click', async () => {
    render(<DatasetsPage />)
    await waitFor(() => expect(screen.getByText('Import')).toBeDefined())
    fireEvent.click(screen.getByText('Import'))
    expect(mockPush).toHaveBeenCalledWith('/training')
  })
})
