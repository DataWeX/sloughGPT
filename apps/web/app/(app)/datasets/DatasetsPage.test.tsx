import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

const {
  mockPush, mockAddToast, mockList, mockDelete, mockSearch, mockListVersions, mockExport,
} = vi.hoisted(() => ({
  mockPush: vi.fn(),
  mockAddToast: vi.fn(),
  mockList: vi.fn(),
  mockDelete: vi.fn(),
  mockSearch: vi.fn(),
  mockListVersions: vi.fn(),
  mockExport: vi.fn(),
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
    IconRefresh: iconMock('refresh'), IconPlus: iconMock('plus'), IconTrash: iconMock('trash'), IconChevronDown: iconMock('chevron-down'), IconDownload: iconMock('download'), IconPlay: iconMock('play'),
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
    preview: vi.fn(),
    search: mockSearch,
    listVersions: mockListVersions,
    export: mockExport,
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel ? sel({ addToast: mockAddToast }) : { addToast: mockAddToast },
}))

vi.mock('@/components/DatasetInlineImportModal', () => ({
  __esModule: true,
  default: ({ open, onOpenChange }: any) => open ? <div data-testid="import-modal">Import Modal</div> : null,
}))

vi.mock('@/lib/conversations-utils', () => ({
  formatDate: (d: string) => d,
}))

vi.mock('@/lib/format-bytes', () => ({
  formatBytes: (b: number) => {
    if (b < 1024) return `${b} B`
    if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`
    return `${(b / (1024 * 1024)).toFixed(1)} MB`
  },
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
  mockSearch.mockResolvedValue(mockDatasets)
  mockListVersions.mockResolvedValue({ versions: [], count: 0 })
  mockExport.mockResolvedValue(new Blob(['test'], { type: 'application/json' }))
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

  it('shows a version count badge when the dataset has versions', async () => {
    mockListVersions.mockImplementation((id: string) =>
      id === 'ds1'
        ? Promise.resolve({ versions: ['20260801120000', '20260801110000'], count: 2 })
        : Promise.resolve({ versions: [], count: 0 }),
    )
    render(<DatasetsPage />)
    await waitFor(() => {
      expect(screen.getByText('2 versions')).toBeDefined()
    })
  })

  it('does not render a version badge for versionless datasets', async () => {
    mockListVersions.mockResolvedValue({ versions: [], count: 0 })
    render(<DatasetsPage />)
    await waitFor(() => expect(screen.getByText('Shakespeare')).toBeDefined())
    await new Promise(r => setTimeout(r, 20))
    expect(screen.queryByText(/\d+ versions?/)).toBeNull()
  })

  it('exports a dataset to a downloadable jsonl file', async () => {
    const createObjectURL = vi.fn(() => 'blob:test')
    vi.stubGlobal('URL', { ...URL, createObjectURL, revokeObjectURL: vi.fn() })
    render(<DatasetsPage />)
    await waitFor(() => expect(screen.getByText('Shakespeare')).toBeDefined())

    fireEvent.click(screen.getByLabelText('Export Shakespeare'))

    await waitFor(() => {
      expect(mockExport).toHaveBeenCalledWith('ds1', 'jsonl')
      expect(createObjectURL).toHaveBeenCalled()
      expect(mockAddToast).toHaveBeenCalledWith(expect.stringContaining('Exported'), 'success')
    })
    vi.unstubAllGlobals()
  })

  it('shows an error toast when export fails', async () => {
    mockExport.mockRejectedValue(new Error('network'))
    vi.stubGlobal('URL', { ...URL, createObjectURL: vi.fn(), revokeObjectURL: vi.fn() })
    render(<DatasetsPage />)
    await waitFor(() => expect(screen.getByText('Shakespeare')).toBeDefined())

    fireEvent.click(screen.getByLabelText('Export Shakespeare'))

    await waitFor(() => {
      expect(mockAddToast).toHaveBeenCalledWith('Export failed', 'error')
    })
    vi.unstubAllGlobals()
  })

  it('filters datasets by search', async () => {
    mockSearch.mockResolvedValue([mockDatasets[0]])
    render(<DatasetsPage />)
    await waitFor(() => expect(screen.getByText('Shakespeare')).toBeDefined())
    const searchInput = screen.getByPlaceholderText('Search datasets...')
    await act(async () => { fireEvent.change(searchInput, { target: { value: 'shakes' } }) })
    await waitFor(() => {
      expect(mockSearch).toHaveBeenCalledWith('shakes')
    })
  })

  it('shows a searching indicator while the search is in flight', async () => {
    let resolveSearch: (v: unknown) => void
    mockSearch.mockReturnValue(new Promise(res => { resolveSearch = res }))
    render(<DatasetsPage />)
    await waitFor(() => expect(screen.getByText('Shakespeare')).toBeDefined())
    const searchInput = screen.getByPlaceholderText('Search datasets...')
    await act(async () => { fireEvent.change(searchInput, { target: { value: 'pending' } }) })
    await waitFor(() => {
      expect(screen.getByRole('status')).toBeDefined()
    })
    await act(async () => { resolveSearch!([mockDatasets[0]]) })
    await waitFor(() => {
      expect(screen.queryByRole('status')).toBeNull()
    })
  })

  it('shows empty state when no datasets', async () => {
    mockList.mockResolvedValue([])
    render(<DatasetsPage />)
    await waitFor(() => expect(screen.getByText('No datasets yet')).toBeDefined(), { timeout: 3000 })
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
    const confirmBtn = Array.from(dialog.querySelectorAll('button')).find(b => b.textContent?.trim() === 'Delete') as HTMLElement
    await act(async () => { confirmBtn.click() })
    await waitFor(() => {
      expect(mockDelete).toHaveBeenCalledWith('ds3')
      expect(mockAddToast).toHaveBeenCalledWith(expect.stringContaining('Wikipedia'), 'info', undefined, expect.any(Function))
    })
  })

  it('opens import modal on Import click', async () => {
    render(<DatasetsPage />)
    await waitFor(() => expect(screen.getByText('Import')).toBeDefined())
    fireEvent.click(screen.getByText('Import'))
    await waitFor(() => expect(screen.getByTestId('import-modal')).toBeTruthy())
  })
})
