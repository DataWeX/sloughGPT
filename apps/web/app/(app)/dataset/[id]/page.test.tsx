import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'
import { act } from 'react'

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...args: any[]) => args.join(' '),
    Button: ({ children, onClick, disabled, variant, size, className, 'aria-label': ariaLabel }: any) => (
      <button onClick={onClick} disabled={disabled} data-variant={variant} aria-label={ariaLabel} className={className}>{children}</button>
    ),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children, className }: any) => <div className={className}>{children}</div>,
    CardDescription: ({ children }: any) => <p>{children}</p>,
    Input: ({ value, onChange, className, autoFocus }: any) => (
      <input value={value} onChange={onChange} className={className} autoFocus={autoFocus} />
    ),
    Label: ({ children }: any) => <label>{children}</label>,
    Skeleton: () => <div data-testid="skeleton" />,
    Badge: ({ children, variant, size, className }: any) => <span data-variant={variant} className={className}>{children}</span>,
    StatCard: ({ label, value }: any) => <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span></div>,
    KpiGrid: ({ children, columns }: any) => <div data-columns={columns}>{children}</div>,
    Breadcrumbs: ({ items, className }: any) => <nav aria-label="Breadcrumb" className={className}>{items?.map((item: any, i: number) => <span key={i}>{item.label}</span>)}</nav>,
    IconTrash: () => <span data-testid="icon-trash">trash</span>,
    IconDownload: () => <span data-testid="icon-download">download</span>,
    IconEdit: () => <span data-testid="icon-edit">edit</span>,
    IconCheck: () => <span data-testid="icon-check">check</span>,
    IconX: () => <span data-testid="icon-x">x</span>,
    IconRefresh: () => <span data-testid="icon-refresh">refresh</span>,
    IconClock: () => <span data-testid="icon-clock">clock</span>,
    IconChevronDown: () => <span data-testid="icon-chevron">chevron</span>,
    AlertDialog: ({ open, children }: any) => open ? <div data-testid="alert-dialog">{children}</div> : null,
    AlertDialogAction: ({ children, onClick, className }: any) => <button onClick={onClick} className={className}>{children}</button>,
    AlertDialogCancel: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
    AlertDialogContent: ({ children }: any) => <div>{children}</div>,
    AlertDialogDescription: ({ children }: any) => <p>{children}</p>,
    AlertDialogFooter: ({ children }: any) => <div>{children}</div>,
    AlertDialogHeader: ({ children }: any) => <div>{children}</div>,
    AlertDialogTitle: ({ children }: any) => <div>{children}</div>,
  }
})

const mocks = vi.hoisted(() => ({
  push: vi.fn(),
  addToast: vi.fn(),
  get: vi.fn(),
  getStats: vi.fn(),
  preview: vi.fn(),
  listVersions: vi.fn(),
  createVersion: vi.fn(),
  restoreVersion: vi.fn(),
  delete: vi.fn(),
  update: vi.fn(),
  convertToMessages: vi.fn(),
  exportDataset: vi.fn(),
}))

const stableRouter = { push: vi.fn() }
vi.mock('next/navigation', () => ({ useParams: () => ({ id: 'ds-1' }), useRouter: () => stableRouter }))
vi.mock('@/lib/toast-store', () => ({ useToastStore: (sel: any) => sel({ addToast: mocks.addToast }) }))
vi.mock('@/lib/dataset-controller', () => ({
  datasetController: {
    get: mocks.get,
    getStats: mocks.getStats,
    preview: mocks.preview,
    listVersions: mocks.listVersions,
    createVersion: mocks.createVersion,
    restoreVersion: mocks.restoreVersion,
    delete: mocks.delete,
    update: mocks.update,
    convertToMessages: mocks.convertToMessages,
    export: mocks.exportDataset,
  },
}))
vi.mock('@/lib/format-bytes', () => ({ formatBytes: (n: number) => `${n} bytes` }))
vi.mock('@/lib/download-utils', () => ({ downloadBlob: vi.fn(), downloadJson: vi.fn() }))
vi.mock('@/components/DatasetPreview', () => ({
  DatasetPreview: ({ datasetId }: any) => <div data-testid="dataset-preview">Preview</div>,
}))
vi.mock('@/components/dataset/DatasetQualityCard', () => ({
  DatasetQualityCard: ({ datasetId }: any) => <div data-testid="quality-card">Quality</div>,
}))
vi.mock('@/components/dataset/DatasetInsightsCard', () => ({
  DatasetInsightsCard: ({ preview }: any) => <div data-testid="insights-card">Insights</div>,
}))
vi.mock('next/dynamic', () => ({ default: () => () => <div data-testid="import-modal" /> }))

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ title, children, loading, loadingContent }: any) => (
    <div className="sl-page mx-auto max-w-4xl">
      <h1>{loading ? '...' : title}</h1>
      {loading ? loadingContent : children}
    </div>
  ),
}))

import Page from './page'

const SAMPLE_DATASET: any = {
  id: 'ds-1', name: 'Shakespeare Dataset', type: 'conversations', source: 'local',
  size: 1024, samples: 500, tags: ['english', 'classic'], created_at: '2026-08-01T00:00:00Z',
}

beforeEach(() => {
  vi.clearAllMocks()
  mocks.get.mockResolvedValue(SAMPLE_DATASET)
  mocks.getStats.mockResolvedValue({ format: 'jsonl', lines: 100, avg_length: 256, chars: 25600, suggested_method: 'distill' })
  mocks.listVersions.mockResolvedValue({ versions: ['20260801120000'] })
  mocks.preview.mockResolvedValue({ rows: [] })
})

afterEach(() => cleanup())

describe('DatasetDetailPage', () => {
  it('renders loading state', async () => {
    mocks.get.mockImplementation(() => new Promise(() => {}))
    render(<Page />)
    expect(screen.getByText('...')).toBeTruthy()
  })

  it('shows dataset name and breadcrumbs', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getAllByText('Shakespeare Dataset').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getByText('Datasets', { selector: 'span' })).toBeTruthy()
  })

  it('shows KPI grid', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getAllByText('Shakespeare Dataset').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getByTestId('stat-ID')).toBeTruthy()
    expect(screen.getByTestId('stat-Source')).toBeTruthy()
    expect(screen.getByTestId('stat-Size')).toBeTruthy()
  })

  it('shows tags', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getAllByText('Shakespeare Dataset').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getByText('english')).toBeTruthy()
    expect(screen.getByText('classic')).toBeTruthy()
  })

  it('shows train buttons', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Train on this dataset')).toBeTruthy()
    })
    expect(screen.getByText('Fine-tune')).toBeTruthy()
  })

  it('shows stats card', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Stats')).toBeTruthy()
    })
    expect(screen.getByText('jsonl')).toBeTruthy()
    expect(screen.getByText('distill')).toBeTruthy()
  })

  it('shows quality and insights cards', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByTestId('quality-card')).toBeTruthy()
    })
    expect(screen.getByTestId('insights-card')).toBeTruthy()
  })

  it('shows versions', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Create snapshot')).toBeTruthy()
    })
  })

  it('shows empty versions', async () => {
    mocks.listVersions.mockResolvedValue({ versions: [] })
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText(/No snapshots yet/)).toBeTruthy()
    })
  })

  it('creates version', async () => {
    mocks.createVersion.mockResolvedValue({})
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Create snapshot')).toBeTruthy()
    })
    fireEvent.click(screen.getByText('Create snapshot'))
    await waitFor(() => {
      expect(mocks.createVersion).toHaveBeenCalledWith('ds-1')
      expect(mocks.addToast).toHaveBeenCalledWith('Snapshot created', 'success')
    })
  })

  it('shows error toast on version creation failure', async () => {
    mocks.createVersion.mockRejectedValue(new Error('fail'))
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Create snapshot')).toBeTruthy()
    })
    fireEvent.click(screen.getByText('Create snapshot'))
    await waitFor(() => {
      expect(mocks.addToast).toHaveBeenCalledWith('Could not snapshot', 'error')
    })
  })

  it('shows delete dialog', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getAllByText('Shakespeare Dataset').length).toBeGreaterThanOrEqual(1)
    })
    const deleteBtn = screen.getAllByRole('button').find(b => b.textContent?.includes('Delete') && !b.textContent?.includes('Webhook'))
    fireEvent.click(deleteBtn!)
    expect(screen.getByTestId('alert-dialog')).toBeTruthy()
  })

  it('deletes dataset', async () => {
    mocks.delete.mockResolvedValue({})
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Import Data')).toBeTruthy()
    })
    const allBtns = screen.getAllByRole('button')
    const deleteBtn = allBtns.find(b => {
      const text = b.textContent || ''
      return text.includes('Delete') && text.includes('trash')
    })
    expect(deleteBtn).toBeTruthy()
    fireEvent.click(deleteBtn!)
    await waitFor(() => {
      expect(screen.getByTestId('alert-dialog')).toBeTruthy()
    })
    const dialogBtns = screen.getByTestId('alert-dialog').querySelectorAll('button')
    const confirmBtn = Array.from(dialogBtns).find(b => b.textContent?.includes('Delete'))
    fireEvent.click(confirmBtn!)
    await waitFor(() => {
      expect(mocks.delete).toHaveBeenCalledWith('ds-1')
    })
  })

  it('shows export menu', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getAllByText('Shakespeare Dataset').length).toBeGreaterThanOrEqual(1)
    })
    const exportBtn = screen.getAllByRole('button').find(b => b.textContent?.includes('Export'))
    fireEvent.click(exportBtn!)
    await waitFor(() => {
      expect(screen.getByText('Export as JSONL')).toBeTruthy()
      expect(screen.getByText('Export as CSV')).toBeTruthy()
    })
  })

  it('shows dataset not found', async () => {
    mocks.get.mockResolvedValue(null)
    render(<Page />)
    await waitFor(() => {
      expect(screen.getByText('Dataset not found')).toBeTruthy()
    })
  })

  it('shows error toast on load failure', async () => {
    mocks.get.mockRejectedValue(new Error('boom'))
    render(<Page />)
    await waitFor(() => {
      expect(mocks.addToast).toHaveBeenCalledWith('Could not load dataset', 'error')
    })
  })

  it('starts rename', async () => {
    render(<Page />)
    await waitFor(() => {
      expect(screen.getAllByText('Shakespeare Dataset').length).toBeGreaterThanOrEqual(1)
    })
    fireEvent.click(screen.getByLabelText('Rename dataset'))
    expect(screen.getByDisplayValue('Shakespeare Dataset')).toBeTruthy()
  })
})
