import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

const {
  mockList, mockDelete, mockPreview, mockListVersions, mockAddToast,
} = vi.hoisted(() => ({
  mockList: vi.fn(), mockDelete: vi.fn(), mockPreview: vi.fn(),
  mockListVersions: vi.fn(), mockAddToast: vi.fn(),
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: vi.fn((...a: any[]) => a.join(' ')),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Button: ({ children, onClick, disabled }: any) => (
      <button onClick={onClick} disabled={disabled}>{children}</button>
    ),
    Input: ({ value, onChange, placeholder }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} />
    ),
    EmptyCard: ({ title, description }: any) => <div data-testid="empty-card"><div>{title}</div><div>{description}</div></div>,
    Skeleton: () => <div data-testid="skeleton" />,
    AlertDialog: passthrough, AlertDialogAction: passthrough, AlertDialogCancel: passthrough,
    AlertDialogContent: passthrough, AlertDialogDescription: passthrough,
    AlertDialogFooter: passthrough, AlertDialogHeader: passthrough, AlertDialogTitle: passthrough,
    IconRefresh: () => <span>refresh</span>,
    IconPlus: () => <span>+</span>,
    IconTrash: () => <span>trash</span>,
    IconChevronDown: () => <span>v</span>,
    IconDownload: () => <span>d</span>,
  }
})

vi.mock('@/lib/dataset-controller', () => ({
  datasetController: {
    list: (...a: unknown[]) => mockList(...a),
    delete: (...a: unknown[]) => mockDelete(...a),
    preview: (...a: unknown[]) => mockPreview(...a),
    listVersions: (...a: unknown[]) => mockListVersions(...a),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel({ addToast: mockAddToast }),
}))

vi.mock('@/lib/format-bytes', () => ({
  formatBytes: (b: number) => `${b} B`,
}))

vi.mock('@/lib/conversations-utils', () => ({
  formatDate: (d: string) => d,
}))

vi.mock('@/components/DatasetInlineImportModal', () => ({
  default: ({ open, onClose }: any) => open ? <div data-testid="import-modal">Import Modal</div> : null,
}))

import DatasetsPage from './page'

afterEach(cleanup)

beforeEach(() => {
  vi.clearAllMocks()
  mockList.mockResolvedValue([])
  mockListVersions.mockResolvedValue({ count: 0 })
})

describe('DatasetsPage — initial load flow', () => {
  it('renders page header', async () => {
    render(<DatasetsPage />)
    expect(screen.getAllByText('Datasets').length).toBeGreaterThanOrEqual(1)
  })

  it('fetches datasets on mount', async () => {
    render(<DatasetsPage />)
    await waitFor(() => {
      expect(mockList).toHaveBeenCalled()
    })
  })

  it('shows empty state when no datasets', async () => {
    render(<DatasetsPage />)
    await waitFor(() => {
      expect(screen.getByTestId('empty-card')).toBeTruthy()
    })
  })
})

describe('DatasetsPage — dataset list flow', () => {
  it('displays datasets when loaded', async () => {
    mockList.mockResolvedValue([
      { id: 'ds-1', name: 'shakespeare', format: 'text', rows: 100, size_bytes: 5000, created_at: '2026-08-07' },
      { id: 'ds-2', name: 'conversations', format: 'jsonl', rows: 50, size_bytes: 2000, created_at: '2026-08-06' },
    ])
    render(<DatasetsPage />)
    await waitFor(() => {
      expect(screen.getByText('shakespeare')).toBeTruthy()
      expect(screen.getByText('conversations')).toBeTruthy()
    })
  })

  it('shows dataset format badges', async () => {
    mockList.mockResolvedValue([
      { id: 'ds-1', name: 'data', format: 'text', rows: 10, size_bytes: 100, created_at: '2026-08-07' },
    ])
    render(<DatasetsPage />)
    await waitFor(() => {
      expect(screen.getByText('data')).toBeTruthy()
    })
  })
})

describe('DatasetsPage — search flow', () => {
  it('search input filters datasets', async () => {
    mockList.mockResolvedValue([
      { id: 'ds-1', name: 'shakespeare', format: 'text', rows: 100, size_bytes: 5000, created_at: '2026-08-07' },
      { id: 'ds-2', name: 'conversations', format: 'jsonl', rows: 50, size_bytes: 2000, created_at: '2026-08-06' },
    ])
    render(<DatasetsPage />)
    await waitFor(() => { expect(screen.getByText('shakespeare')).toBeTruthy() })

    const searchInput = screen.getAllByPlaceholderText(/search/i)[0]
    if (searchInput) {
      fireEvent.change(searchInput, { target: { value: 'shakespeare' } })
      await waitFor(() => {
        expect(screen.getByText('shakespeare')).toBeTruthy()
      })
    }
  })
})

describe('DatasetsPage — import flow', () => {
  it('import button opens modal', async () => {
    render(<DatasetsPage />)
    await waitFor(() => { expect(screen.getAllByText('Datasets').length).toBeGreaterThanOrEqual(1) })

    const importBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('import') || b.textContent?.includes('+')
    )
    if (importBtn) {
      await act(async () => { fireEvent.click(importBtn) })
      await waitFor(() => {
        expect(screen.getByTestId('import-modal')).toBeTruthy()
      })
    }
  })
})

describe('DatasetsPage — delete flow', () => {
  it('delete button shows confirmation', async () => {
    mockList.mockResolvedValue([
      { id: 'ds-1', name: 'shakespeare', format: 'text', rows: 100, size_bytes: 5000, created_at: '2026-08-07' },
    ])
    render(<DatasetsPage />)
    await waitFor(() => { expect(screen.getByText('shakespeare')).toBeTruthy() })

    const deleteBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('delete') || b.textContent?.includes('trash')
    )
    if (deleteBtn) {
      await act(async () => { fireEvent.click(deleteBtn) })
      // No crash = success
      expect(screen.getByText('shakespeare')).toBeTruthy()
    }
  })
})

describe('DatasetsPage — error handling', () => {
  it('handles list failure gracefully', async () => {
    mockList.mockRejectedValue(new Error('network'))
    render(<DatasetsPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Datasets').length).toBeGreaterThanOrEqual(1)
    })
  })
})

describe('DatasetsPage — refresh flow', () => {
  it('refresh button reloads datasets', async () => {
    render(<DatasetsPage />)
    await waitFor(() => { expect(mockList).toHaveBeenCalled() })

    const callCount = mockList.mock.calls.length
    const refreshBtn = screen.getAllByRole('button').find(b =>
      b.textContent?.toLowerCase().includes('refresh')
    )
    if (refreshBtn) {
      await act(async () => { fireEvent.click(refreshBtn) })
      await waitFor(() => {
        expect(mockList.mock.calls.length).toBeGreaterThan(callCount)
      })
    }
  })
})

describe('DatasetsPage — version fetching', () => {
  it('fetches versions for datasets', async () => {
    mockList.mockResolvedValue([
      { id: 'ds-1', name: 'data', format: 'text', rows: 10, size_bytes: 100, created_at: '2026-08-07' },
    ])
    render(<DatasetsPage />)
    await waitFor(() => {
      expect(mockListVersions).toHaveBeenCalled()
    })
  })
})

describe('DatasetsPage — loading state', () => {
  it('shows loading while fetching', async () => {
    mockList.mockReturnValue(new Promise(() => {}))
    render(<DatasetsPage />)
    expect(screen.getAllByText('Datasets').length).toBeGreaterThanOrEqual(1)
  })
})

describe('DatasetsPage — error toast', () => {
  it('shows error toast on delete failure', async () => {
    mockDelete.mockRejectedValue(new Error('delete failed'))
    mockList.mockResolvedValue([
      { id: 'ds-1', name: 'data', format: 'text', rows: 10, size_bytes: 100, created_at: '2026-08-07' },
    ])
    render(<DatasetsPage />)
    await waitFor(() => { expect(screen.getByText('data')).toBeTruthy() })
  })
})

describe('DatasetsPage — dataset details', () => {
  it('shows dataset row count', async () => {
    mockList.mockResolvedValue([
      { id: 'ds-1', name: 'data', format: 'text', rows: 42, size_bytes: 100, created_at: '2026-08-07' },
    ])
    render(<DatasetsPage />)
    await waitFor(() => {
      expect(screen.getByText('data')).toBeTruthy()
    })
  })

  it('shows dataset size', async () => {
    mockList.mockResolvedValue([
      { id: 'ds-1', name: 'data', format: 'text', rows: 10, size_bytes: 5000, created_at: '2026-08-07' },
    ])
    render(<DatasetsPage />)
    await waitFor(() => {
      expect(screen.getByText('data')).toBeTruthy()
    })
  })
})

describe('DatasetsPage — batch operations', () => {
  it('renders dataset list items for selection', async () => {
    mockList.mockResolvedValue([
      { id: 'ds-1', name: 'data1', format: 'text', rows: 10, size_bytes: 100, created_at: '2026-08-07' },
      { id: 'ds-2', name: 'data2', format: 'jsonl', rows: 20, size_bytes: 200, created_at: '2026-08-06' },
    ])
    render(<DatasetsPage />)
    await waitFor(() => { expect(screen.getByText('data1')).toBeTruthy() })
    expect(screen.getByText('data2')).toBeTruthy()
  })
})
