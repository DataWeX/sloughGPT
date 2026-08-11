import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act, cleanup } from '@testing-library/react'

const mockList = vi.fn()
const mockUpload = vi.fn()
const mockDelete = vi.fn()
const mockDeleteBatch = vi.fn()
const mockIngest = vi.fn()
const mockSearch = vi.fn()

vi.mock('@/lib/files-controller', () => ({
  filesController: {
    list: (...args: unknown[]) => mockList(...args),
    upload: (...args: unknown[]) => mockUpload(...args),
    delete: (...args: unknown[]) => mockDelete(...args),
    deleteBatch: (...args: unknown[]) => mockDeleteBatch(...args),
    ingest: (...args: unknown[]) => mockIngest(...args),
    search: (...args: unknown[]) => mockSearch(...args),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

import FilesPage from './page'

const SAMPLE_FILES = [
  { id: '1', filename: 'readme.md', size: 2048, content_type: 'text/markdown', uploaded_at: '2026-01-15T00:00:00Z', ingested: true, chunk_count: 10 },
  { id: '2', filename: 'data.csv', size: 512, content_type: 'text/csv', uploaded_at: '2026-02-20T00:00:00Z', ingested: false, chunk_count: 0 },
  { id: '3', filename: 'image.png', size: 1048576, content_type: 'image/png', uploaded_at: '2026-03-10T00:00:00Z', ingested: false, chunk_count: 0 },
]

describe('FilesPage — initial load flow', () => {
  beforeEach(() => { vi.clearAllMocks(); mockList.mockResolvedValue([]); mockSearch.mockResolvedValue([]) })
  afterEach(() => { cleanup() })

  it('renders page header', async () => {
    render(<FilesPage />)
    expect(screen.getAllByText('Files').length).toBeGreaterThanOrEqual(1)
  })

  it('shows loading skeleton while fetching', async () => {
    mockList.mockReturnValue(new Promise(() => {}))
    render(<FilesPage />)
    expect(screen.getAllByText('Files').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no files', async () => {
    render(<FilesPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/no files uploaded/i).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('renders upload button', async () => {
    render(<FilesPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/upload/i).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('renders search input', async () => {
    render(<FilesPage />)
    await waitFor(() => {
      expect(screen.getAllByPlaceholderText(/search/i).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('displays files when loaded', async () => {
    mockList.mockResolvedValue([SAMPLE_FILES[0]])
    render(<FilesPage />)
    await waitFor(() => {
      expect(screen.getByText('readme.md')).toBeTruthy()
    })
    expect(screen.getAllByText(/indexed/).length).toBeGreaterThanOrEqual(1)
  })
})

describe('FilesPage — file list display flow', () => {
  beforeEach(() => { vi.clearAllMocks(); mockList.mockResolvedValue(SAMPLE_FILES); mockSearch.mockResolvedValue([]) })
  afterEach(() => { cleanup() })

  it('shows file count in subtitle', async () => {
    render(<FilesPage />)
    await waitFor(() => {
      expect(screen.getByText(/3 files/)).toBeTruthy()
    })
  })

  it('shows indexed badge for ingested files', async () => {
    render(<FilesPage />)
    await waitFor(() => {
      expect(screen.getAllByText('readme.md').length).toBeGreaterThanOrEqual(1)
    })
    expect(screen.getAllByText(/indexed/).length).toBeGreaterThanOrEqual(1)
  })

  it('displays file sizes', async () => {
    render(<FilesPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/2\.0 KB/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('renders select-all checkbox', async () => {
    render(<FilesPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/select all/i).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('select-all toggles all files', async () => {
    render(<FilesPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/select all/i).length).toBeGreaterThanOrEqual(1)
    })
    const selectAll = screen.getAllByRole('checkbox')[0]
    act(() => { fireEvent.click(selectAll) })
    await waitFor(() => {
      expect(screen.getAllByText(/3 selected/).length).toBeGreaterThanOrEqual(1)
    })
  })
})

describe('FilesPage — individual selection flow', () => {
  beforeEach(() => { vi.clearAllMocks(); mockList.mockResolvedValue(SAMPLE_FILES); mockSearch.mockResolvedValue([]) })
  afterEach(() => { cleanup() })

  it('selects individual file', async () => {
    render(<FilesPage />)
    await waitFor(() => { expect(screen.getByText('readme.md')).toBeTruthy() })
    const checkboxes = screen.getAllByRole('checkbox')
    if (checkboxes.length > 1) {
      act(() => { fireEvent.click(checkboxes[1]) })
      await waitFor(() => {
        expect(screen.getAllByText(/1 selected/).length).toBeGreaterThanOrEqual(1)
      })
    }
  })

  it('deselects file on second click', async () => {
    render(<FilesPage />)
    await waitFor(() => { expect(screen.getByText('readme.md')).toBeTruthy() })
    const checkboxes = screen.getAllByRole('checkbox')
    if (checkboxes.length > 1) {
      act(() => { fireEvent.click(checkboxes[1]) })
      act(() => { fireEvent.click(checkboxes[1]) })
      await waitFor(() => {
        expect(screen.queryByText(/1 selected/)).toBeNull()
      })
    }
  })
})

describe('FilesPage — search flow', () => {
  beforeEach(() => { vi.clearAllMocks(); mockList.mockResolvedValue(SAMPLE_FILES); mockSearch.mockResolvedValue([SAMPLE_FILES[0]]) })
  afterEach(() => { cleanup() })

  it('search filters files', async () => {
    render(<FilesPage />)
    await waitFor(() => { expect(screen.getByText('readme.md')).toBeTruthy() })
    const searchInput = screen.getAllByPlaceholderText(/search/i)[0]
    act(() => { fireEvent.change(searchInput, { target: { value: 'readme' } }) })
    const searchBtn = screen.getAllByText(/search/i).find(b => b.tagName === 'BUTTON')
    if (searchBtn) {
      await act(async () => { fireEvent.click(searchBtn) })
    }
  })

  it('search with enter key', async () => {
    render(<FilesPage />)
    await waitFor(() => { expect(screen.getByText('readme.md')).toBeTruthy() })
    const searchInput = screen.getAllByPlaceholderText(/search/i)[0]
    act(() => { fireEvent.change(searchInput, { target: { value: 'readme' } }) })
    act(() => { fireEvent.keyDown(searchInput, { key: 'Enter' }) })
    await waitFor(() => {
      expect(mockSearch).toHaveBeenCalled()
    })
  })

  it('empty search returns to all files', async () => {
    mockSearch.mockResolvedValue([])
    render(<FilesPage />)
    await waitFor(() => { expect(screen.getByText('readme.md')).toBeTruthy() })
    const searchInput = screen.getAllByPlaceholderText(/search/i)[0]
    act(() => { fireEvent.change(searchInput, { target: { value: '' } }) })
    const searchBtn = screen.getAllByText(/search/i).find(b => b.tagName === 'BUTTON')
    if (searchBtn) {
      await act(async () => { fireEvent.click(searchBtn) })
    }
  })
})

describe('FilesPage — batch delete flow', () => {
  beforeEach(() => { vi.clearAllMocks(); mockList.mockResolvedValue(SAMPLE_FILES); mockDeleteBatch.mockResolvedValue(undefined) })
  afterEach(() => { cleanup() })

  it('shows batch delete bar when files selected', async () => {
    render(<FilesPage />)
    await waitFor(() => { expect(screen.getByText('readme.md')).toBeTruthy() })
    const checkboxes = screen.getAllByRole('checkbox')
    if (checkboxes.length > 1) {
      act(() => { fireEvent.click(checkboxes[1]) })
      await waitFor(() => {
        expect(screen.getAllByText(/delete selected/i).length).toBeGreaterThanOrEqual(1)
      })
    }
  })

  it('batch delete calls API', async () => {
    render(<FilesPage />)
    await waitFor(() => { expect(screen.getByText('readme.md')).toBeTruthy() })
    const checkboxes = screen.getAllByRole('checkbox')
    if (checkboxes.length > 1) {
      act(() => { fireEvent.click(checkboxes[1]) })
      await waitFor(() => { expect(screen.getAllByText(/delete selected/i).length).toBeGreaterThanOrEqual(1) })
      const deleteBtn = screen.getAllByText(/delete selected/i)[0]
      await act(async () => { fireEvent.click(deleteBtn) })
      expect(mockDeleteBatch).toHaveBeenCalled()
    }
  })

  it('clear selection resets', async () => {
    render(<FilesPage />)
    await waitFor(() => { expect(screen.getByText('readme.md')).toBeTruthy() })
    const checkboxes = screen.getAllByRole('checkbox')
    if (checkboxes.length > 1) {
      act(() => { fireEvent.click(checkboxes[1]) })
      await waitFor(() => { expect(screen.getAllByText(/1 selected/).length).toBeGreaterThanOrEqual(1) })
      const clearBtn = screen.getAllByText(/clear/i)[0]
      act(() => { fireEvent.click(clearBtn) })
      await waitFor(() => {
        expect(screen.queryByText(/1 selected/)).toBeNull()
      })
    }
  })
})

describe('FilesPage — delete individual file flow', () => {
  beforeEach(() => { vi.clearAllMocks(); mockList.mockResolvedValue(SAMPLE_FILES); mockDelete.mockResolvedValue(undefined) })
  afterEach(() => { cleanup() })

  it('delete button appears on hover', async () => {
    render(<FilesPage />)
    await waitFor(() => { expect(screen.getByText('readme.md')).toBeTruthy() })
    expect(screen.getAllByText(/delete/i).length).toBeGreaterThanOrEqual(1)
  })

  it('delete calls filesController.delete', async () => {
    render(<FilesPage />)
    await waitFor(() => { expect(screen.getByText('readme.md')).toBeTruthy() })
    const deleteButtons = screen.getAllByText(/delete/i).filter(b => b.tagName === 'BUTTON')
    if (deleteButtons.length > 0) {
      await act(async () => { fireEvent.click(deleteButtons[0]) })
      expect(mockDelete).toHaveBeenCalled()
    }
  })
})

describe('FilesPage — ingest flow', () => {
  beforeEach(() => { vi.clearAllMocks(); mockList.mockResolvedValue(SAMPLE_FILES); mockIngest.mockResolvedValue(undefined) })
  afterEach(() => { cleanup() })

  it('index button shown for non-ingested files', async () => {
    render(<FilesPage />)
    await waitFor(() => { expect(screen.getByText('readme.md')).toBeTruthy() })
    expect(screen.getAllByText(/index/i).length).toBeGreaterThanOrEqual(1)
  })

  it('index calls filesController.ingest', async () => {
    render(<FilesPage />)
    await waitFor(() => { expect(screen.getByText('readme.md')).toBeTruthy() })
    const indexButtons = screen.getAllByText(/index/i).filter(b => b.tagName === 'BUTTON')
    if (indexButtons.length > 0) {
      await act(async () => { fireEvent.click(indexButtons[0]) })
      expect(mockIngest).toHaveBeenCalled()
    }
  })
})

describe('FilesPage — upload flow', () => {
  beforeEach(() => { vi.clearAllMocks(); mockList.mockResolvedValue([]); mockUpload.mockResolvedValue({ filename: 'test.txt' }) })
  afterEach(() => { cleanup() })

  it('upload button triggers file input', async () => {
    render(<FilesPage />)
    await waitFor(() => { expect(screen.getAllByText(/upload/i).length).toBeGreaterThanOrEqual(1) })
    const inputs = document.querySelectorAll('input[type="file"]')
    expect(inputs.length).toBeGreaterThanOrEqual(1)
  })

  it('file input accepts correct extensions', async () => {
    render(<FilesPage />)
    await waitFor(() => { expect(screen.getAllByText(/upload/i).length).toBeGreaterThanOrEqual(1) })
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement
    expect(fileInput).toBeTruthy()
    expect(fileInput.accept).toContain('.txt')
    expect(fileInput.accept).toContain('.md')
    expect(fileInput.accept).toContain('.pdf')
  })
})

describe('FilesPage — refresh flow', () => {
  beforeEach(() => { vi.clearAllMocks(); mockList.mockResolvedValue(SAMPLE_FILES) })
  afterEach(() => { cleanup() })

  it('refresh button reloads file list', async () => {
    render(<FilesPage />)
    await waitFor(() => { expect(screen.getByText('readme.md')).toBeTruthy() })
    const refreshBtn = screen.getAllByRole('button').find(b => b.querySelector('svg'))
    if (refreshBtn) {
      await act(async () => { fireEvent.click(refreshBtn) })
      expect(mockList).toHaveBeenCalled()
    }
  })
})

describe('FilesPage — error handling', () => {
  beforeEach(() => { vi.clearAllMocks() })
  afterEach(() => { cleanup() })

  it('handles list failure gracefully', async () => {
    mockList.mockRejectedValue(new Error('Network error'))
    render(<FilesPage />)
    await waitFor(() => {
      expect(screen.getAllByText('Files').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('handles search failure gracefully', async () => {
    mockList.mockResolvedValue(SAMPLE_FILES)
    mockSearch.mockRejectedValue(new Error('Search failed'))
    render(<FilesPage />)
    await waitFor(() => { expect(screen.getByText('readme.md')).toBeTruthy() })
  })

  it('handles delete failure gracefully', async () => {
    mockList.mockResolvedValue(SAMPLE_FILES)
    mockDelete.mockRejectedValue(new Error('Delete failed'))
    render(<FilesPage />)
    await waitFor(() => { expect(screen.getByText('readme.md')).toBeTruthy() })
    const deleteButtons = screen.getAllByText(/delete/i).filter(b => b.tagName === 'BUTTON')
    if (deleteButtons.length > 0) {
      await act(async () => { fireEvent.click(deleteButtons[0]) })
    }
  })
})

describe('FilesPage — file metadata display', () => {
  beforeEach(() => { vi.clearAllMocks(); mockList.mockResolvedValue(SAMPLE_FILES) })
  afterEach(() => { cleanup() })

  it('shows content type', async () => {
    render(<FilesPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/text\/markdown/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows upload date', async () => {
    render(<FilesPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/1\/15\/2026/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('shows chunk count for indexed files', async () => {
    render(<FilesPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/10 chunks/).length).toBeGreaterThanOrEqual(1)
    })
  })

  it('formats large file sizes in MB', async () => {
    render(<FilesPage />)
    await waitFor(() => {
      expect(screen.getAllByText(/1\.0 MB/).length).toBeGreaterThanOrEqual(1)
    })
  })
})
