import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

const mockList = vi.fn()
const mockUpload = vi.fn()
const mockDelete = vi.fn()
const mockIngest = vi.fn()
const mockSearch = vi.fn()

vi.mock('@/lib/files-controller', () => ({
  filesController: {
    list: (...args: unknown[]) => mockList(...args),
    upload: (...args: unknown[]) => mockUpload(...args),
    delete: (...args: unknown[]) => mockDelete(...args),
    ingest: (...args: unknown[]) => mockIngest(...args),
    search: (...args: unknown[]) => mockSearch(...args),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

import FilesPage from './page'

describe('FilesPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue([])
    mockSearch.mockResolvedValue([])
  })

  it('renders page header', async () => {
    render(<FilesPage />)
    expect(screen.getAllByText('Files').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no files', async () => {
    render(<FilesPage />)
    await screen.findAllByText(/no files|upload/i)
  })

  it('renders upload button', async () => {
    render(<FilesPage />)
    expect(screen.getAllByText(/upload/i).length).toBeGreaterThanOrEqual(1)
  })

  it('renders search input', async () => {
    render(<FilesPage />)
    expect(screen.getAllByPlaceholderText(/search/i).length).toBeGreaterThanOrEqual(1)
  })

  it('displays files when loaded', async () => {
    mockList.mockResolvedValue([
      { id: '1', filename: 'test.txt', size: 1024, content_type: 'text/plain', uploaded_at: '2026-01-01', ingested: true, chunk_count: 5 },
    ])
    render(<FilesPage />)
    await screen.findByText('test.txt')
    expect(screen.getByText(/indexed/)).toBeTruthy()
  })
})
