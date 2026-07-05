import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const { mockAddToast, mockList, mockUpload, mockDelete, mockSearch, mockIngest } = vi.hoisted(() => ({
  mockAddToast: vi.fn(),
  mockList: vi.fn(),
  mockUpload: vi.fn(),
  mockDelete: vi.fn(),
  mockSearch: vi.fn(),
  mockIngest: vi.fn(),
}))

const mockFiles = {
  files: [
    { id: 'f1', filename: 'notes.txt', extension: 'txt', size_bytes: 1024, uploaded_at: '2026-06-01T00:00:00Z' },
    { id: 'f2', filename: 'data.csv', extension: 'csv', size_bytes: 20480, uploaded_at: '2026-06-15T00:00:00Z' },
  ],
  total: 2,
}

const mockUploadResult = { filename: 'new.txt', size_bytes: 512 }

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
  useParams: () => ({}),
}))

vi.mock('@/lib/controllers', () => ({
  filesController: {
    list: mockList,
    upload: mockUpload,
    delete: mockDelete,
    search: mockSearch,
    ingest: mockIngest,
    formatSize: (b: number) => b >= 1024 ? `${(b/1024).toFixed(1)} KB` : `${b} B`,
    formatDate: (d: string) => d,
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel: any) => sel ? sel({ addToast: mockAddToast }) : { addToast: mockAddToast },
}))

import FilesPage from './page'

beforeEach(() => {
  vi.clearAllMocks()
  mockList.mockResolvedValue(mockFiles)
  mockUpload.mockResolvedValue(mockUploadResult)
  mockDelete.mockResolvedValue(undefined)
  mockSearch.mockResolvedValue({ files: [], total: 0 })
})

afterEach(cleanup)

describe('FilesPage', () => {
  it('renders header and upload area', async () => {
    render(<FilesPage />)
    await waitFor(() => expect(screen.getByText('Files')).toBeDefined())
    expect(screen.getByText('Upload')).toBeDefined()
    expect(screen.getByPlaceholderText('Search files...')).toBeDefined()
  })

  it('displays file list', async () => {
    render(<FilesPage />)
    await waitFor(() => expect(screen.getByText('notes.txt')).toBeDefined())
    expect(screen.getByText('data.csv')).toBeDefined()
  })

  it('shows upload modal via file input click', async () => {
    render(<FilesPage />)
    await waitFor(() => expect(screen.getByText('Upload')).toBeDefined())
    const file = new File(['hello'], 'new.txt', { type: 'text/plain' })
    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    expect(input).not.toBeNull()
    Object.defineProperty(input, 'files', { value: [file] })
    fireEvent.change(input)
    await waitFor(() => expect(mockUpload).toHaveBeenCalledWith(file))
  })

  it('shows drag overlay on dragEnter and hides on dragLeave', async () => {
    render(<FilesPage />)
    await waitFor(() => expect(screen.getByText('Upload')).toBeDefined())
    const dropZone = screen.getByTestId('drop-zone')
    fireEvent.dragEnter(dropZone)
    await waitFor(() => expect(screen.getByText('Drop file to upload')).toBeDefined())
    fireEvent.dragLeave(dropZone)
    await waitFor(() => expect(screen.queryByText('Drop file to upload')).toBeNull())
  })

  it('uploads file on drop', async () => {
    render(<FilesPage />)
    await waitFor(() => expect(screen.getByText('Upload')).toBeDefined())
    const dropZone = screen.getByTestId('drop-zone')
    const file = new File(['data'], 'dropped.csv', { type: 'text/csv' })
    fireEvent.drop(dropZone, { dataTransfer: { files: [file] } })
    await waitFor(() => expect(mockUpload).toHaveBeenCalledWith(file))
  })

  it('renders search input and filters files', async () => {
    mockSearch.mockResolvedValue({ files: [mockFiles.files[0]], total: 1 })
    render(<FilesPage />)
    await waitFor(() => expect(screen.getByText('notes.txt')).toBeDefined())
    const searchInput = screen.getByPlaceholderText('Search files...')
    fireEvent.change(searchInput, { target: { value: 'notes' } })
    await waitFor(() => {
      expect(mockSearch).toHaveBeenCalledWith('notes')
    })
  })
})
