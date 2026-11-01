// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'

const mockIngestFile = vi.fn()
const mockPush = vi.fn()

vi.mock('@/lib/kb-controller', () => ({
  kbController: {
    ingestFile: (...args: unknown[]) => mockIngestFile(...args),
  },
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: mockPush }),
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, className, onDrop, onDragOver, onDragLeave }: any) => (
    <div data-testid="card" className={className} onDrop={onDrop} onDragOver={onDragOver} onDragLeave={onDragLeave}>
      {children}
    </div>
  ),
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, ...p }: any) => (
    <button onClick={onClick} disabled={disabled} {...p}>
      {children}
    </button>
  ),
  Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
  IconUpload: () => <span data-testid="icon-upload">upload</span>,
}))

import { ReadMyFilesCard } from './ReadMyFilesCard'

function renderCard(props: Partial<Parameters<typeof ReadMyFilesCard>[0]> = {}) {
  const addToast = props.addToast ?? vi.fn()
  const onIngested = props.onIngested ?? vi.fn()
  render(<ReadMyFilesCard addToast={addToast} onIngested={onIngested} />)
  return { addToast, onIngested }
}

function uploadFile(file: File) {
  const input = screen.getByLabelText('Choose a file to read') as HTMLInputElement
  fireEvent.change(input, { target: { files: [file] } })
}

describe('ReadMyFilesCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => cleanup())

  it('renders the drop zone copy from the Read My Files journey', () => {
    renderCard()
    expect(screen.getByText('Read my files')).toBeTruthy()
    expect(screen.getByText('Drop a file here or click to upload')).toBeTruthy()
    expect(screen.getByLabelText('Upload a file to read')).toBeTruthy()
  })

  it('ingests a file, reports the result, and shows suggested questions', async () => {
    mockIngestFile.mockResolvedValue({
      status: 'imported',
      stored: 2,
      total_chunks: 2,
      topic: 'imported',
      filename: 'notes.txt',
      file_size: 11,
    })
    const { addToast, onIngested } = renderCard()
    uploadFile(new File(['hello world'], 'notes.txt', { type: 'text/plain' }))

    await waitFor(() =>
      expect(screen.getByText(/Got it — I read 2 pieces from/)).toBeTruthy(),
    )
    expect(screen.getByText('notes.txt')).toBeTruthy()
    expect(screen.getByText('Summarize this')).toBeTruthy()
    expect(screen.getByText('What are the key points?')).toBeTruthy()
    expect(screen.getByText('Explain this in simple terms')).toBeTruthy()
    expect(mockIngestFile).toHaveBeenCalledTimes(1)
    expect(onIngested).toHaveBeenCalledTimes(1)
    expect(addToast).toHaveBeenCalledWith('Read notes.txt', 'success')
  })

  it('shows the reading state while ingesting', async () => {
    let resolveIngest!: (v: unknown) => void
    mockIngestFile.mockReturnValue(new Promise((r) => (resolveIngest = r)))
    renderCard()
    uploadFile(new File(['slow'], 'slow.txt', { type: 'text/plain' }))
    await waitFor(() => expect(screen.getByText('Reading your file...')).toBeTruthy())
    resolveIngest({
      status: 'imported',
      stored: 1,
      total_chunks: 1,
      topic: 'imported',
      filename: 'slow.txt',
      file_size: 4,
    })
    await waitFor(() => expect(screen.getByText(/Got it/)).toBeTruthy())
  })

  it('navigates to chat with the question prefilled when a chip is clicked', async () => {
    mockIngestFile.mockResolvedValue({
      status: 'imported',
      stored: 1,
      total_chunks: 1,
      topic: 'imported',
      filename: 'doc.pdf',
      file_size: 4,
    })
    renderCard()
    uploadFile(new File(['pdf-ish'], 'doc.pdf', { type: 'application/pdf' }))
    await waitFor(() => expect(screen.getByText('Summarize this')).toBeTruthy())
    fireEvent.click(screen.getByText('Summarize this'))
    expect(mockPush).toHaveBeenCalledWith('/chat?q=Summarize%20this')
  })

  it('rejects Word files with a friendly message without calling the API', () => {
    const { addToast } = renderCard()
    uploadFile(new File(['PK'], 'report.docx'))
    expect(mockIngestFile).not.toHaveBeenCalled()
    expect(addToast).toHaveBeenCalledWith(
      "Word files aren't supported yet — upload a PDF or a text file",
      'error',
    )
  })

  it('surfaces ingest failures as an error toast', async () => {
    mockIngestFile.mockRejectedValue(new Error('No readable text found in that PDF'))
    const { addToast } = renderCard()
    uploadFile(new File(['%PDF'], 'bad.pdf', { type: 'application/pdf' }))
    await waitFor(() =>
      expect(addToast).toHaveBeenCalledWith('No readable text found in that PDF', 'error'),
    )
    expect(screen.queryByText(/Got it/)).toBeNull()
  })
})
