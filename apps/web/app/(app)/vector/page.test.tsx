import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'

const mockGetStats = vi.fn()
const mockInit = vi.fn()
const mockSearch = vi.fn()
const mockUpsert = vi.fn()
const mockIngestStatus = vi.fn()

vi.mock('@/lib/vector-controller', () => ({
  vectorController: {
    getStats: (...args: unknown[]) => mockGetStats(...args),
    init: (...args: unknown[]) => mockInit(...args),
    search: (...args: unknown[]) => mockSearch(...args),
    upsert: (...args: unknown[]) => mockUpsert(...args),
    ingestStatus: (...args: unknown[]) => mockIngestStatus(...args),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (...a: unknown[]) => void }) => unknown) => selector({ addToast: vi.fn() }),
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...a: any[]) => a.join(' '),
    Button: ({ children, onClick, disabled, variant }: any) => <button onClick={onClick} disabled={disabled} data-variant={variant}>{children}</button>,
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Input: ({ value, onChange, placeholder, onKeyDown }: any) => <input value={value} onChange={onChange} placeholder={placeholder} onKeyDown={onKeyDown} />,
    StatCard: ({ label, value }: any) => <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span></div>,
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    IconRefresh: () => <span data-testid="icon-refresh" />,
  }
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title, subtitle }: any) => (
    <div data-testid="page-container"><h1>{title}</h1><p>{subtitle}</p>{children}</div>
  ),
}))

import VectorPage from './page'

describe('VectorPage', () => {
  afterEach(() => cleanup())

  it('renders page title and subtitle', async () => {
    mockGetStats.mockResolvedValue({ provider: 'in_memory', count: 0 })
    render(<VectorPage />)
    expect(screen.getByText('Vector Store')).toBeInTheDocument()
    expect(screen.getByText('Manage embeddings and similarity search')).toBeInTheDocument()
  })

  it('fetches stats on mount and shows provider/count', async () => {
    mockGetStats.mockResolvedValue({ provider: 'in_memory', count: 42 })
    render(<VectorPage />)
    await waitFor(() => {
      expect(screen.getByText('42')).toBeInTheDocument()
    })
    expect(mockGetStats).toHaveBeenCalled()
  })

  it('shows Empty when count is 0', async () => {
    mockGetStats.mockResolvedValue({ provider: 'in_memory', count: 0 })
    render(<VectorPage />)
    await waitFor(() => {
      expect(screen.getByText('Empty')).toBeInTheDocument()
    })
  })

  it('shows Active when count > 0', async () => {
    mockGetStats.mockResolvedValue({ provider: 'chromadb', count: 10 })
    render(<VectorPage />)
    await waitFor(() => {
      expect(screen.getByText('Active')).toBeInTheDocument()
    })
  })

  it('initializes vector store when button clicked', async () => {
    mockGetStats.mockResolvedValue({ provider: 'in_memory', count: 0 })
    mockInit.mockResolvedValue({ status: 'ok', provider: 'chromadb' })
    render(<VectorPage />)
    await screen.findByText('Empty')

    fireEvent.click(screen.getByText('ChromaDB'))
    await waitFor(() => {
      expect(mockInit).toHaveBeenCalledWith('chromadb')
    })
  })

  it('searches vector store', async () => {
    mockGetStats.mockResolvedValue({ provider: 'in_memory', count: 5 })
    mockSearch.mockResolvedValue({ results: [{ text: 'found', score: 0.95, id: '1' }], elapsed_ms: 3 })
    render(<VectorPage />)
    await screen.findByText('Active')

    const input = screen.getByPlaceholderText('Search for similar text...')
    fireEvent.change(input, { target: { value: 'test query' } })
    fireEvent.click(screen.getByText('Search'))

    await waitFor(() => {
      expect(screen.getByText('found')).toBeInTheDocument()
    })
    expect(screen.getByText('95.0%')).toBeInTheDocument()
  })

  it('upserts text entries', async () => {
    mockGetStats.mockResolvedValue({ provider: 'in_memory', count: 0 })
    mockUpsert.mockResolvedValue({ status: 'ok', count: 2, elapsed_ms: 10 })
    render(<VectorPage />)
    await screen.findByText('Empty')

    const textarea = screen.getByPlaceholderText(/Enter text entries/)
    fireEvent.change(textarea, { target: { value: 'line 1\nline 2' } })
    fireEvent.click(screen.getByText('Add entries'))

    await waitFor(() => {
      expect(mockUpsert).toHaveBeenCalledWith(['line 1', 'line 2'])
    })
  })

  it('shows no results message when search returns empty', async () => {
    mockGetStats.mockResolvedValue({ provider: 'in_memory', count: 5 })
    mockSearch.mockResolvedValue({ results: [], elapsed_ms: 1 })
    render(<VectorPage />)
    await screen.findByText('Active')

    const input = screen.getByPlaceholderText('Search for similar text...')
    fireEvent.change(input, { target: { value: 'nothing' } })
    fireEvent.click(screen.getByText('Search'))

    await waitFor(() => {
      expect(screen.getByText('No results found')).toBeInTheDocument()
    })
  })
})
