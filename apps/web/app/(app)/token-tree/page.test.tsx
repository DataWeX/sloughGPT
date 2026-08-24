import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import React from 'react'

const mockGetStats = vi.fn()
const mockTrain = vi.fn()
const mockGetVocab = vi.fn()
const mockGetMerges = vi.fn()
const mockListSaved = vi.fn()
const mockSaveTree = vi.fn()
const mockLoadTree = vi.fn()
const mockDeleteSavedTree = vi.fn()
const mockSimilar = vi.fn()
const mockGetEmbedding = vi.fn()
const mockGetMatrixSummary = vi.fn()
const mockCompare = vi.fn()

vi.mock('@/lib/token-tree-controller', () => ({
  tokenTreeController: {
    getStats: (...args: unknown[]) => mockGetStats(...args),
    train: (...args: unknown[]) => mockTrain(...args),
    getVocab: (...args: unknown[]) => mockGetVocab(...args),
    getMerges: (...args: unknown[]) => mockGetMerges(...args),
    listSaved: (...args: unknown[]) => mockListSaved(...args),
    saveTree: (...args: unknown[]) => mockSaveTree(...args),
    loadTree: (...args: unknown[]) => mockLoadTree(...args),
    deleteSavedTree: (...args: unknown[]) => mockDeleteSavedTree(...args),
    similar: (...args: unknown[]) => mockSimilar(...args),
    getEmbedding: (...args: unknown[]) => mockGetEmbedding(...args),
    getMatrixSummary: (...args: unknown[]) => mockGetMatrixSummary(...args),
    compare: (...args: unknown[]) => mockCompare(...args),
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
    CardDescription: ({ children }: any) => <p>{children}</p>,
    Input: ({ value, onChange, placeholder, onKeyDown, type }: any) => <input value={value} onChange={onChange} placeholder={placeholder} onKeyDown={onKeyDown} type={type} />,
    Label: ({ children }: any) => <label>{children}</label>,
    Textarea: ({ value, onChange, placeholder, rows }: any) => <textarea value={value} onChange={onChange} placeholder={placeholder} rows={rows} />,
    Skeleton: () => <div data-testid="skeleton" />,
    Badge: ({ children }: any) => <span>{children}</span>,
    StatCard: ({ label, value }: any) => <div data-testid={`stat-${label}`}><span>{label}</span><span>{String(value)}</span></div>,
    KpiGrid: ({ children }: any) => <div>{children}</div>,
    IconRefresh: () => <span data-testid="icon-refresh" />,
    IconTrash: () => <span data-testid="icon-trash" />,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
    AlertDialog: ({ open, children }: any) => open ? <div data-testid="alert-dialog">{children}</div> : null,
    AlertDialogAction: ({ children, onClick }: any) => <button onClick={onClick}>{children}</button>,
    AlertDialogCancel: ({ children }: any) => <button>{children}</button>,
    AlertDialogContent: ({ children }: any) => <div>{children}</div>,
    AlertDialogDescription: ({ children }: any) => <p>{children}</p>,
    AlertDialogFooter: ({ children }: any) => <div>{children}</div>,
    AlertDialogHeader: ({ children }: any) => <div>{children}</div>,
    AlertDialogTitle: ({ children }: any) => <div>{children}</div>,
  }
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title, subtitle, headerRight }: any) => (
    <div data-testid="page-container"><h1>{title}</h1><p>{subtitle}</p>{headerRight && <div>{headerRight}</div>}{children}</div>
  ),
}))

import TokenTreePage from './page'

describe('TokenTreePage', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders page title and subtitle', async () => {
    mockGetStats.mockResolvedValue({ trained: false, vocab_size: 0, num_merges: 0, embedding_points: 0, num_base_tokens: 0, embedding_compression_ratio: 0, embed_dim: 0 })
    render(<TokenTreePage />)
    expect(screen.getByText('Token Tree')).toBeInTheDocument()
    expect(screen.getByText('BPE merge tree with learned embeddings')).toBeInTheDocument()
  })

  it('trains tree', async () => {
    mockGetStats.mockResolvedValue({ trained: false, vocab_size: 0, num_merges: 0, embedding_points: 0, num_base_tokens: 0, embedding_compression_ratio: 0, embed_dim: 0 })
    mockTrain.mockResolvedValue({ vocab_size: 256, embedding_points: 128 })
    render(<TokenTreePage />)

    fireEvent.click(screen.getByText('Train Token Tree'))

    await waitFor(() => {
      expect(mockTrain).toHaveBeenCalled()
    })
  })

  it('loads vocab', async () => {
    mockGetStats.mockResolvedValue({ trained: true, vocab_size: 300, num_merges: 100, embedding_points: 50, num_base_tokens: 256, embedding_compression_ratio: 1.2, embed_dim: 8 })
    mockGetVocab.mockResolvedValue({ entries: [{ id: 0, token: 'alpha', freq: 100, is_special: false, is_merged: false }, { id: 1, token: 'beta', freq: 80, is_special: false, is_merged: true }], total: 300 })
    render(<TokenTreePage />)

    fireEvent.click(screen.getByText('Vocabulary'))

    await waitFor(() => {
      expect(mockGetVocab).toHaveBeenCalledWith(50, 0)
      expect(screen.getByText('alpha')).toBeInTheDocument()
      expect(screen.getByText('beta')).toBeInTheDocument()
    })
  })

  it('searches merges', async () => {
    mockGetStats.mockResolvedValue({ trained: true, vocab_size: 300, num_merges: 100, embedding_points: 50, num_base_tokens: 256, embedding_compression_ratio: 1.2, embed_dim: 8 })
    mockGetMerges.mockResolvedValue([{ rank: 1, left: 'x', right: 'y', token: 'xy', count: 50 }])
    render(<TokenTreePage />)

    fireEvent.click(screen.getByText('Merges'))

    await waitFor(() => {
      expect(mockGetMerges).toHaveBeenCalledWith(30, '')
      expect(screen.getByText('xy')).toBeInTheDocument()
    })
  })

  it('find similar', async () => {
    mockGetStats.mockResolvedValue({ trained: true, vocab_size: 300, num_merges: 100, embedding_points: 50, num_base_tokens: 256, embedding_compression_ratio: 1.2, embed_dim: 8 })
    mockSimilar.mockResolvedValue({ query: 'test', neighbors: [{ id: 1, token: 'test', score: 1.0 }, { id: 2, token: 'best', score: 0.85 }] })
    render(<TokenTreePage />)

    fireEvent.click(screen.getByText('Similar'))
    const input = screen.getByPlaceholderText('Enter a token...')
    fireEvent.change(input, { target: { value: 'test' } })
    fireEvent.click(screen.getByText('Find'))

    await waitFor(() => {
      expect(mockSimilar).toHaveBeenCalledWith('test', 10)
      expect(screen.getByText('test')).toBeInTheDocument()
      expect(screen.getByText('best')).toBeInTheDocument()
    })
  })

  it('get embedding', async () => {
    mockGetStats.mockResolvedValue({ trained: true, vocab_size: 300, num_merges: 100, embedding_points: 50, num_base_tokens: 256, embedding_compression_ratio: 1.2, embed_dim: 8 })
    mockGetEmbedding.mockResolvedValue({ token: 'hello', dim: 8, norm: 1.2345, top: [[0.5, 0], [0.3, 1], [0.2, 2]] })
    render(<TokenTreePage />)

    fireEvent.click(screen.getByText('Embedding'))
    const input = screen.getByPlaceholderText('Enter a token...')
    fireEvent.change(input, { target: { value: 'hello' } })
    fireEvent.click(screen.getByText('Get Embedding'))

    await waitFor(() => {
      expect(mockGetEmbedding).toHaveBeenCalledWith('hello', 8)
      expect(screen.getByText('hello')).toBeInTheDocument()
      expect(screen.getByText('1.2345')).toBeInTheDocument()
    })
  })

  it('compare tokens', async () => {
    mockGetStats.mockResolvedValue({ trained: true, vocab_size: 300, num_merges: 100, embedding_points: 50, num_base_tokens: 256, embedding_compression_ratio: 1.2, embed_dim: 8 })
    mockCompare.mockResolvedValue({
      a: { name: 'abc' },
      b: { name: 'def' },
      shared_tokens: 5,
      only_a_tokens: 3,
      only_b_tokens: 4,
      shared_merges: 2,
      only_a_merges: 1,
      only_b_merges: 2,
      shared_examples: [['x', 10]],
      only_a_examples: [['a', 5]],
      only_b_examples: [['d', 6]],
    })
    render(<TokenTreePage />)

    fireEvent.click(screen.getByText('Compare'))
    const inputA = screen.getAllByRole('textbox')[0]
    const inputB = screen.getAllByRole('textbox')[1]
    fireEvent.change(inputA, { target: { value: 'abc' } })
    fireEvent.change(inputB, { target: { value: 'def' } })
    fireEvent.click(screen.getByText('Compare'))

    await waitFor(() => {
      expect(mockCompare).toHaveBeenCalledWith('abc', 'def', 10)
      expect(screen.getByText('abc')).toBeInTheDocument()
      expect(screen.getByText('def')).toBeInTheDocument()
    })
  })
})