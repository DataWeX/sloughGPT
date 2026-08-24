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
const mockAddToast = vi.fn()

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
  useToastStore: () => mockAddToast,
}))

vi.mock('@sloughgpt/strui', () => {
  const passthrough = ({ children }: any) => <div>{children}</div>
  return {
    cn: (...a: any[]) => a.join(' '),
    Button: ({ children, onClick, disabled, variant, className }: any) => (
      <button onClick={onClick} disabled={disabled} data-variant={variant} className={className}>{children}</button>
    ),
    Card: passthrough, CardContent: passthrough, CardHeader: passthrough,
    CardTitle: ({ children }: any) => <div>{children}</div>,
    Input: ({ value, onChange, placeholder, className, type, onKeyDown }: any) => (
      <input value={value} onChange={onChange} placeholder={placeholder} className={className} type={type} onKeyDown={onKeyDown} />
    ),
    Label: ({ children, className }: any) => <label className={className}>{children}</label>,
  }
})

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ children, title, subtitle, headerRight }: any) => (
    <div data-testid="page-container"><h1>{title}</h1><p>{subtitle}</p><div>{headerRight}</div>{children}</div>
  ),
}))

import TokenTreePage from './page'

describe('TokenTreePage', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders title and subtitle', async () => {
    mockGetStats.mockResolvedValue({ trained: false, vocab_size: 0, num_merges: 0, embedding_points: 0, num_base_tokens: 0, embedding_compression_ratio: 0, embed_dim: 0 })
    render(<TokenTreePage />)
    expect(screen.getByText('Token Tree')).toBeInTheDocument()
    expect(screen.getByText('BPE merge tree with learned embeddings')).toBeInTheDocument()
  })

  it('fetches stats on mount', async () => {
    mockGetStats.mockResolvedValue({ trained: true, vocab_size: 256, num_merges: 100, embedding_points: 50, num_base_tokens: 256, embedding_compression_ratio: 1.5, embed_dim: 64 })
    render(<TokenTreePage />)
    await waitFor(() => {
      expect(mockGetStats).toHaveBeenCalled()
    }, { timeout: 5000 })
  })

  it('displays stats', async () => {
    mockGetStats.mockResolvedValue({ trained: true, vocab_size: 256, num_merges: 100, embedding_points: 50, num_base_tokens: 128, embedding_compression_ratio: 1.5, embed_dim: 64 })
    render(<TokenTreePage />)
    await waitFor(() => {
      expect(screen.getByText('Yes')).toBeInTheDocument()
    }, { timeout: 5000 })
    expect(screen.getByText('256')).toBeInTheDocument()
    expect(screen.getByText('100')).toBeInTheDocument()
  })

  it('renders tab buttons', async () => {
    mockGetStats.mockResolvedValue({ trained: false, vocab_size: 0, num_merges: 0, embedding_points: 0, num_base_tokens: 0, embedding_compression_ratio: 0, embed_dim: 0 })
    render(<TokenTreePage />)
    expect(screen.getByText('Stats')).toBeInTheDocument()
    expect(screen.getByText('Vocabulary')).toBeInTheDocument()
    expect(screen.getByText('Merges')).toBeInTheDocument()
    expect(screen.getByText('Similar')).toBeInTheDocument()
    expect(screen.getByText('Embedding')).toBeInTheDocument()
    expect(screen.getByText('Saved')).toBeInTheDocument()
    expect(screen.getByText('Matrix')).toBeInTheDocument()
    expect(screen.getByText('Compare')).toBeInTheDocument()
  })

  it('trains token tree', async () => {
    mockGetStats.mockResolvedValue({ trained: false, vocab_size: 0, num_merges: 0, embedding_points: 0, num_base_tokens: 0, embedding_compression_ratio: 0, embed_dim: 0 })
    mockTrain.mockResolvedValue({ vocab_size: 128, embedding_points: 20 })
    render(<TokenTreePage />)
    fireEvent.click(screen.getByText('Train Token Tree'))
    await waitFor(() => {
      expect(mockTrain).toHaveBeenCalled()
    }, { timeout: 5000 })
    expect(mockAddToast).toHaveBeenCalledWith('Trained — 128 vocab, 20 embeddings', 'success')
  })

  it('loads vocab when Vocabulary tab clicked', async () => {
    mockGetStats.mockResolvedValue({ trained: false, vocab_size: 0, num_merges: 0, embedding_points: 0, num_base_tokens: 0, embedding_compression_ratio: 0, embed_dim: 0 })
    mockGetVocab.mockResolvedValue({ entries: [{ id: 0, token: 'a', freq: 100, is_special: false, is_merged: false }], total: 1 })
    render(<TokenTreePage />)
    fireEvent.click(screen.getByText('Vocabulary'))
    await waitFor(() => {
      expect(mockGetVocab).toHaveBeenCalledWith(50, 0)
    }, { timeout: 5000 })
  })

  it('loads merges when Merges tab clicked', async () => {
    mockGetStats.mockResolvedValue({ trained: false, vocab_size: 0, num_merges: 0, embedding_points: 0, num_base_tokens: 0, embedding_compression_ratio: 0, embed_dim: 0 })
    mockGetMerges.mockResolvedValue([{ rank: 1, left: 'a', right: 'b', token: 'ab', count: 50 }])
    render(<TokenTreePage />)
    fireEvent.click(screen.getByText('Merges'))
    await waitFor(() => {
      expect(mockGetMerges).toHaveBeenCalledWith(30, '')
    }, { timeout: 5000 })
  })

  it('loads saved trees when Saved tab clicked', async () => {
    mockGetStats.mockResolvedValue({ trained: false, vocab_size: 0, num_merges: 0, embedding_points: 0, num_base_tokens: 0, embedding_compression_ratio: 0, embed_dim: 0 })
    mockListSaved.mockResolvedValue([{ name: 'v1', vocab_size: 128, num_merges: 50, saved_at: '2024-01-01' }])
    render(<TokenTreePage />)
    fireEvent.click(screen.getByText('Saved'))
    await waitFor(() => {
      expect(mockListSaved).toHaveBeenCalled()
    }, { timeout: 5000 })
  })

  it('shows empty saved state', async () => {
    mockGetStats.mockResolvedValue({ trained: false, vocab_size: 0, num_merges: 0, embedding_points: 0, num_base_tokens: 0, embedding_compression_ratio: 0, embed_dim: 0 })
    mockListSaved.mockResolvedValue([])
    render(<TokenTreePage />)
    fireEvent.click(screen.getByText('Saved'))
    await waitFor(() => {
      expect(screen.getByText('No saved trees.')).toBeInTheDocument()
    }, { timeout: 5000 })
  })

  it('finds similar tokens', async () => {
    mockGetStats.mockResolvedValue({ trained: false, vocab_size: 0, num_merges: 0, embedding_points: 0, num_base_tokens: 0, embedding_compression_ratio: 0, embed_dim: 0 })
    mockSimilar.mockResolvedValue({ query: 'test', neighbors: [{ id: 1, token: 'test', score: 0.99 }] })
    render(<TokenTreePage />)
    fireEvent.click(screen.getByText('Similar'))
    fireEvent.change(screen.getByPlaceholderText('Enter a token...'), { target: { value: 'test' } })
    fireEvent.click(screen.getByText('Find'))
    await waitFor(() => {
      expect(mockSimilar).toHaveBeenCalledWith('test', 10)
    }, { timeout: 5000 })
  })

  it('gets embedding', async () => {
    mockGetStats.mockResolvedValue({ trained: false, vocab_size: 0, num_merges: 0, embedding_points: 0, num_base_tokens: 0, embedding_compression_ratio: 0, embed_dim: 0 })
    mockGetEmbedding.mockResolvedValue({ token: 'hello', dim: 64, norm: 1.0, top: [[0.5, 0], [0.3, 1]], embedding_points: 10, compression_ratio: 1.2 })
    render(<TokenTreePage />)
    fireEvent.click(screen.getByText('Embedding'))
    fireEvent.change(screen.getByPlaceholderText('Enter a token...'), { target: { value: 'hello' } })
    fireEvent.click(screen.getByText('Get Embedding'))
    await waitFor(() => {
      expect(mockGetEmbedding).toHaveBeenCalledWith('hello', 8)
    }, { timeout: 5000 })
  })

  it('compares tokens', async () => {
    mockGetStats.mockResolvedValue({ trained: false, vocab_size: 0, num_merges: 0, embedding_points: 0, num_base_tokens: 0, embedding_compression_ratio: 0, embed_dim: 0 })
    mockCompare.mockResolvedValue({ a: { name: 'tree', stats: {} }, b: { name: 'forest', stats: {} }, shared_tokens: 5, only_a_tokens: 2, only_b_tokens: 3, shared_merges: 4, only_a_merges: 1, only_b_merges: 2, shared_examples: [['ab', 10]], only_a_examples: [['cd', 5]], only_b_examples: [['ef', 3]] })
    render(<TokenTreePage />)
    fireEvent.click(screen.getByText('Compare'))
    const inputs = screen.getAllByRole('textbox')
    fireEvent.change(inputs[0], { target: { value: 'tree' } })
    fireEvent.change(inputs[1], { target: { value: 'forest' } })
    fireEvent.click(screen.getByText('Compare', { selector: 'button' }))
    await waitFor(() => {
      expect(mockCompare).toHaveBeenCalledWith('tree', 'forest', 10)
    }, { timeout: 5000 })
  })

  it('refreshes on Refresh click', async () => {
    mockGetStats.mockResolvedValue({ trained: false, vocab_size: 0, num_merges: 0, embedding_points: 0, num_base_tokens: 0, embedding_compression_ratio: 0, embed_dim: 0 })
    render(<TokenTreePage />)
    await waitFor(() => {
      expect(mockGetStats).toHaveBeenCalledTimes(1)
    }, { timeout: 5000 })
    fireEvent.click(screen.getByText('Refresh'))
    await waitFor(() => {
      expect(mockGetStats).toHaveBeenCalledTimes(2)
    }, { timeout: 5000 })
  })
})
