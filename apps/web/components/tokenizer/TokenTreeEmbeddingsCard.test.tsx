import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mocks = vi.hoisted(() => ({
  mockGetEmbedding: vi.fn(),
}))

vi.mock('@/lib/token-tree-controller', () => ({
  tokenTreeController: {
    getEmbedding: mocks.mockGetEmbedding,
  },
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Input: (props: any) => <input {...props} />,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Chip: ({ label }: { label: string }) => <span>{label}</span>,
}))

import { TokenTreeEmbeddingsCard } from './TokenTreeEmbeddingsCard'

const EMBED = {
  token: 'the',
  id: 3,
  dim: 8,
  norm: 1,
  top: [
    [0, 0.9],
    [1, -0.8],
  ],
  embedding_points: 200,
  compression_ratio: 4,
}

describe('TokenTreeEmbeddingsCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(cleanup)

  it('renders the token input with a default', () => {
    render(<TokenTreeEmbeddingsCard />)
    expect(screen.getByLabelText('Token to inspect')).toBeDefined()
    expect((screen.getByLabelText('Token to inspect') as HTMLInputElement).value).toBe('quick')
  })

  it('inspects a token and shows the embedding summary', async () => {
    mocks.mockGetEmbedding.mockResolvedValue(EMBED)
    render(<TokenTreeEmbeddingsCard />)

    fireEvent.change(screen.getByLabelText('Token to inspect'), { target: { value: 'the' } })
    fireEvent.click(screen.getByRole('button', { name: /^Inspect$/ }))
    await waitFor(() => expect(mocks.mockGetEmbedding).toHaveBeenCalledWith('the', 8))

    expect(screen.getByText(/"the" · id 3/)).toBeDefined()
    expect(screen.getByText('Dim 8')).toBeDefined()
    expect(screen.getByText('L2 norm 1.0000')).toBeDefined()
    expect(screen.getByText('200 points')).toBeDefined()
    expect(screen.getByText('4x compressed')).toBeDefined()
  })

  it('disables the button for a blank token', () => {
    render(<TokenTreeEmbeddingsCard />)
    fireEvent.change(screen.getByLabelText('Token to inspect'), { target: { value: '   ' } })
    expect((screen.getByRole('button', { name: /^Inspect$/ }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('inspects on Enter key', async () => {
    mocks.mockGetEmbedding.mockResolvedValue(EMBED)
    render(<TokenTreeEmbeddingsCard />)
    fireEvent.change(screen.getByLabelText('Token to inspect'), { target: { value: 'the' } })
    fireEvent.keyDown(screen.getByLabelText('Token to inspect'), { key: 'Enter' })
    await waitFor(() => expect(mocks.mockGetEmbedding).toHaveBeenCalledWith('the', 8))
  })

  it('shows an error box when the token is unknown or embeddings are disabled', async () => {
    mocks.mockGetEmbedding.mockRejectedValue(new Error('Token not in vocabulary'))
    render(<TokenTreeEmbeddingsCard />)
    fireEvent.click(screen.getByRole('button', { name: /^Inspect$/ }))
    await waitFor(() =>
      expect(screen.getByText(/Token not in the vocabulary, or embeddings are disabled/)).toBeDefined(),
    )
  })

  it('shows top dimension values with sign', async () => {
    mocks.mockGetEmbedding.mockResolvedValue(EMBED)
    render(<TokenTreeEmbeddingsCard />)
    fireEvent.click(screen.getByRole('button', { name: /^Inspect$/ }))
    await waitFor(() => expect(screen.getByText('+0.9000')).toBeDefined())
    expect(screen.getByText('-0.8000')).toBeDefined()
  })

  it('shows inspecting... text while loading', async () => {
    let resolvePromise: any
    mocks.mockGetEmbedding.mockImplementation(() => new Promise(r => { resolvePromise = r }))
    render(<TokenTreeEmbeddingsCard />)
    fireEvent.click(screen.getByRole('button', { name: /^Inspect$/ }))
    expect(screen.getByText('Inspecting...')).toBeDefined()
    resolvePromise(EMBED)
    await waitFor(() => expect(screen.getByText('Inspect')).toBeDefined())
  })

  it('allows changing the token value', async () => {
    mocks.mockGetEmbedding.mockResolvedValue(EMBED)
    render(<TokenTreeEmbeddingsCard />)
    const input = screen.getByLabelText('Token to inspect')
    fireEvent.change(input, { target: { value: 'hello' } })
    expect((input as HTMLInputElement).value).toBe('hello')
  })

  it('renders CardTitle', () => {
    render(<TokenTreeEmbeddingsCard />)
    expect(screen.getByText('Token Embedding Explorer')).toBeDefined()
  })
})
