/// <reference types="vitest" />
import { render, screen, cleanup } from '@testing-library/react'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { TokenTreeStatsCard, type TokenTreeStats } from './TokenTreeStatsCard'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => <h3 data-testid="card-title" {...props}>{children}</h3>,
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-content" {...props}>{children}</div>,
}))

afterEach(() => cleanup())

const mockStats: TokenTreeStats = {
  trained: true,
  vocab_size: 1024,
  num_merges: 256,
  embedding_points: 512,
  num_base_tokens: 256,
  embedding_compression_ratio: 1.5,
  embed_dim: 64,
}

describe('TokenTreeStatsCard', () => {
  it('renders the card title', () => {
    render(<TokenTreeStatsCard stats={mockStats} />)
    expect(screen.getByText('Token Tree Stats')).toBeDefined()
  })

  it('renders vocab size', () => {
    render(<TokenTreeStatsCard stats={mockStats} />)
    expect(screen.getByText('1,024')).toBeDefined()
  })

  it('renders trained status as Yes when true', () => {
    render(<TokenTreeStatsCard stats={mockStats} />)
    expect(screen.getByText('Yes')).toBeDefined()
  })

  it('renders trained status as No when false', () => {
    render(<TokenTreeStatsCard stats={{ ...mockStats, trained: false }} />)
    expect(screen.getByText('No')).toBeDefined()
  })

  it('renders all stat labels', () => {
    render(<TokenTreeStatsCard stats={mockStats} />)
    expect(screen.getByText('Merges')).toBeDefined()
    expect(screen.getByText('Embeddings')).toBeDefined()
    expect(screen.getByText('Compression')).toBeDefined()
    expect(screen.getByText('Embed Dim')).toBeDefined()
  })

  it('renders compression ratio with two decimals', () => {
    render(<TokenTreeStatsCard stats={mockStats} />)
    expect(screen.getByText('1.50')).toBeDefined()
  })
})
