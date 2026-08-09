// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { TokenizerEfficiencyCard } from './TokenizerEfficiencyCard'

afterEach(() => { cleanup() })

const stats = {
  vocab_size: 512,
  base_chars: 256,
  total_merges: 200,
  special_tokens: 4,
}

const samples = [
  { word: 'hello', tokens: ['hel', 'lo'], ids: [1, 2] },
  { word: 'world', tokens: ['wor', 'ld'], ids: [3, 4] },
  { word: 'the', tokens: ['the'], ids: [5] },
  { word: 'assistant', tokens: ['ass', 'ist', 'ant'], ids: [6, 7, 8] },
]

describe('TokenizerEfficiencyCard', () => {
  it('returns null for null stats', () => {
    const { container } = render(<TokenizerEfficiencyCard stats={null} samples={[]} />)
    expect(container.querySelector('[data-testid="tokenizer-efficiency"]')).toBeNull()
  })

  it('renders card with stats', () => {
    render(<TokenizerEfficiencyCard stats={stats} samples={samples} />)
    expect(screen.getAllByTestId('tokenizer-efficiency').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Tokenizer Efficiency').length).toBeGreaterThanOrEqual(1)
  })

  it('shows vocab utilization', () => {
    render(<TokenizerEfficiencyCard stats={stats} samples={samples} />)
    expect(screen.getAllByText('50%').length).toBeGreaterThanOrEqual(1)
  })

  it('shows compression ratio', () => {
    render(<TokenizerEfficiencyCard stats={stats} samples={samples} />)
    expect(screen.getAllByText(/x/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows avg tokens per word', () => {
    render(<TokenizerEfficiencyCard stats={stats} samples={samples} />)
    expect(screen.getAllByText(/Avg Tokens\/Word/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows merge efficiency', () => {
    render(<TokenizerEfficiencyCard stats={stats} samples={samples} />)
    expect(screen.getAllByText(/Merge Efficiency/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows special tokens count', () => {
    render(<TokenizerEfficiencyCard stats={stats} samples={samples} />)
    expect(screen.getAllByText(/4 special tokens/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows merges learned', () => {
    render(<TokenizerEfficiencyCard stats={stats} samples={samples} />)
    expect(screen.getAllByText(/200 merges learned/).length).toBeGreaterThanOrEqual(1)
  })

  it('handles empty samples', () => {
    render(<TokenizerEfficiencyCard stats={stats} samples={[]} />)
    expect(screen.getAllByTestId('tokenizer-efficiency').length).toBeGreaterThanOrEqual(1)
  })
})
