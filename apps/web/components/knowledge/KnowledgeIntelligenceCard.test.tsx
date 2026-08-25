// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

import { KnowledgeIntelligenceCard } from './KnowledgeIntelligenceCard'

vi.mock('@/lib/knowledge-controller', () => ({
  knowledgeController: {
    label: vi.fn(),
    checkDuplicate: vi.fn(),
    categorize: vi.fn(),
    getEmbedderStatus: vi.fn(),
    trainEmbedder: vi.fn(),
    gaps: vi.fn(),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => ({ addToast: vi.fn() }),
}))

vi.mock('@/lib/error-utils', () => ({
  extractErrorMessage: (_e: unknown, fallback: string) => fallback,
}))

import { knowledgeController } from '@/lib/knowledge-controller'

describe('KnowledgeIntelligenceCard', () => {
  afterEach(cleanup)
  beforeEach(() => { vi.clearAllMocks() })

  it('renders all sections', () => {
    render(<KnowledgeIntelligenceCard />)
    expect(screen.getByText('Knowledge Intelligence')).toBeDefined()
    expect(screen.getByText('Auto-label')).toBeDefined()
    expect(screen.getByText('Duplicate check')).toBeDefined()
    expect(screen.getByText('Auto-categorize')).toBeDefined()
    expect(screen.getByText('Embedder')).toBeDefined()
    expect(screen.getByText('Analyze Knowledge Gaps')).toBeDefined()
  })

  it('auto-label shows result', async () => {
    vi.mocked(knowledgeController.label).mockResolvedValue({
      label: 'code', confidence: 0.95, reason: 'programming content', scores: { code: 0.95, science: 0.05 },
    })
    render(<KnowledgeIntelligenceCard />)
    const input = screen.getByPlaceholderText('Enter text to classify...')
    fireEvent.change(input, { target: { value: 'Python is a language' } })
    fireEvent.click(screen.getByText('Label'))
    await waitFor(() => {
      expect(screen.getByText('Label: code')).toBeDefined()
      expect(screen.getByText('(95%)')).toBeDefined()
    })
  })

  it('duplicate check shows duplicate found', async () => {
    vi.mocked(knowledgeController.checkDuplicate).mockResolvedValue({
      is_duplicate: true, best_match: 'Existing entry', score: 0.92, threshold: 0.85,
    })
    render(<KnowledgeIntelligenceCard />)
    const input = screen.getByPlaceholderText('Enter text to check for duplicates...')
    fireEvent.change(input, { target: { value: 'test content' } })
    fireEvent.click(screen.getByText('Check'))
    await waitFor(() => {
      expect(screen.getByText('Duplicate found')).toBeDefined()
    })
  })

  it('duplicate check shows no duplicates', async () => {
    vi.mocked(knowledgeController.checkDuplicate).mockResolvedValue({
      is_duplicate: false, best_match: null, score: 0.3, threshold: 0.85,
    })
    render(<KnowledgeIntelligenceCard />)
    const input = screen.getByPlaceholderText('Enter text to check for duplicates...')
    fireEvent.change(input, { target: { value: 'unique content' } })
    fireEvent.click(screen.getByText('Check'))
    await waitFor(() => {
      expect(screen.getByText('No duplicates found')).toBeDefined()
    })
  })

  it('categorize shows result', async () => {
    vi.mocked(knowledgeController.categorize).mockResolvedValue({
      topic: 'technology',
      suggestions: [{ topic: 'technology', score: 0.9 }, { topic: 'science', score: 0.1 }],
    })
    render(<KnowledgeIntelligenceCard />)
    const input = screen.getByPlaceholderText('Enter text to categorize...')
    fireEvent.change(input, { target: { value: 'machine learning' } })
    fireEvent.click(screen.getByText('Categorize'))
    await waitFor(() => {
      expect(screen.getByText('Topic: technology')).toBeDefined()
    })
  })

  it('embedder status shows trained state', async () => {
    vi.mocked(knowledgeController.getEmbedderStatus).mockResolvedValue({
      trained: true, info: { embed_dim: 128, vocab_size: 5000, path: '/tmp/embed' },
    })
    render(<KnowledgeIntelligenceCard />)
    fireEvent.click(screen.getAllByRole('button').find(b => b.getAttribute('aria-label') === 'Refresh embeddings') ?? document.querySelector('button[class*="ghost"]')!)
    await waitFor(() => {
      expect(screen.getByText('Trained')).toBeDefined()
    })
  })

  it('gaps analysis shows results', async () => {
    vi.mocked(knowledgeController.gaps).mockResolvedValue({
      gaps: [{ topic: 'math', suggestion: 'Add calculus basics' }],
      total_facts: 42,
      topics: ['code', 'science'],
    })
    render(<KnowledgeIntelligenceCard />)
    fireEvent.click(screen.getByText('Analyze Knowledge Gaps'))
    await waitFor(() => {
      expect(screen.getByText('Facts: 42')).toBeDefined()
      expect(screen.getByText('Topics: 2')).toBeDefined()
      expect(screen.getByText('Gaps: 1')).toBeDefined()
      expect(screen.getByText('Add calculus basics')).toBeDefined()
    })
  })

  it('disables buttons during loading', async () => {
    vi.mocked(knowledgeController.label).mockReturnValue(new Promise(() => {}))
    render(<KnowledgeIntelligenceCard />)
    const input = screen.getByPlaceholderText('Enter text to classify...')
    fireEvent.change(input, { target: { value: 'test' } })
    fireEvent.click(screen.getByText('Label'))
    await waitFor(() => {
      expect(screen.getByText('...')).toBeDefined()
    })
  })
})
