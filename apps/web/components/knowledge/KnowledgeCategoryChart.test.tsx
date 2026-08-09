// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { KnowledgeCategoryChart } from './KnowledgeCategoryChart'
import type { KnowledgeItem, KnowledgeStats } from '@/lib/knowledge-controller'

afterEach(() => { cleanup() })

const items: KnowledgeItem[] = [
  { id: '1', content: 'User prefers dark mode', topic: 'preferences', source: 'manual', url: '', timestamp: 1, importance: 0.9, score: 0.8 },
  { id: '2', content: 'User works at Acme', topic: 'personal', source: 'manual', url: '', timestamp: 1, importance: 0.7, score: 0.6 },
  { id: '3', content: 'User likes TypeScript', topic: 'technical', source: 'ingest', url: '', timestamp: 1, importance: 0.8, score: 0.7 },
  { id: '4', content: 'User has a dog', topic: 'personal', source: 'manual', url: '', timestamp: 1, importance: 0.5, score: 0.4 },
  { id: '5', content: 'User uses React', topic: 'technical', source: 'ingest', url: '', timestamp: 1, importance: 0.6, score: 0.5 },
]

const stats: KnowledgeStats = {
  total_items: 5,
  topics: { preferences: 1, personal: 2, technical: 2 },
  topic_count: 3,
  sources: { manual: 3, ingest: 2 },
  avg_importance: 0.7,
  searchable: true,
}

describe('KnowledgeCategoryChart', () => {
  it('returns null for empty items', () => {
    const { container } = render(<KnowledgeCategoryChart items={[]} stats={null} />)
    expect(container.querySelector('[data-testid="knowledge-category-chart"]')).toBeNull()
  })

  it('renders chart card', () => {
    render(<KnowledgeCategoryChart items={items} stats={stats} />)
    expect(screen.getAllByTestId('knowledge-category-chart').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Knowledge Breakdown').length).toBeGreaterThanOrEqual(1)
  })

  it('shows topic labels', () => {
    render(<KnowledgeCategoryChart items={items} stats={stats} />)
    expect(screen.getAllByText('personal').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('technical').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('preferences').length).toBeGreaterThanOrEqual(1)
  })

  it('shows importance labels', () => {
    render(<KnowledgeCategoryChart items={items} stats={stats} />)
    expect(screen.getAllByText('Critical').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('High').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Medium').length).toBeGreaterThanOrEqual(1)
  })

  it('shows item count', () => {
    render(<KnowledgeCategoryChart items={items} stats={stats} />)
    expect(screen.getAllByText('5 items').length).toBeGreaterThanOrEqual(1)
  })

  it('shows topic count', () => {
    render(<KnowledgeCategoryChart items={items} stats={stats} />)
    expect(screen.getAllByText('3 topics').length).toBeGreaterThanOrEqual(1)
  })

  it('shows avg importance', () => {
    render(<KnowledgeCategoryChart items={items} stats={stats} />)
    expect(screen.getAllByText(/avg importance/).length).toBeGreaterThanOrEqual(1)
  })

  it('shows searchable badge when true', () => {
    render(<KnowledgeCategoryChart items={items} stats={stats} />)
    expect(screen.getAllByText('searchable').length).toBeGreaterThanOrEqual(1)
  })

  it('hides searchable badge when false', () => {
    const noSearch = { ...stats, searchable: false }
    render(<KnowledgeCategoryChart items={items} stats={noSearch} />)
    expect(screen.queryAllByText('searchable').length).toBe(0)
  })

  it('renders bar widths', () => {
    const { container } = render(<KnowledgeCategoryChart items={items} stats={stats} />)
    const bars = container.querySelectorAll('[style*="width"]')
    expect(bars.length).toBeGreaterThan(0)
  })
})
