import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { KnowledgeCard } from './KnowledgeCard'

const knowledgeStats = { total_items: 42, topic_count: 7, avg_importance: 0.63, searchable: true }
const adapterStatus = { adapter_exists: true, fact_count: 12, total_facts_available: 40 }

function renderCard(props: Partial<Parameters<typeof KnowledgeCard>[0]> = {}) {
  const base = {
    knowledgeStats: null,
    adapterStatus: null,
    loaded: false,
  }
  return render(<KnowledgeCard {...base} {...props} />)
}

describe('KnowledgeCard', () => {
  afterEach(cleanup)

  it('shows skeleton placeholders when nothing is loaded', () => {
    renderCard()
    expect(screen.getByText('Knowledge')).toBeDefined()
    const skeletons = document.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBeGreaterThanOrEqual(4)
  })

  it('shows zero values once loaded without stats', () => {
    renderCard({ loaded: true })
    expect(screen.getAllByText('0').length).toBe(2)
  })

  it('renders items and topics from stats', () => {
    renderCard({ knowledgeStats })
    expect(screen.getByText('42')).toBeDefined()
    expect(screen.getByText('7')).toBeDefined()
  })

  it('renders importance with two decimals', () => {
    renderCard({ knowledgeStats })
    expect(screen.getByText('0.63')).toBeDefined()
  })

  it('renders "Trained" when adapter exists', () => {
    renderCard({ knowledgeStats, adapterStatus })
    expect(screen.getByText('Trained')).toBeDefined()
  })

  it('renders "Not" when adapter is missing', () => {
    renderCard({ knowledgeStats, adapterStatus: { adapter_exists: false, fact_count: 0, total_facts_available: 0 } })
    expect(screen.getByText('Not')).toBeDefined()
  })

  it('renders fact count when adapter exists', () => {
    renderCard({ adapterStatus })
    expect(screen.getByText('12 facts (40 avail)')).toBeDefined()
  })

  it('does not render fact line when adapter is absent', () => {
    renderCard({ knowledgeStats, adapterStatus: { adapter_exists: false, fact_count: 0, total_facts_available: 0 } })
    expect(screen.queryByText(/facts/)).toBeNull()
  })
})
