import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { MemoryStatsGrid } from './MemoryStatsGrid'

afterEach(() => cleanup())

describe('MemoryStatsGrid', () => {
  it('shows loading skeletons', () => {
    const { container } = render(<MemoryStatsGrid stats={null} loading={true} />)
    expect(container.querySelectorAll('[class*="animate-pulse"]').length).toBeGreaterThanOrEqual(1)
  })
  it('shows stats when loaded', () => {
    render(<MemoryStatsGrid stats={{ total_facts: 42, topics: 5, visited_urls: 12, enabled: true }} loading={false} />)
    expect(screen.getAllByText('42').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('5').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('12').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Yes').length).toBeGreaterThanOrEqual(1)
  })
  it('shows defaults when stats null', () => {
    render(<MemoryStatsGrid stats={null} loading={false} />)
    expect(screen.getAllByText('0').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('No').length).toBeGreaterThanOrEqual(1)
  })
  it('shows labels', () => {
    render(<MemoryStatsGrid stats={null} loading={false} />)
    expect(screen.getAllByText('Facts').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Topics').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Visited URLs').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Searchable').length).toBeGreaterThanOrEqual(1)
  })
  it('shows No when enabled is false', () => {
    render(<MemoryStatsGrid stats={{ total_facts: 0, topics: 0, visited_urls: 0, enabled: false }} loading={false} />)
    expect(screen.getAllByText('No').length).toBeGreaterThanOrEqual(1)
  })
})
