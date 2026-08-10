/** @vitest-environment jsdom */
import { render, screen, fireEvent } from '@testing-library/react'
import { vi, describe, it, expect } from 'vitest'
import { TrainingSearchBar, useTrainingSearch } from './TrainingSearch'
import type { Checkpoint } from '@/lib/souls-controller'

function TestHook({ checkpoints }: { checkpoints: Checkpoint[] }) {
  const { filtered, query, setQuery } = useTrainingSearch(checkpoints)
  return (
    <div>
      <input data-testid="query" value={query} onChange={e => setQuery(e.target.value)} />
      <div data-testid="count">{filtered.length}</div>
      {filtered.map(c => (
        <div key={c.name} data-testid={`item-${c.name}`}>{c.name}</div>
      ))}
    </div>
  )
}

const fixtures: Checkpoint[] = [
  { name: 'alpha', soul: 'sage', model_type: 'llama', loss: 1.2 },
  { name: 'beta', soul: 'oracle', model_type: 'mistral', loss: 2.5 },
  { name: 'gamma', soul: 'sage', model_type: 'llama', tags: ['production'], lineage: 'alpha', training_dataset: 'wiki', description: 'main model', tagline: 'fast and accurate' },
]

function getCount() {
  return parseInt(screen.getAllByTestId('count')[0].textContent || '0', 10)
}

function hasItem(testId: string) {
  return screen.queryAllByTestId(testId).length > 0
}

function typeQuery(value: string) {
  fireEvent.change(screen.getAllByTestId('query')[0], { target: { value } })
}

describe('useTrainingSearch', () => {
  it('returns all checkpoints when query is empty', () => {
    render(<TestHook checkpoints={fixtures} />)
    expect(getCount()).toBe(3)
  })

  it('filters by name', () => {
    render(<TestHook checkpoints={fixtures} />)
    typeQuery('beta')
    expect(getCount()).toBe(1)
    expect(hasItem('item-beta')).toBe(true)
  })

  it('filters by soul', () => {
    render(<TestHook checkpoints={fixtures} />)
    typeQuery('oracle')
    expect(getCount()).toBe(1)
    expect(hasItem('item-beta')).toBe(true)
  })

  it('filters by model_type', () => {
    render(<TestHook checkpoints={fixtures} />)
    typeQuery('mistral')
    expect(getCount()).toBe(1)
    expect(hasItem('item-beta')).toBe(true)
  })

  it('filters by lineage', () => {
    render(<TestHook checkpoints={fixtures} />)
    typeQuery('gamma')
    expect(getCount()).toBe(1)
    expect(hasItem('item-gamma')).toBe(true)
  })

  it('filters by training_dataset', () => {
    render(<TestHook checkpoints={fixtures} />)
    typeQuery('wiki')
    expect(getCount()).toBe(1)
    expect(hasItem('item-gamma')).toBe(true)
  })

  it('filters by description', () => {
    render(<TestHook checkpoints={fixtures} />)
    typeQuery('main model')
    expect(getCount()).toBe(1)
    expect(hasItem('item-gamma')).toBe(true)
  })

  it('filters by tagline', () => {
    render(<TestHook checkpoints={fixtures} />)
    typeQuery('fast and accurate')
    expect(getCount()).toBe(1)
    expect(hasItem('item-gamma')).toBe(true)
  })

  it('filters by tags', () => {
    render(<TestHook checkpoints={fixtures} />)
    typeQuery('production')
    expect(getCount()).toBe(1)
    expect(hasItem('item-gamma')).toBe(true)
  })

  it('filters by loss value', () => {
    render(<TestHook checkpoints={fixtures} />)
    typeQuery('2.5')
    expect(getCount()).toBe(1)
    expect(hasItem('item-beta')).toBe(true)
  })

  it('is case-insensitive', () => {
    render(<TestHook checkpoints={fixtures} />)
    typeQuery('BETA')
    expect(getCount()).toBe(1)
    expect(hasItem('item-beta')).toBe(true)
  })
})

describe('TrainingSearchBar', () => {
  it('does not render when total < 3', () => {
    render(<TrainingSearchBar query="" onQueryChange={() => {}} total={2} shown={2} />)
    expect(screen.queryByText('Search')).toBeNull()
  })

  it('renders when total >= 3', () => {
    render(<TrainingSearchBar query="" onQueryChange={() => {}} total={5} shown={5} />)
    expect(screen.getAllByText('Search').length).toBeGreaterThanOrEqual(1)
  })

  it('shows match count when query is set', () => {
    render(<TrainingSearchBar query="llama" onQueryChange={() => {}} total={5} shown={2} />)
    expect(screen.getAllByText('2/5 match').length).toBeGreaterThanOrEqual(1)
  })

  it('does not show match count when query is empty', () => {
    render(<TrainingSearchBar query="" onQueryChange={() => {}} total={5} shown={5} />)
    expect(screen.queryByText('/5 match')).toBeNull()
  })

  it('renders search input', () => {
    render(<TrainingSearchBar query="" onQueryChange={() => {}} total={5} shown={5} />)
    const inputs = screen.getAllByPlaceholderText('Search checkpoints by name, type, dataset...')
    expect(inputs.length).toBeGreaterThanOrEqual(1)
  })
})
