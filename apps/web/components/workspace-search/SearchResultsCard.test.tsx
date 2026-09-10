/// <reference types="vitest" />
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { SearchResultsCard } from './SearchResultsCard'

afterEach(() => cleanup())

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
}))

const mockGroups = [
  {
    type: 'member',
    label: 'Members',
    color: 'bg-purple-100 text-purple-700',
    link: '/workspaces',
    items: [
      { id: '1', type: 'member', title: 'alice', detail: 'alice@test.com' },
    ],
  },
  {
    type: 'dataset',
    label: 'Datasets',
    color: 'bg-green-100 text-green-700',
    link: '/datasets',
    items: [
      { id: '2', type: 'dataset', title: 'wiki-text', detail: 'Wikipedia dump' },
    ],
  },
]

describe('SearchResultsCard', () => {
  it('renders result count', () => {
    render(<SearchResultsCard groups={mockGroups} total={2} query="test" />)
    expect(screen.getByText('2 result(s) found')).toBeDefined()
  })

  it('renders group labels', () => {
    render(<SearchResultsCard groups={mockGroups} total={2} query="test" />)
    expect(screen.getByText('Members')).toBeDefined()
    expect(screen.getByText('Datasets')).toBeDefined()
  })

  it('renders item titles', () => {
    render(<SearchResultsCard groups={mockGroups} total={2} query="test" />)
    expect(screen.getByText('alice')).toBeDefined()
    expect(screen.getByText('wiki-text')).toBeDefined()
  })

  it('shows no results message when total is 0', () => {
    render(<SearchResultsCard groups={[]} total={0} query="nothing" />)
    expect(screen.getByText(/No results for/)).toBeDefined()
  })

  it('renders item details', () => {
    render(<SearchResultsCard groups={mockGroups} total={2} query="test" />)
    expect(screen.getByText('alice@test.com')).toBeDefined()
  })

  it('renders result counts per group', () => {
    render(<SearchResultsCard groups={mockGroups} total={2} query="test" />)
    const counts = screen.getAllByText('(1)')
    expect(counts.length).toBe(2)
  })
})
