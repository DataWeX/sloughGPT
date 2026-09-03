import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { MemoryItemList } from './MemoryItemList'
import type { MemoryItem } from '@/lib/memory-controller'

vi.mock('@/lib/format-bytes', () => ({ formatRelativeTime: () => '2h ago' }))

afterEach(() => cleanup())

const makeItem = (overrides: Partial<MemoryItem> = {}): MemoryItem => ({
  id: 'm1', content: 'The sky is blue', topic: 'nature', source: 'chat',
  importance: 0.8, timestamp: Date.now() / 1000, ...overrides,
} as MemoryItem)

const defaultProps = {
  items: [], searchResults: null, loading: false, searched: false,
  activeTopic: null, sortOrder: 'newest' as const, showAllItems: true,
  selectedIds: new Set<string>(),
  onToggleSelect: vi.fn(), onToggleSelectAll: vi.fn(), onClearSelection: vi.fn(),
  onStartEdit: vi.fn(), onSetPendingDelete: vi.fn(), onSetPendingBatchDelete: vi.fn(),
  onExportSelected: vi.fn(), onCopy: vi.fn(), onClearSearch: vi.fn(),
  setShowAllItems: vi.fn(), setSearch: vi.fn(), setSearchResults: vi.fn(), setSearched: vi.fn(),
}

describe('MemoryItemList', () => {
  it('shows loading skeletons', () => {
    const { container } = render(<MemoryItemList {...defaultProps} loading={true} items={[]} />)
    expect(container.querySelectorAll('[class*="animate-pulse"]').length).toBeGreaterThanOrEqual(1)
  })
  it('shows empty state', () => {
    render(<MemoryItemList {...defaultProps} />)
    expect(screen.getAllByText('Nothing remembered yet. The AI stores facts automatically as you chat.').length).toBeGreaterThanOrEqual(1)
  })
  it('shows empty state when searched with no results', () => {
    render(<MemoryItemList {...defaultProps} searched={true} searchResults={[]} />)
    expect(screen.getAllByText('No memory matches that search.').length).toBeGreaterThanOrEqual(1)
  })
  it('shows topic-specific empty state', () => {
    render(<MemoryItemList {...defaultProps} activeTopic="science" items={[]} />)
    expect(screen.getAllByText(/No memory in the "science" topic/).length).toBeGreaterThanOrEqual(1)
  })
  it('renders items', () => {
    render(<MemoryItemList {...defaultProps} items={[makeItem()]} />)
    expect(screen.getAllByText('The sky is blue').length).toBeGreaterThanOrEqual(1)
  })
  it('renders item topic badge', () => {
    render(<MemoryItemList {...defaultProps} items={[makeItem({ topic: 'science' })]} />)
    expect(screen.getAllByText('science').length).toBeGreaterThanOrEqual(1)
  })
  it('renders item source badge', () => {
    render(<MemoryItemList {...defaultProps} items={[makeItem({ source: 'url' })]} />)
    expect(screen.getAllByText('url').length).toBeGreaterThanOrEqual(1)
  })
  it('renders importance badge', () => {
    render(<MemoryItemList {...defaultProps} items={[makeItem({ importance: 0.9 })]} />)
    expect(screen.getAllByText('importance 0.9').length).toBeGreaterThanOrEqual(1)
  })
  it('shows select all when multiple items', () => {
    render(<MemoryItemList {...defaultProps} items={[makeItem({ id: 'm1' }), makeItem({ id: 'm2' })]} />)
    expect(screen.getAllByText(/Select all/).length).toBeGreaterThanOrEqual(1)
  })
  it('shows batch actions when items selected', () => {
    render(<MemoryItemList {...defaultProps} items={[makeItem()]} selectedIds={new Set(['m1'])} />)
    expect(screen.getAllByText(/Export/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Delete/).length).toBeGreaterThanOrEqual(1)
  })
  it('shows Show all when more than 10 items', () => {
    const items = Array.from({ length: 15 }, (_, i) => makeItem({ id: `m${i}`, content: `Item ${i}` }))
    render(<MemoryItemList {...defaultProps} items={items} showAllItems={false} />)
    expect(screen.getAllByText('Show all').length).toBeGreaterThanOrEqual(1)
  })
  it('calls onCopy when item clicked', () => {
    const onCopy = vi.fn()
    render(<MemoryItemList {...defaultProps} items={[makeItem()]} onCopy={onCopy} />)
    fireEvent.click(screen.getAllByText('The sky is blue')[0])
    expect(onCopy).toHaveBeenCalledWith('The sky is blue')
  })
  it('shows score when in search results', () => {
    const item = { ...makeItem(), score: 0.95 }
    render(<MemoryItemList {...defaultProps} searchResults={[item]} items={[]} />)
    expect(screen.getAllByText('0.95').length).toBeGreaterThanOrEqual(1)
  })
})
