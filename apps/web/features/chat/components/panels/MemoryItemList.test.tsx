import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  IconTrash: (props: any) => <span data-testid="icon-trash" {...props} />,
  IconEdit: (props: any) => <span data-testid="icon-edit" {...props} />,
}))

vi.mock('@/lib/memory-controller', () => ({
  memoryController: {},
}))

vi.mock('@/lib/format-bytes', () => ({
  formatRelativeTime: vi.fn(() => '2h ago'),
}))

import { MemoryItemList } from './MemoryItemList'
import type { MemoryItem } from '@/lib/memory-controller'

const baseItem: MemoryItem = {
  id: 'item-1',
  content: 'This is a test memory fact',
  topic: 'testing',
  source: 'manual',
  url: '',
  timestamp: Math.floor(Date.now() / 1000),
  importance: 0.7,
  score: 0,
}

afterEach(cleanup)

describe('MemoryItemList', () => {
  const onCopy = vi.fn()
  const onEdit = vi.fn()
  const onDelete = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders without crashing with empty list', () => {
    render(
      <MemoryItemList
        items={[]}
        highlightedId={null}
        copiedId={null}
        searchResults={null}
        onCopy={onCopy}
        onEdit={onEdit}
        onDelete={onDelete}
      />,
    )
    expect(screen.getByRole('list')).toBeDefined()
  })

  it('renders item content', () => {
    render(
      <MemoryItemList
        items={[baseItem]}
        highlightedId={null}
        copiedId={null}
        searchResults={null}
        onCopy={onCopy}
        onEdit={onEdit}
        onDelete={onDelete}
      />,
    )
    expect(screen.getByText('This is a test memory fact')).toBeDefined()
  })

  it('truncates content longer than 160 characters', () => {
    const longItem = { ...baseItem, content: 'A'.repeat(200) }
    render(
      <MemoryItemList
        items={[longItem]}
        highlightedId={null}
        copiedId={null}
        searchResults={null}
        onCopy={onCopy}
        onEdit={onEdit}
        onDelete={onDelete}
      />,
    )
    const truncated = screen.getByText(/A{160}…/)
    expect(truncated).toBeDefined()
  })

  it('calls onCopy when content is clicked', () => {
    render(
      <MemoryItemList
        items={[baseItem]}
        highlightedId={null}
        copiedId={null}
        searchResults={null}
        onCopy={onCopy}
        onEdit={onEdit}
        onDelete={onDelete}
      />,
    )
    fireEvent.click(screen.getByText('This is a test memory fact'))
    expect(onCopy).toHaveBeenCalledWith('This is a test memory fact', 'item-1')
  })

  it('calls onEdit when edit button is clicked', () => {
    render(
      <MemoryItemList
        items={[baseItem]}
        highlightedId={null}
        copiedId={null}
        searchResults={null}
        onCopy={onCopy}
        onEdit={onEdit}
        onDelete={onDelete}
      />,
    )
    fireEvent.click(screen.getByLabelText('Edit memory item'))
    expect(onEdit).toHaveBeenCalledWith(baseItem)
  })

  it('calls onDelete when delete button is clicked', () => {
    render(
      <MemoryItemList
        items={[baseItem]}
        highlightedId={null}
        copiedId={null}
        searchResults={null}
        onCopy={onCopy}
        onEdit={onEdit}
        onDelete={onDelete}
      />,
    )
    fireEvent.click(screen.getByLabelText('Delete memory item'))
    expect(onDelete).toHaveBeenCalledWith(baseItem)
  })

  it('shows Copied badge when copiedId matches', () => {
    render(
      <MemoryItemList
        items={[baseItem]}
        highlightedId={null}
        copiedId="item-1"
        searchResults={null}
        onCopy={onCopy}
        onEdit={onEdit}
        onDelete={onDelete}
      />,
    )
    expect(screen.getByText('Copied')).toBeDefined()
  })

  it('applies highlighted style when highlightedId matches', () => {
    render(
      <MemoryItemList
        items={[baseItem]}
        highlightedId="item-1"
        copiedId={null}
        searchResults={null}
        onCopy={onCopy}
        onEdit={onEdit}
        onDelete={onDelete}
      />,
    )
    const li = screen.getByRole('listitem')
    expect(li.className).toContain('border-primary/60')
  })

  it('shows topic and importance when present', () => {
    render(
      <MemoryItemList
        items={[baseItem]}
        highlightedId={null}
        copiedId={null}
        searchResults={null}
        onCopy={onCopy}
        onEdit={onEdit}
        onDelete={onDelete}
      />,
    )
    expect(screen.getByText('testing')).toBeDefined()
    expect(screen.getByText(/importance 0\.7/)).toBeDefined()
  })

  it('shows score when searchResults is not null', () => {
    const scoredItem = { ...baseItem, score: 0.92 }
    render(
      <MemoryItemList
        items={[scoredItem]}
        highlightedId={null}
        copiedId={null}
        searchResults={[scoredItem]}
        onCopy={onCopy}
        onEdit={onEdit}
        onDelete={onDelete}
      />,
    )
    expect(screen.getByText('0.92')).toBeDefined()
  })

  it('hides score when searchResults is null', () => {
    const scoredItem = { ...baseItem, score: 0.92 }
    render(
      <MemoryItemList
        items={[scoredItem]}
        highlightedId={null}
        copiedId={null}
        searchResults={null}
        onCopy={onCopy}
        onEdit={onEdit}
        onDelete={onDelete}
      />,
    )
    expect(screen.queryByText('0.92')).toBeNull()
  })
})
