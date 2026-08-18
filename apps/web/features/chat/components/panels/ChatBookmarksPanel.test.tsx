// @vitest-environment jsdom
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { ChatBookmarksPanel } from './ChatBookmarksPanel'
import type { BookmarkedMessage } from '@/features/chat/hooks/useChatBookmarks'

vi.mock('@sloughgpt/strui', () => {
  const Btn = (props: any) => <button {...props} />
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    Button: Btn,
    IconStar: (p: any) => <svg data-testid="icon-star" {...p} />,
    IconTrash: (p: any) => <svg data-testid="icon-trash" {...p} />,
    IconX: (p: any) => <svg data-testid="icon-x" {...p} />,
    IconChevronDown: (p: any) => <svg data-testid="icon-chevron-down" {...p} />,
  }
})

describe('ChatBookmarksPanel', () => {
  const defaultProps = {
    bookmarks: [] as BookmarkedMessage[],
    onRemove: vi.fn(),
    onClear: vi.fn(),
  }

  const sampleBookmark: BookmarkedMessage = {
    id: 'bm-1',
    content: 'This is a bookmarked message',
    role: 'user',
    sessionTitle: 'Test Session',
    timestamp: Date.now(),
  }

  const assistantBookmark: BookmarkedMessage = {
    id: 'bm-2',
    content: 'Assistant response here',
    role: 'assistant',
    timestamp: Date.now(),
  }

  it('renders bookmarks header', () => {
    render(<ChatBookmarksPanel {...defaultProps} bookmarks={[sampleBookmark]} />)
    expect(screen.getAllByText('Bookmarks').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no bookmarks', () => {
    render(<ChatBookmarksPanel {...defaultProps} />)
    expect(screen.getAllByText('No bookmarks yet').length).toBeGreaterThanOrEqual(1)
  })

  it('renders bookmark content (user role)', () => {
    render(<ChatBookmarksPanel {...defaultProps} bookmarks={[sampleBookmark]} />)
    expect(screen.getAllByText('This is a bookmarked message').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('You').length).toBeGreaterThanOrEqual(1)
  })

  it('renders assistant role label', () => {
    render(<ChatBookmarksPanel {...defaultProps} bookmarks={[assistantBookmark]} />)
    expect(screen.getAllByText('Assistant').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onRemove when remove button clicked', () => {
    const onRemove = vi.fn()
    const { container } = render(<ChatBookmarksPanel {...defaultProps} bookmarks={[sampleBookmark]} onRemove={onRemove} />)
    const removeBtn = container.querySelector('[aria-label="Remove bookmark"]')!
    expect(removeBtn).toBeTruthy()
    fireEvent.click(removeBtn)
    expect(onRemove).toHaveBeenCalledWith('bm-1')
  })

  it('calls onClear when Clear all clicked', () => {
    const onClear = vi.fn()
    const { container } = render(<ChatBookmarksPanel {...defaultProps} bookmarks={[sampleBookmark]} onClear={onClear} />)
    const clearBtn = container.querySelector('.hover\\:text-error')!
    expect(clearBtn).toBeTruthy()
    fireEvent.click(clearBtn)
    expect(onClear).toHaveBeenCalledOnce()
  })

  it('calls onJumpToMessage when bookmark content clicked', () => {
    const onJumpToMessage = vi.fn()
    const { container } = render(<ChatBookmarksPanel {...defaultProps} bookmarks={[sampleBookmark]} onJumpToMessage={onJumpToMessage} />)
    const contentBtn = container.querySelector('.flex-1.min-w-0')!
    expect(contentBtn).toBeTruthy()
    fireEvent.click(contentBtn)
    expect(onJumpToMessage).toHaveBeenCalledWith('bm-1')
  })

  it('collapses and expands on header click', () => {
    const { container } = render(<ChatBookmarksPanel {...defaultProps} bookmarks={[sampleBookmark]} />)
    expect(screen.getAllByText('This is a bookmarked message').length).toBeGreaterThanOrEqual(1)
    const header = container.querySelector('[aria-expanded]')!
    expect(header).toBeTruthy()
    fireEvent.click(header)
    expect(container.querySelector('.divide-y')).toBeNull()
  })

  it('hides Clear all when empty', () => {
    const { container } = render(<ChatBookmarksPanel {...defaultProps} bookmarks={[]} />)
    expect(container.querySelector('.hover\\:text-error')).toBeNull()
  })
})
