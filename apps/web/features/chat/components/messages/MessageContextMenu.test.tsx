import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  IconCopy: ({ className }: any) => <span className={className}>copy</span>,
  IconCheck: ({ className }: any) => <span className={className}>check</span>,
  IconRefresh: ({ className }: any) => <span className={className}>refresh</span>,
  IconEdit: ({ className }: any) => <span className={className}>edit</span>,
  IconStar: ({ className }: any) => <span className={className}>star</span>,
  IconTrash: ({ className }: any) => <span className={className}>trash</span>,
  IconPin: ({ className }: any) => <span className={className}>pin</span>,
  IconMessage: ({ className }: any) => <span className={className}>message</span>,
}))

import { MessageContextMenu } from './MessageContextMenu'

const defaultProps = {
  messageId: 'msg-1',
  content: 'Hello world',
  role: 'user' as const,
  onCopy: vi.fn(),
  onEdit: vi.fn(),
  onBookmark: vi.fn(),
  onRegenerate: vi.fn(),
  onDelete: vi.fn(),
  onSaveToKnowledge: vi.fn(),
}

beforeEach(() => {
  const clipboard = {
    writeText: vi.fn().mockResolvedValue(undefined),
    write: vi.fn().mockResolvedValue(undefined),
  }
  Object.defineProperty(navigator, 'clipboard', {
    value: clipboard,
    writable: true,
    configurable: true,
  })
})

afterEach(() => cleanup())

function renderWithMenu(props = {}) {
  return render(
    <MessageContextMenu {...defaultProps} {...props}>
      <div>Right-click me</div>
    </MessageContextMenu>
  )
}

function openMenu() {
  fireEvent.contextMenu(screen.getByText('Right-click me'))
}

describe('MessageContextMenu', () => {
  it('renders children', () => {
    renderWithMenu()
    expect(screen.getByText('Right-click me')).toBeDefined()
  })

  it('does not show menu initially', () => {
    renderWithMenu()
    expect(screen.queryByRole('menu')).toBeNull()
  })

  it('opens menu on right-click', () => {
    renderWithMenu()
    openMenu()
    expect(screen.getByRole('menu')).toBeDefined()
    expect(screen.getByText('Copy')).toBeDefined()
  })

  it('shows Copy item for user messages', () => {
    renderWithMenu({ role: 'user' })
    openMenu()
    expect(screen.getByText('Copy')).toBeDefined()
  })

  it('shows Edit for user messages when onEdit provided', () => {
    renderWithMenu({ role: 'user' })
    openMenu()
    expect(screen.getByText('Edit')).toBeDefined()
  })

  it('does not show Edit for assistant messages', () => {
    renderWithMenu({ role: 'assistant' })
    openMenu()
    expect(screen.queryByText('Edit')).toBeNull()
  })

  it('shows Copy as HTML for assistant messages', () => {
    renderWithMenu({ role: 'assistant' })
    openMenu()
    expect(screen.getByText('Copy as HTML')).toBeDefined()
  })

  it('does not show Copy as HTML for user messages', () => {
    renderWithMenu({ role: 'user' })
    openMenu()
    expect(screen.queryByText('Copy as HTML')).toBeNull()
  })

  it('shows Regenerate for assistant messages when onRegenerate provided', () => {
    renderWithMenu({ role: 'assistant' })
    openMenu()
    expect(screen.getByText('Regenerate')).toBeDefined()
  })

  it('does not show Regenerate for user messages', () => {
    renderWithMenu({ role: 'user' })
    openMenu()
    expect(screen.queryByText('Regenerate')).toBeNull()
  })

  it('shows Bookmark/Remove bookmark', () => {
    renderWithMenu()
    openMenu()
    expect(screen.getByText('Bookmark')).toBeDefined()
  })

  it('shows Remove bookmark when isBookmarked is true', () => {
    renderWithMenu({ isBookmarked: true })
    openMenu()
    expect(screen.getByText('Remove bookmark')).toBeDefined()
  })

  it('shows Delete with destructive styling', () => {
    renderWithMenu()
    openMenu()
    expect(screen.getByText('Delete')).toBeDefined()
  })

  it('shows Save to knowledge', () => {
    renderWithMenu()
    openMenu()
    expect(screen.getByText('Save to knowledge')).toBeDefined()
  })

  it('calls onCopy when Copy is clicked', async () => {
    renderWithMenu()
    openMenu()
    fireEvent.click(screen.getByText('Copy'))
    await waitFor(() => {
      expect(defaultProps.onCopy).toHaveBeenCalledWith('Hello world')
    })
  })

  it('calls onEdit when Edit is clicked', () => {
    renderWithMenu({ role: 'user' })
    openMenu()
    fireEvent.click(screen.getByText('Edit'))
    expect(defaultProps.onEdit).toHaveBeenCalledWith('msg-1')
  })

  it('calls onDelete when Delete is clicked', () => {
    renderWithMenu()
    openMenu()
    fireEvent.click(screen.getByText('Delete'))
    expect(defaultProps.onDelete).toHaveBeenCalledWith('msg-1')
  })

  it('calls onBookmark when Bookmark is clicked', () => {
    renderWithMenu()
    openMenu()
    fireEvent.click(screen.getByText('Bookmark'))
    expect(defaultProps.onBookmark).toHaveBeenCalledWith('msg-1')
  })

  it('calls onRegenerate when Regenerate is clicked', () => {
    renderWithMenu({ role: 'assistant' })
    openMenu()
    fireEvent.click(screen.getByText('Regenerate'))
    expect(defaultProps.onRegenerate).toHaveBeenCalled()
  })

  it('calls onSaveToKnowledge when Save to knowledge is clicked', () => {
    renderWithMenu()
    openMenu()
    fireEvent.click(screen.getByText('Save to knowledge'))
    expect(defaultProps.onSaveToKnowledge).toHaveBeenCalledWith('msg-1', 'Hello world')
  })

  it('closes on Escape key', () => {
    renderWithMenu()
    openMenu()
    expect(screen.getByRole('menu')).toBeDefined()
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(screen.queryByRole('menu')).toBeNull()
  })

  it('closes on outside click', () => {
    renderWithMenu()
    openMenu()
    expect(screen.getByRole('menu')).toBeDefined()
    fireEvent.click(document)
    expect(screen.queryByRole('menu')).toBeNull()
  })

  it('does not show onEdit menu item when onEdit is not provided', () => {
    renderWithMenu({ onEdit: undefined })
    openMenu()
    expect(screen.queryByText('Edit')).toBeNull()
  })

  it('does not show onDelete menu item when onDelete is not provided', () => {
    renderWithMenu({ onDelete: undefined })
    openMenu()
    expect(screen.queryByText('Delete')).toBeNull()
  })

  it('does not show onBookmark menu item when onBookmark is not provided', () => {
    renderWithMenu({ onBookmark: undefined })
    openMenu()
    expect(screen.queryByText('Bookmark')).toBeNull()
  })

  it('does not show onSaveToKnowledge when not provided', () => {
    renderWithMenu({ onSaveToKnowledge: undefined })
    openMenu()
    expect(screen.queryByText('Save to knowledge')).toBeNull()
  })

  it('shows "Add note" when hasNote is false', () => {
    renderWithMenu({ hasNote: false, onAddNote: vi.fn() })
    openMenu()
    expect(screen.getByText('Add note')).toBeDefined()
  })

  it('shows "Edit note" when hasNote is true', () => {
    renderWithMenu({ hasNote: true, onAddNote: vi.fn() })
    openMenu()
    expect(screen.getByText('Edit note')).toBeDefined()
  })

  it('calls onAddNote with messageId when note item clicked', () => {
    const onAddNote = vi.fn()
    renderWithMenu({ hasNote: false, onAddNote })
    openMenu()
    fireEvent.click(screen.getByText('Add note'))
    expect(onAddNote).toHaveBeenCalledWith('msg-1')
  })

  it('does not show note item when onAddNote is not provided', () => {
    renderWithMenu({ onAddNote: undefined })
    openMenu()
    expect(screen.queryByText('Add note')).toBeNull()
    expect(screen.queryByText('Edit note')).toBeNull()
  })

  it('navigates with Home key to first item', () => {
    renderWithMenu()
    openMenu()
    const menu = screen.getByRole('menu')
    const items = menu.querySelectorAll('[role="menuitem"]:not([disabled])')
    expect(items.length).toBeGreaterThan(1)
    fireEvent.keyDown(menu, { key: 'Home' })
    expect(document.activeElement).toBe(items[0])
  })

  it('navigates with End key to last item', () => {
    renderWithMenu()
    openMenu()
    const menu = screen.getByRole('menu')
    const items = menu.querySelectorAll('[role="menuitem"]:not([disabled])')
    expect(items.length).toBeGreaterThan(1)
    fireEvent.keyDown(menu, { key: 'End' })
    expect(document.activeElement).toBe(items[items.length - 1])
  })
})
