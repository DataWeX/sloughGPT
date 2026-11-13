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
  
Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,
    Separator: () => <hr />,
    Tooltip: ({ children }: any) => <>{children}</>,
    TooltipTrigger: ({ children }: any) => <>{children}</>,
    TooltipContent: ({ children }: any) => <>{children}</>,
    Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
    Avatar: ({ children }: any) => <div>{children}</div>,
    AvatarFallback: ({ children }: any) => <div>{children}</div>,
    ScrollArea: ({ children }: any) => <div>{children}</div>,
    Table: ({ children }: any) => <table>{children}</table>,
    TableBody: ({ children }: any) => <tbody>{children}</tbody>,
    TableRow: ({ children }: any) => <tr>{children}</tr>,
    TableCell: ({ children }: any) => <td>{children}</td>,
    TableHead: ({ children }: any) => <th>{children}</th>,
    TableHeader: ({ children }: any) => <thead>{children}</thead>,
    Collapsible: ({ children }: any) => <div>{children}</div>,
    CollapsibleTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    CollapsibleContent: ({ children }: any) => <div>{children}</div>,
    Toggle: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Command: ({ children }: any) => <div>{children}</div>,
    CommandInput: ({ ...props }: any) => <input {...props} />,
    CommandList: ({ children }: any) => <div>{children}</div>,
    CommandEmpty: ({ children }: any) => <div>{children}</div>,
    CommandGroup: ({ children }: any) => <div>{children}</div>,
    CommandItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
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
