// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react'
import { SearchConversationsDialog } from './SearchConversationsDialog'
import { chatDB } from '@/lib/db'

vi.mock('@/lib/db', () => ({
  chatDB: { searchAllSessions: vi.fn() },
}))

const mockPush = vi.fn()
vi.mock('next/navigation', () => ({ useRouter: () => ({ push: mockPush }) }))

describe('SearchConversationsDialog', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(cleanup)

  it('renders dialog when open', () => {
    render(<SearchConversationsDialog open onOpenChange={vi.fn()} />)
    expect(screen.getByText('Search Conversations')).toBeDefined()
    expect(screen.getByTestId('search-input')).toBeDefined()
  })

  it('does not render dialog content when closed', () => {
    const { container } = render(<SearchConversationsDialog open={false} onOpenChange={vi.fn()} />)
    expect(container.querySelector('[role="dialog"]')).toBeNull()
  })

  it('shows empty state when no query entered', () => {
    render(<SearchConversationsDialog open onOpenChange={vi.fn()} />)
    expect(screen.getByText('Type to search across all conversations')).toBeDefined()
  })

  it('renders search results', async () => {
    const mockResults = [
      {
        session: { id: 's1', name: 'Chat about AI' },
        matches: [{ role: 'user', content: 'hello world' }],
      },
    ]
    vi.mocked(chatDB.searchAllSessions).mockResolvedValue(mockResults as any)
    render(<SearchConversationsDialog open onOpenChange={vi.fn()} />)
    const input = screen.getByTestId('search-input')
    fireEvent.change(input, { target: { value: 'hello' } })
    await waitFor(() => {
      expect(screen.getByText('Chat about AI')).toBeDefined()
      expect(screen.getByText('1 match')).toBeDefined()
    })
  })

  it('shows no matches when search empty', async () => {
    vi.mocked(chatDB.searchAllSessions).mockResolvedValue([])
    render(<SearchConversationsDialog open onOpenChange={vi.fn()} />)
    const input = screen.getByTestId('search-input')
    fireEvent.change(input, { target: { value: 'zzzz' } })
    await waitFor(() => {
      expect(screen.getByText('No matches found')).toBeDefined()
    })
  })

  it('calls onOpenChange(false) and navigates on result click', async () => {
    const onOpenChange = vi.fn()
    const mockResults = [
      {
        session: { id: 's1', name: 'My Chat' },
        matches: [{ role: 'user', content: 'hello' }],
      },
    ]
    vi.mocked(chatDB.searchAllSessions).mockResolvedValue(mockResults as any)
    render(<SearchConversationsDialog open onOpenChange={onOpenChange} />)
    const input = screen.getByTestId('search-input')
    fireEvent.change(input, { target: { value: 'hello' } })
    await waitFor(() => expect(screen.getByText('My Chat')).toBeDefined())
    fireEvent.click(screen.getByText('My Chat'))
    expect(onOpenChange).toHaveBeenCalledWith(false)
    expect(mockPush).toHaveBeenCalledWith('/chat?session=s1')
  })

  it('clears query when clear button clicked', () => {
    render(<SearchConversationsDialog open onOpenChange={vi.fn()} />)
    const input = screen.getByTestId('search-input')
    fireEvent.change(input, { target: { value: 'hello' } })
    const clearBtn = screen.getAllByLabelText('Clear search')[0]
    fireEvent.click(clearBtn)
    expect((input as HTMLInputElement).value).toBe('')
  })
})
