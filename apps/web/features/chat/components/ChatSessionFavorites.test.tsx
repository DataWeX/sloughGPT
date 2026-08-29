import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { ChatSessionFavorites, useSessionFavorites } from './ChatSessionFavorites'
import { renderHook, act as actHook } from '@testing-library/react'

afterEach(cleanup)
beforeEach(() => {
  localStorage.clear()
})

const mockSessions = [
  { id: 's1', title: 'Session 1', lastMessage: 'Hello', updatedAt: Date.now() },
  { id: 's2', title: 'Session 2', lastMessage: 'World', updatedAt: Date.now() - 3600000 },
  { id: 's3', title: 'Session 3', updatedAt: Date.now() - 86400000 },
]

describe('ChatSessionFavorites', () => {
  it('renders empty state', () => {
    render(
      <ChatSessionFavorites
        sessions={mockSessions}
        onOpenSession={vi.fn()}
        onRemoveFavorite={vi.fn()}
      />
    )
    expect(screen.getByText(/No favorites yet/)).toBeInTheDocument()
  })

  it('renders title', () => {
    render(
      <ChatSessionFavorites
        sessions={mockSessions}
        onOpenSession={vi.fn()}
        onRemoveFavorite={vi.fn()}
      />
    )
    expect(screen.getByText('Favorites')).toBeInTheDocument()
  })

  it('renders favorite sessions', () => {
    localStorage.setItem('chat-session-favorites', JSON.stringify(['s1', 's2']))
    render(
      <ChatSessionFavorites
        sessions={mockSessions}
        onOpenSession={vi.fn()}
        onRemoveFavorite={vi.fn()}
      />
    )
    expect(screen.getByText('Session 1')).toBeInTheDocument()
    expect(screen.getByText('Session 2')).toBeInTheDocument()
  })

  it('calls onOpenSession when clicked', () => {
    const onOpenSession = vi.fn()
    localStorage.setItem('chat-session-favorites', JSON.stringify(['s1']))
    render(
      <ChatSessionFavorites
        sessions={mockSessions}
        onOpenSession={onOpenSession}
        onRemoveFavorite={vi.fn()}
      />
    )
    fireEvent.click(screen.getByText('Session 1'))
    expect(onOpenSession).toHaveBeenCalledWith('s1')
  })

  it('removes from favorites', () => {
    const onRemoveFavorite = vi.fn()
    localStorage.setItem('chat-session-favorites', JSON.stringify(['s1']))
    render(
      <ChatSessionFavorites
        sessions={mockSessions}
        onOpenSession={vi.fn()}
        onRemoveFavorite={onRemoveFavorite}
      />
    )
    fireEvent.click(screen.getByTitle('Remove from favorites'))
    expect(onRemoveFavorite).toHaveBeenCalledWith('s1')
  })

  it('formats relative time', () => {
    localStorage.setItem('chat-session-favorites', JSON.stringify(['s1']))
    render(
      <ChatSessionFavorites
        sessions={mockSessions}
        onOpenSession={vi.fn()}
        onRemoveFavorite={vi.fn()}
      />
    )
    expect(screen.getByText(/\d+m ago/)).toBeInTheDocument()
  })

  it('shows count', () => {
    localStorage.setItem('chat-session-favorites', JSON.stringify(['s1', 's2']))
    render(
      <ChatSessionFavorites
        sessions={mockSessions}
        onOpenSession={vi.fn()}
        onRemoveFavorite={vi.fn()}
      />
    )
    expect(screen.getByText('2')).toBeInTheDocument()
  })

  it('renders with last message preview', () => {
    localStorage.setItem('chat-session-favorites', JSON.stringify(['s1']))
    render(
      <ChatSessionFavorites
        sessions={mockSessions}
        onOpenSession={vi.fn()}
        onRemoveFavorite={vi.fn()}
      />
    )
    expect(screen.getByText('Hello')).toBeInTheDocument()
  })
})

describe('useSessionFavorites', () => {
  it('loads favorites from localStorage', () => {
    localStorage.setItem('chat-session-favorites', JSON.stringify(['s1', 's2']))
    const { result } = renderHook(() => useSessionFavorites())
    expect(result.current.favorites).toEqual(['s1', 's2'])
  })

  it('adds a favorite', () => {
    const { result } = renderHook(() => useSessionFavorites())
    actHook(() => {
      result.current.addFavorite('s1')
    })
    expect(result.current.favorites).toContain('s1')
  })

  it('removes a favorite', () => {
    localStorage.setItem('chat-session-favorites', JSON.stringify(['s1']))
    const { result } = renderHook(() => useSessionFavorites())
    actHook(() => {
      result.current.removeFavorite('s1')
    })
    expect(result.current.favorites).not.toContain('s1')
  })

  it('checks if favorite', () => {
    localStorage.setItem('chat-session-favorites', JSON.stringify(['s1']))
    const { result } = renderHook(() => useSessionFavorites())
    expect(result.current.isFavorite('s1')).toBe(true)
    expect(result.current.isFavorite('s2')).toBe(false)
  })

  it('toggles favorite', () => {
    const { result } = renderHook(() => useSessionFavorites())
    actHook(() => {
      result.current.toggleFavorite('s1')
    })
    expect(result.current.favorites).toContain('s1')
    actHook(() => {
      result.current.toggleFavorite('s1')
    })
    expect(result.current.favorites).not.toContain('s1')
  })

  it('persists to localStorage', () => {
    const { result } = renderHook(() => useSessionFavorites())
    actHook(() => {
      result.current.addFavorite('s1')
    })
    const stored = JSON.parse(localStorage.getItem('chat-session-favorites') || '[]')
    expect(stored).toContain('s1')
  })
})