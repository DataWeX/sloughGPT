'use client'

import { useState, useCallback, useEffect, memo } from 'react'
import { Button } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

interface Session {
  id: string
  title: string
  lastMessage?: string
  updatedAt: number
}

interface ChatSessionFavoritesProps {
  sessions: Session[]
  onOpenSession: (sessionId: string) => void
  onRemoveFavorite: (sessionId: string) => void
  className?: string
}

const STORAGE_KEY = 'chat-session-favorites'

function loadFavorites(): string[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveFavorites(favorites: string[]) {
  if (typeof window === 'undefined') return
  localStorage.setItem(STORAGE_KEY, JSON.stringify(favorites))
}

export const ChatSessionFavorites = memo(function ChatSessionFavorites({
  sessions,
  onOpenSession,
  onRemoveFavorite,
  className,
}: ChatSessionFavoritesProps) {
  const [favoriteIds, setFavoriteIds] = useState<string[]>([])

  useEffect(() => {
    setFavoriteIds(loadFavorites())
  }, [])

  const favoriteSessions = favoriteIds
    .map(id => sessions.find(s => s.id === id))
    .filter((s): s is Session => s !== undefined)

  const handleRemove = useCallback((sessionId: string) => {
    const next = favoriteIds.filter(id => id !== sessionId)
    setFavoriteIds(next)
    saveFavorites(next)
    onRemoveFavorite(sessionId)
  }, [favoriteIds, onRemoveFavorite])

  const formatDate = useCallback((timestamp: number) => {
    const diff = Date.now() - timestamp
    const minutes = Math.floor(diff / 60000)
    if (minutes < 60) return `${minutes}m ago`
    const hours = Math.floor(minutes / 60)
    if (hours < 24) return `${hours}h ago`
    const days = Math.floor(hours / 24)
    return `${days}d ago`
  }, [])

  return (
    <div className={cn('border rounded-lg bg-card overflow-hidden', className)}>
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30">
        <span className="text-xs font-medium">Favorites</span>
        <span className="text-[10px] text-muted-foreground">{favoriteSessions.length}</span>
      </div>

      <div className="max-h-[300px] overflow-y-auto">
        {favoriteSessions.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">
            No favorites yet. Star a session to add it here.
          </p>
        ) : (
          <div className="divide-y">
            {favoriteSessions.map(session => (
              <div
                key={session.id}
                className="px-3 py-2 hover:bg-muted/30 group cursor-pointer"
                onClick={() => onOpenSession(session.id)}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <p className="text-xs font-medium truncate">{session.title}</p>
                    {session.lastMessage && (
                      <p className="text-[10px] text-muted-foreground truncate mt-0.5">
                        {session.lastMessage.slice(0, 50)}
                      </p>
                    )}
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    <span className="text-[10px] text-muted-foreground">
                      {formatDate(session.updatedAt)}
                    </span>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      className="h-5 w-5 opacity-0 group-hover:opacity-100"
                      onClick={(e) => {
                        e.stopPropagation()
                        handleRemove(session.id)
                      }}
                      title="Remove from favorites"
                    >
                      <span className="text-destructive">×</span>
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
})

export function useSessionFavorites() {
  const [favorites, setFavorites] = useState<string[]>([])

  useEffect(() => {
    setFavorites(loadFavorites())
  }, [])

  const addFavorite = useCallback((sessionId: string) => {
    const next = [...new Set([...favorites, sessionId])]
    setFavorites(next)
    saveFavorites(next)
  }, [favorites])

  const removeFavorite = useCallback((sessionId: string) => {
    const next = favorites.filter(id => id !== sessionId)
    setFavorites(next)
    saveFavorites(next)
  }, [favorites])

  const isFavorite = useCallback((sessionId: string) => {
    return favorites.includes(sessionId)
  }, [favorites])

  const toggleFavorite = useCallback((sessionId: string) => {
    if (favorites.includes(sessionId)) {
      removeFavorite(sessionId)
    } else {
      addFavorite(sessionId)
    }
  }, [favorites, addFavorite, removeFavorite])

  return { favorites, addFavorite, removeFavorite, isFavorite, toggleFavorite }
}