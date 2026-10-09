'use client'

import { useState } from 'react'
import { cn, Button } from '@sloughgpt/strui'
import type { Note } from './types'

interface NotesViewProps {
  notes: Note[]
  onRefresh: () => void
}

export function NotesView({ notes, onRefresh }: NotesViewProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [filterStatus, setFilterStatus] = useState<string | null>(null)

  const filteredNotes = notes.filter((note) => {
    if (filterStatus && note.status !== filterStatus) return false
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase()
      return (
        note.title.toLowerCase().includes(q) ||
        note.body.toLowerCase().includes(q) ||
        note.tags.some((t) => t.toLowerCase().includes(q))
      )
    }
    return true
  })

  const statuses = ['open', 'wip', 'done', 'blocked', 'review', 'todo']

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search notes..."
          className="flex-1 max-w-xs rounded-lg border border-border bg-card px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
        />
        <div className="flex items-center gap-1">
          <Button
            variant={filterStatus === null ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setFilterStatus(null)}
          >
            All
          </Button>
          {statuses.map((status) => (
            <Button
              key={status}
              variant={filterStatus === status ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setFilterStatus(filterStatus === status ? null : status)}
            >
              {status}
            </Button>
          ))}
        </div>
      </div>

      <div className="space-y-2">
        {filteredNotes.length === 0 ? (
          <div className="text-center py-8 text-sm text-muted-foreground">
            No notes found
          </div>
        ) : (
          filteredNotes.map((note) => (
            <div
              key={note.id}
              className="rounded-lg border border-border bg-card p-3 hover:bg-muted/50 transition-colors"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-foreground truncate">
                    {note.title}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground line-clamp-2">
                    {note.body}
                  </p>
                </div>
                <span
                  className={cn(
                    'inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium uppercase tracking-wider',
                    note.status === 'done'
                      ? 'bg-success/15 text-success'
                      : note.status === 'wip'
                        ? 'bg-warning/15 text-warning'
                        : 'bg-muted text-muted-foreground',
                  )}
                >
                  {note.status}
                </span>
              </div>
              <div className="mt-2 flex items-center gap-1.5 flex-wrap">
                {note.tags.slice(0, 3).map((tag) => (
                  <span
                    key={tag}
                    className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] bg-muted text-muted-foreground"
                  >
                    {tag}
                  </span>
                ))}
                {note.tags.length > 3 && (
                  <span className="text-[10px] text-muted-foreground">
                    +{note.tags.length - 3}
                  </span>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
