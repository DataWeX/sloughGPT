'use client'

import { useState, useEffect } from 'react'
import { cn, Button, IconX } from '@sloughgpt/strui'
import type { Note } from './types'

interface NoteEditorProps {
  note: Note | null
  onClose: () => void
  onUpdate: (noteId: string, data: Partial<Note>) => void
  onDelete: (noteId: string) => void
}

export function NoteEditor({ note, onClose, onUpdate, onDelete }: NoteEditorProps) {
  const [title, setTitle] = useState('')
  const [body, setBody] = useState('')
  const [status, setStatus] = useState('open')
  const [tags, setTags] = useState('')
  const [sprint, setSprint] = useState('')
  const [gh, setGh] = useState('')

  useEffect(() => {
    if (note) {
      setTitle(note.title)
      setBody(note.body)
      setStatus(note.status)
      setTags(note.tags.join(', '))
      setSprint(note.sprint)
      setGh(note.gh)
    }
  }, [note])

  if (!note) return null

  const handleSave = () => {
    onUpdate(note.id, {
      title: title.trim(),
      body: body.trim(),
      status,
      tags: tags.split(',').map((t) => t.trim()).filter(Boolean),
      sprint: sprint.trim(),
      gh: gh.trim(),
    })
  }

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div className="relative w-full max-w-md bg-card border-l border-border shadow-lg overflow-y-auto">
        <div className="sticky top-0 bg-card border-b border-border px-4 py-3 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Edit Note</h2>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <IconX className="h-4 w-4" />
          </Button>
        </div>
        <div className="p-4 space-y-4">
          <div>
            <label className="text-sm font-medium">Title</label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="w-full mt-1 rounded-lg border border-border bg-background px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="text-sm font-medium">Body</label>
            <textarea
              value={body}
              onChange={(e) => setBody(e.target.value)}
              className="w-full mt-1 rounded-lg border border-border bg-background px-3 py-2 text-sm font-mono"
              rows={12}
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium">Status</label>
              <select
                value={status}
                onChange={(e) => setStatus(e.target.value)}
                className="w-full mt-1 rounded-lg border border-border bg-background px-3 py-2 text-sm"
              >
                <option value="open">Open</option>
                <option value="wip">WIP</option>
                <option value="done">Done</option>
                <option value="blocked">Blocked</option>
                <option value="review">Review</option>
                <option value="todo">Todo</option>
              </select>
            </div>
            <div>
              <label className="text-sm font-medium">Tags (comma-separated)</label>
              <input
                type="text"
                value={tags}
                onChange={(e) => setTags(e.target.value)}
                className="w-full mt-1 rounded-lg border border-border bg-background px-3 py-2 text-sm"
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium">Sprint</label>
              <input
                type="text"
                value={sprint}
                onChange={(e) => setSprint(e.target.value)}
                className="w-full mt-1 rounded-lg border border-border bg-background px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label className="text-sm font-medium">GitHub</label>
              <input
                type="text"
                value={gh}
                onChange={(e) => setGh(e.target.value)}
                className="w-full mt-1 rounded-lg border border-border bg-background px-3 py-2 text-sm"
              />
            </div>
          </div>
          <div className="flex justify-between pt-4">
            <Button
              variant="destructive"
              size="sm"
              onClick={() => onDelete(note.id)}
            >
              Delete
            </Button>
            <div className="flex gap-2">
              <Button variant="ghost" size="sm" onClick={onClose}>
                Cancel
              </Button>
              <Button size="sm" onClick={handleSave}>
                Save
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
