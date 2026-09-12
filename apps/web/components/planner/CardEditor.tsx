'use client'

import { useState, useEffect } from 'react'
import { Button, IconX } from '@sloughgpt/strui'
import type { Card, HashTree } from './types'
import { COLUMN_LABELS } from './types'
import { oon } from '@/lib/oon'

interface CardEditorProps {
  card: Card | null
  onClose: () => void
  onUpdate: (cardId: string, data: Partial<Card>) => void
  onDelete: (cardId: string) => void
}

export function CardEditor({ card, onClose, onUpdate, onDelete }: CardEditorProps) {
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [column, setColumn] = useState('todo')
  const [priority, setPriority] = useState('medium')
  const [tags, setTags] = useState('')
  const [assignee, setAssignee] = useState('')
  const [dueDate, setDueDate] = useState('')
  const [sprint, setSprint] = useState('')
  const [gh, setGh] = useState('')
  const [hashTree, setHashTree] = useState<HashTree | null>(null)
  const [showHashTree, setShowHashTree] = useState(false)

  useEffect(() => {
    if (card) {
      setTitle(card.title)
      setDescription(card.description)
      setColumn(card.column)
      setPriority(card.priority)
      setTags(card.tags.join(', '))
      setAssignee(card.assignee)
      setDueDate(card.due_date)
      setSprint(card.sprint)
      setGh(card.gh)
      // Load hash tree
      oon.hashTree(card.id)
        .then(data => setHashTree(data as HashTree | null))
        .catch(() => {})
    }
  }, [card])

  if (!card) return null

  const handleSave = () => {
    onUpdate(card.id, {
      title: title.trim(),
      description: description.trim(),
      column,
      priority: priority as Card['priority'],
      tags: tags.split(',').map((t) => t.trim()).filter(Boolean),
      assignee: assignee.trim(),
      due_date: dueDate.trim(),
      sprint: sprint.trim(),
      gh: gh.trim(),
    })
  }

  return (
    <div className="fixed inset-0 z-50 flex justify-end" role="dialog" aria-modal="true" aria-label="Edit card">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} aria-hidden="true" />
      <div className="relative w-full max-w-md bg-card border-l border-border shadow-lg overflow-y-auto">
        <div className="sticky top-0 bg-card border-b border-border px-4 py-3 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Edit Card</h2>
          <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close">
            <IconX className="h-4 w-4" />
          </Button>
        </div>
        <div className="p-4 space-y-4">
          <div>
            <label htmlFor="card-title" className="text-sm font-medium">Title</label>
            <input
              id="card-title"
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="w-full mt-1 rounded-lg border border-border bg-background px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label htmlFor="card-description" className="text-sm font-medium">Description</label>
            <textarea
              id="card-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full mt-1 rounded-lg border border-border bg-background px-3 py-2 text-sm"
              rows={4}
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="card-column" className="text-sm font-medium">Column</label>
              <select
                id="card-column"
                value={column}
                onChange={(e) => setColumn(e.target.value)}
                className="w-full mt-1 rounded-lg border border-border bg-background px-3 py-2 text-sm"
              >
                {Object.entries(COLUMN_LABELS).map(([value, label]) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="card-priority" className="text-sm font-medium">Priority</label>
              <select
                id="card-priority"
                value={priority}
                onChange={(e) => setPriority(e.target.value)}
                className="w-full mt-1 rounded-lg border border-border bg-background px-3 py-2 text-sm"
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
                <option value="critical">Critical</option>
              </select>
            </div>
          </div>
          <div>
            <label htmlFor="card-tags" className="text-sm font-medium">Tags (comma-separated)</label>
            <input
              id="card-tags"
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              className="w-full mt-1 rounded-lg border border-border bg-background px-3 py-2 text-sm"
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="card-assignee" className="text-sm font-medium">Assignee</label>
              <input
                id="card-assignee"
                type="text"
                value={assignee}
                onChange={(e) => setAssignee(e.target.value)}
                className="w-full mt-1 rounded-lg border border-border bg-background px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label htmlFor="card-due-date" className="text-sm font-medium">Due Date</label>
              <input
                id="card-due-date"
                type="text"
                value={dueDate}
                onChange={(e) => setDueDate(e.target.value)}
                className="w-full mt-1 rounded-lg border border-border bg-background px-3 py-2 text-sm"
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="card-sprint" className="text-sm font-medium">Sprint</label>
              <input
                id="card-sprint"
                type="text"
                value={sprint}
                onChange={(e) => setSprint(e.target.value)}
                className="w-full mt-1 rounded-lg border border-border bg-background px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label htmlFor="card-gh" className="text-sm font-medium">GitHub</label>
              <input
                id="card-gh"
                type="text"
                value={gh}
                onChange={(e) => setGh(e.target.value)}
                className="w-full mt-1 rounded-lg border border-border bg-background px-3 py-2 text-sm"
              />
            </div>
          </div>

          {/* Hash Tree */}
          {hashTree && (
            <div className="border border-border rounded-lg p-3">
              <button
                type="button"
                onClick={() => setShowHashTree(!showHashTree)}
                className="flex items-center justify-between w-full text-sm font-medium"
              >
                <span>Hash Tree</span>
                <span className="text-muted-foreground text-xs">{showHashTree ? '▲' : '▼'}</span>
              </button>
              {showHashTree && (
                <div className="mt-3 space-y-2">
                  <div>
                    <span className="text-xs text-muted-foreground">Root: </span>
                    <code className="text-xs font-mono break-all">{hashTree.root.root.slice(0, 16)}...</code>
                  </div>
                  <div>
                    <span className="text-xs text-muted-foreground">Tray: </span>
                    <span className="text-xs">{hashTree.root.tray}</span>
                  </div>
                  {hashTree.notes.length > 0 && (
                    <div>
                      <span className="text-xs text-muted-foreground">Notes: </span>
                      <span className="text-xs">{hashTree.notes.length} hashed</span>
                    </div>
                  )}
                  {hashTree.history.length > 0 && (
                    <div>
                      <span className="text-xs text-muted-foreground">History: </span>
                      <span className="text-xs">{hashTree.history.length} entries</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
          <div className="flex justify-between pt-4">
            <Button
              variant="destructive"
              size="sm"
              onClick={() => onDelete(card.id)}
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
