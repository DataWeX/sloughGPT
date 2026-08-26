'use client'

import { useState, useEffect, useRef, memo } from 'react'
import { cn } from '@sloughgpt/strui'
import { IconX, IconPlus, IconTrash, IconCheck } from '@sloughgpt/strui'

interface Template {
  id: string
  name: string
  content: string
  createdAt: number
}

interface TemplateDialogProps {
  open: boolean
  onClose: () => void
  onSelect?: (content: string) => void
}

const STORAGE_KEY = 'chat-templates'

function loadTemplates(): Template[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch { return [] }
}

function saveTemplates(templates: Template[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(templates))
}

export const TemplateDialog = memo(function TemplateDialog({
  open,
  onClose,
  onSelect,
}: TemplateDialogProps) {
  const [templates, setTemplates] = useState<Template[]>([])
  const [editingId, setEditingId] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [content, setContent] = useState('')
  const nameRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) setTemplates(loadTemplates())
  }, [open])

  useEffect(() => {
    if (editingId === null && open) {
      setTimeout(() => nameRef.current?.focus(), 100)
    }
  }, [editingId, open])

  const handleSave = () => {
    if (!name.trim() || !content.trim()) return
    const newTemplate: Template = {
      id: editingId || `tpl-${Date.now()}`,
      name: name.trim(),
      content: content.trim(),
      createdAt: Date.now(),
    }
    const updated = editingId
      ? templates.map(t => t.id === editingId ? newTemplate : t)
      : [...templates, newTemplate]
    setTemplates(updated)
    saveTemplates(updated)
    setName('')
    setContent('')
    setEditingId(null)
  }

  const handleEdit = (template: Template) => {
    setEditingId(template.id)
    setName(template.name)
    setContent(template.content)
  }

  const handleDelete = (id: string) => {
    const updated = templates.filter(t => t.id !== id)
    setTemplates(updated)
    saveTemplates(updated)
  }

  const handleUse = (template: Template) => {
    onSelect?.(template.content)
    onClose()
  }

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div
        className="bg-background border border-border rounded-lg shadow-xl w-[480px] max-h-[80vh] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <h2 className="text-sm font-medium">Conversation Templates</h2>
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded-md hover:bg-muted transition-colors"
            aria-label="Close"
          >
            <IconX className="h-4 w-4 text-muted-foreground" />
          </button>
        </div>

        <div className="p-4 border-b border-border">
          <div className="space-y-2">
            <input
              ref={nameRef}
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Template name"
              className="w-full px-3 py-1.5 text-sm rounded-md border border-border bg-background focus:outline-none focus:ring-1 focus:ring-primary/50"
            />
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="Template content (prompt, system message, etc.)"
              rows={3}
              className="w-full px-3 py-1.5 text-sm rounded-md border border-border bg-background focus:outline-none focus:ring-1 focus:ring-primary/50 resize-none"
            />
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">
                {editingId ? 'Editing template' : `${templates.length} templates saved`}
              </span>
              <div className="flex items-center gap-2">
                {editingId && (
                  <button
                    type="button"
                    onClick={() => { setEditingId(null); setName(''); setContent('') }}
                    className="px-2 py-1 text-xs rounded-md hover:bg-muted transition-colors"
                  >
                    Cancel
                  </button>
                )}
                <button
                  type="button"
                  onClick={handleSave}
                  disabled={!name.trim() || !content.trim()}
                  className="px-2 py-1 text-xs rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
                >
                  {editingId ? 'Update' : 'Save'}
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="p-4 overflow-y-auto max-h-[calc(80vh-200px)]">
          {templates.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-4">
              No templates yet. Create one above.
            </p>
          ) : (
            <div className="space-y-2">
              {templates.map((template) => (
                <div
                  key={template.id}
                  className={cn(
                    "group p-3 rounded-md border border-border/40 hover:border-border transition-colors",
                    editingId === template.id && "border-primary/40 bg-primary/5"
                  )}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{template.name}</p>
                      <p className="text-xs text-muted-foreground line-clamp-2 mt-0.5">
                        {template.content}
                      </p>
                    </div>
                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      {onSelect && (
                        <button
                          type="button"
                          onClick={() => handleUse(template)}
                          className="p-1 rounded hover:bg-primary/10 text-primary"
                          title="Use template"
                        >
                          <IconCheck className="h-3.5 w-3.5" />
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => handleEdit(template)}
                        className="p-1 rounded hover:bg-muted text-muted-foreground"
                        title="Edit template"
                      >
                        <IconPlus className="h-3.5 w-3.5 rotate-45" />
                      </button>
                      <button
                        type="button"
                        onClick={() => handleDelete(template.id)}
                        className="p-1 rounded hover:bg-destructive/10 text-destructive"
                        title="Delete template"
                      >
                        <IconTrash className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
})
