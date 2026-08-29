'use client'

import { useState, useCallback, useMemo, useEffect, memo } from 'react'
import { Button } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

interface PromptTemplate {
  id: string
  name: string
  content: string
  category: string
  isBuiltIn: boolean
}

interface SmartPromptSuggestionsProps {
  onSelect: (prompt: string) => void
  className?: string
}

const STORAGE_KEY = 'chat-prompt-templates'

const BUILT_IN_PROMPTS: PromptTemplate[] = [
  { id: 'explain', name: 'Explain', content: 'Explain this concept in simple terms:', category: 'Learning', isBuiltIn: true },
  { id: 'summarize', name: 'Summarize', content: 'Summarize the key points from:', category: 'Learning', isBuiltIn: true },
  { id: 'debug', name: 'Debug', content: 'Help me debug this code:', category: 'Code', isBuiltIn: true },
  { id: 'review', name: 'Code Review', content: 'Review this code and suggest improvements:', category: 'Code', isBuiltIn: true },
  { id: 'refactor', name: 'Refactor', content: 'Refactor this code to be cleaner and more efficient:', category: 'Code', isBuiltIn: true },
  { id: 'test', name: 'Write Tests', content: 'Write unit tests for:', category: 'Code', isBuiltIn: true },
  { id: 'translate', name: 'Translate', content: 'Translate this to English:', category: 'Language', isBuiltIn: true },
  { id: 'rewrite', name: 'Rewrite', content: 'Rewrite this more clearly:', category: 'Language', isBuiltIn: true },
  { id: 'pros_cons', name: 'Pros & Cons', content: 'What are the pros and cons of:', category: 'Analysis', isBuiltIn: true },
  { id: 'compare', name: 'Compare', content: 'Compare and contrast:', category: 'Analysis', isBuiltIn: true },
  { id: 'brainstorm', name: 'Brainstorm', content: 'Brainstorm ideas for:', category: 'Creative', isBuiltIn: true },
  { id: 'outline', name: 'Outline', content: 'Create an outline for:', category: 'Creative', isBuiltIn: true },
]

function loadTemplates(): PromptTemplate[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveTemplates(templates: PromptTemplate[]) {
  if (typeof window === 'undefined') return
  localStorage.setItem(STORAGE_KEY, JSON.stringify(templates))
}

export const SmartPromptSuggestions = memo(function SmartPromptSuggestions({
  onSelect,
  className,
}: SmartPromptSuggestionsProps) {
  const [userTemplates, setUserTemplates] = useState<PromptTemplate[]>([])
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [showAll, setShowAll] = useState(false)
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [newContent, setNewContent] = useState('')

  useEffect(() => {
    setUserTemplates(loadTemplates())
  }, [])

  const categories = useMemo(() => {
    const all = [...BUILT_IN_PROMPTS, ...userTemplates]
    const cats = [...new Set(all.map(p => p.category))]
    return cats.sort()
  }, [userTemplates])

  const prompts = useMemo(() => {
    const all = [...BUILT_IN_PROMPTS, ...userTemplates]
    let filtered = selectedCategory
      ? all.filter(p => p.category === selectedCategory)
      : all
    if (!showAll) {
      filtered = filtered.slice(0, 6)
    }
    return filtered
  }, [userTemplates, selectedCategory, showAll])

  const handleCreate = useCallback(() => {
    const trimmed = newName.trim()
    if (!trimmed || !newContent.trim()) return

    const template: PromptTemplate = {
      id: crypto.randomUUID(),
      name: trimmed,
      content: newContent.trim(),
      category: 'Custom',
      isBuiltIn: false,
    }

    const next = [...userTemplates, template]
    setUserTemplates(next)
    saveTemplates(next)
    setNewName('')
    setNewContent('')
    setCreating(false)
  }, [newName, newContent, userTemplates])

  const handleDelete = useCallback((id: string) => {
    const next = userTemplates.filter(t => t.id !== id)
    setUserTemplates(next)
    saveTemplates(next)
  }, [userTemplates])

  return (
    <div className={cn('border rounded-lg bg-card overflow-hidden', className)}>
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30">
        <span className="text-xs font-medium">Quick Prompts</span>
        <Button
          variant="ghost"
          size="sm"
          className="text-[10px] h-5"
          onClick={() => setCreating(!creating)}
        >
          {creating ? 'Cancel' : '+ Custom'}
        </Button>
      </div>

      {creating && (
        <div className="p-2 border-b space-y-2">
          <input
            type="text"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="Template name..."
            className="w-full text-xs bg-transparent border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary/50"
          />
          <textarea
            value={newContent}
            onChange={(e) => setNewContent(e.target.value)}
            placeholder="Prompt text..."
            className="w-full text-xs bg-transparent border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary/50 resize-none"
            rows={3}
          />
          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="sm"
              className="text-[10px] h-5"
              onClick={handleCreate}
              disabled={!newName.trim() || !newContent.trim()}
            >
              Save
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="text-[10px] h-5"
              onClick={() => setCreating(false)}
            >
              Cancel
            </Button>
          </div>
        </div>
      )}

      <div className="flex gap-1 p-2 flex-wrap border-b">
        <button
          type="button"
          onClick={() => setSelectedCategory(null)}
          className={cn(
            'text-[10px] px-2 py-0.5 rounded transition-colors',
            selectedCategory === null
              ? 'bg-primary/20 text-primary'
              : 'text-muted-foreground hover:bg-muted/50',
          )}
        >
          All
        </button>
        {categories.map(cat => (
          <button
            key={cat}
            type="button"
            onClick={() => setSelectedCategory(cat)}
            className={cn(
              'text-[10px] px-2 py-0.5 rounded transition-colors',
              selectedCategory === cat
                ? 'bg-primary/20 text-primary'
                : 'text-muted-foreground hover:bg-muted/50',
            )}
          >
            {cat}
          </button>
        ))}
      </div>

      <div className="p-2">
        <div className="grid grid-cols-2 gap-1">
          {prompts.map(prompt => (
            <div
              key={prompt.id}
              className="group flex items-center justify-between p-1.5 rounded hover:bg-muted/50 cursor-pointer"
              onClick={() => onSelect(prompt.content)}
            >
              <div className="min-w-0 flex-1">
                <span className="text-xs font-medium truncate block">{prompt.name}</span>
                <span className="text-[10px] text-muted-foreground truncate block">
                  {prompt.content.slice(0, 30)}...
                </span>
              </div>
              {!prompt.isBuiltIn && (
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="h-4 w-4 opacity-0 group-hover:opacity-100 shrink-0"
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDelete(prompt.id)
                  }}
                  title="Delete"
                >
                  <span className="text-destructive text-[10px]">×</span>
                </Button>
              )}
            </div>
          ))}
        </div>
      </div>

      {prompts.length === 0 && (
        <div className="px-3 py-4 text-center text-xs text-muted-foreground">
          No prompts in this category
        </div>
      )}

      {!showAll && prompts.length >= 6 && (
        <div className="px-3 pb-2 text-center">
          <Button
            variant="ghost"
            size="sm"
            className="text-[10px] h-5"
            onClick={() => setShowAll(true)}
          >
            Show All
          </Button>
        </div>
      )}

      {showAll && (
        <div className="px-3 pb-2 text-center">
          <Button
            variant="ghost"
            size="sm"
            className="text-[10px] h-5"
            onClick={() => setShowAll(false)}
          >
            Show Less
          </Button>
        </div>
      )}
    </div>
  )
})