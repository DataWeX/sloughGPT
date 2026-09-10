'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, cn } from '@sloughgpt/strui'

interface ChecklistItem {
  id: string
  label: string
  description: string
  completed: boolean
  link?: string
}

const STORAGE_KEY = 'sloughgpt-onboarding-checklist'

const DEFAULT_ITEMS: Omit<ChecklistItem, 'completed'>[] = [
  { id: 'first-chat', label: 'Send your first message', description: 'Start a conversation in the Chat tab' },
  { id: 'add-knowledge', label: 'Add a knowledge entry', description: 'Tell me something about yourself in the Knowledge tab' },
  { id: 'customize-companion', label: 'Customize your companion', description: 'Adjust personality traits in the Companion tab' },
  { id: 'try-voice', label: 'Try voice input', description: 'Use the microphone button in Chat to speak' },
  { id: 'upload-file', label: 'Upload a file', description: 'Share a document in the Files tab' },
  { id: 'generate-image', label: 'Generate an image', description: 'Create art in the Images tab' },
  { id: 'check-analytics', label: 'View your analytics', description: 'See your usage stats in the Analytics tab' },
  { id: 'explore-settings', label: 'Explore settings', description: 'Tweak your preferences in Settings' },
]

function loadCompleted(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : {}
  } catch { return {} }
}

function saveCompleted(data: Record<string, boolean>) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
}

interface OnboardingChecklistCardProps {
  onNavigate?: (link: string) => void
}

export function OnboardingChecklistCard({ onNavigate }: OnboardingChecklistCardProps) {
  const [completed, setCompleted] = useState<Record<string, boolean>>(() => loadCompleted())

  const items: ChecklistItem[] = DEFAULT_ITEMS.map(item => ({
    ...item,
    completed: !!completed[item.id],
  }))

  const doneCount = items.filter(i => i.completed).length
  const allDone = doneCount === items.length

  const toggle = (id: string) => {
    const updated = { ...completed, [id]: !completed[id] }
    setCompleted(updated)
    saveCompleted(updated)
  }

  if (allDone) return null

  return (
    <Card data-testid="onboarding-checklist">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">
            Getting Started
            <span className="text-muted-foreground font-normal ml-2">({doneCount}/{items.length})</span>
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <div className="mb-3">
          <div className="h-1.5 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all"
              style={{ width: `${(doneCount / items.length) * 100}%` }}
            />
          </div>
        </div>
        <div className="space-y-1">
          {items.map(item => (
            <button
              key={item.id}
              className={cn(
                'w-full flex items-center gap-2 p-2 rounded text-left transition-colors',
                item.completed ? 'bg-success/5' : 'hover:bg-muted/50'
              )}
              onClick={() => toggle(item.id)}
              data-testid={`checklist-${item.id}`}
            >
              <div className={cn(
                'w-4 h-4 rounded border flex items-center justify-center text-[10px] shrink-0',
                item.completed ? 'bg-success border-success text-white' : 'border-muted-foreground/30'
              )}>
                {item.completed && '✓'}
              </div>
              <div className="min-w-0 flex-1">
                <div className={cn('text-xs font-medium', item.completed && 'line-through text-muted-foreground')}>
                  {item.label}
                </div>
                <div className="text-[10px] text-muted-foreground">{item.description}</div>
              </div>
              {item.link && !item.completed && (
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-[9px] shrink-0"
                  onClick={(e) => { e.stopPropagation(); onNavigate?.(item.link!) }}
                >
                  Go
                </Button>
              )}
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
