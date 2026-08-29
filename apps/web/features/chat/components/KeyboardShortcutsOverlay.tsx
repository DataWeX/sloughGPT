'use client'

import { memo } from 'react'
import { Button, IconX } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

interface KeyboardShortcutsOverlayProps {
  onClose: () => void
  className?: string
}

interface Shortcut {
  keys: string[]
  label: string
  category: string
}

const shortcuts: Shortcut[] = [
  { keys: ['Ctrl', 'N'], label: 'New chat', category: 'General' },
  { keys: ['Ctrl', 'B'], label: 'Toggle tools panel', category: 'General' },
  { keys: ['Ctrl', '\\'], label: 'Toggle sidebar', category: 'General' },
  { keys: ['Ctrl', '?'], label: 'Show shortcuts', category: 'General' },
  { keys: ['Escape'], label: 'Cancel stream / dismiss error', category: 'General' },
  { keys: ['Ctrl', 'R'], label: 'Regenerate response', category: 'Chat' },
  { keys: ['Ctrl', 'Shift', 'R'], label: 'Rename conversation', category: 'Chat' },
  { keys: ['Ctrl', 'Shift', 'E'], label: 'Export as Markdown', category: 'Chat' },
  { keys: ['Ctrl', 'Shift', 'D'], label: 'Duplicate conversation', category: 'Chat' },
  { keys: ['Ctrl', 'Shift', 'B'], label: 'Toggle bookmarks', category: 'Chat' },
  { keys: ['Ctrl', 'Shift', 'N'], label: 'Add note to last message', category: 'Chat' },
  { keys: ['/'], label: 'Focus search', category: 'Search' },
  { keys: ['Ctrl', 'F'], label: 'Focus message search', category: 'Search' },
  { keys: ['Ctrl', 'Shift', 'F'], label: 'Search conversations', category: 'Search' },
  { keys: ['Ctrl', 'Y'], label: 'Approve tool call', category: 'Tools' },
  { keys: ['Ctrl', 'N'], label: 'Deny tool call', category: 'Tools' },
]

const categories = [...new Set(shortcuts.map(s => s.category))]

export const KeyboardShortcutsOverlay = memo(function KeyboardShortcutsOverlay({
  onClose,
  className,
}: KeyboardShortcutsOverlayProps) {
  return (
    <div className={cn('fixed inset-0 z-50 flex items-center justify-center bg-black/50', className)}>
      <div className="bg-background border rounded-lg shadow-xl w-[480px] max-h-[80vh] overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 border-b">
          <span className="text-sm font-medium">Keyboard Shortcuts</span>
          <Button variant="ghost" size="icon-sm" className="h-6 w-6" onClick={onClose} aria-label="Close shortcuts">
            <IconX className="h-3.5 w-3.5" />
          </Button>
        </div>

        <div className="overflow-y-auto p-4 space-y-4">
          {categories.map(cat => (
            <div key={cat}>
              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                {cat}
              </h3>
              <div className="space-y-1">
                {shortcuts.filter(s => s.category === cat).map((s, i) => (
                  <div key={i} className="flex items-center justify-between py-1">
                    <span className="text-xs">{s.label}</span>
                    <div className="flex items-center gap-0.5">
                      {s.keys.map((key, ki) => (
                        <span key={ki}>
                          <kbd className="px-1.5 py-0.5 text-[10px] font-mono bg-muted border rounded">
                            {key}
                          </kbd>
                          {ki < s.keys.length - 1 && (
                            <span className="text-muted-foreground text-[10px] mx-0.5">+</span>
                          )}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
})