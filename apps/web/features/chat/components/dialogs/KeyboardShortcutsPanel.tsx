'use client'

import { memo } from 'react'
import { cn } from '@sloughgpt/strui'
import { IconX, IconCode } from '@sloughgpt/strui'

interface Shortcut {
  keys: string[]
  description: string
}

const shortcuts: Shortcut[] = [
  { keys: ['Ctrl', 'N'], description: 'New chat' },
  { keys: ['Ctrl', 'R'], description: 'Regenerate last response' },
  { keys: ['Ctrl', 'F'], description: 'Focus search' },
  { keys: ['Ctrl', 'Shift', 'F'], description: 'Search notes across conversations' },
  { keys: ['Ctrl', '/'], description: 'Focus search' },
  { keys: ['Ctrl', '\\'], description: 'Toggle sidebar' },
  { keys: ['Ctrl', 'Shift', 'N'], description: 'Add note to last message' },
  { keys: ['Ctrl', 'Shift', 'R'], description: 'Rename conversation' },
  { keys: ['Ctrl', 'Shift', 'E'], description: 'Export conversation as Markdown' },
  { keys: ['Ctrl', 'Shift', 'D'], description: 'Duplicate conversation' },
  { keys: ['Ctrl', 'Y'], description: 'Approve tool call' },
  { keys: ['Escape'], description: 'Cancel stream / close panel' },
  { keys: ['?'], description: 'Toggle settings' },
  { keys: ['/'], description: 'Focus search (when not in input)' },
]

function formatKeys(keys: string[]): string {
  return keys.map(k => {
    if (k === 'Ctrl') return '⌘/Ctrl'
    if (k === 'Shift') return '⇧'
    if (k === 'Escape') return 'Esc'
    return k
  }).join(' + ')
}

interface KeyboardShortcutsPanelProps {
  open: boolean
  onClose: () => void
}

export const KeyboardShortcutsPanel = memo(function KeyboardShortcutsPanel({
  open,
  onClose,
}: KeyboardShortcutsPanelProps) {
  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div
        className="bg-background border border-border rounded-lg shadow-xl w-[420px] max-h-[80vh] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <div className="flex items-center gap-2">
            <IconCode className="h-4 w-4 text-primary" />
            <h2 className="text-sm font-medium">Keyboard Shortcuts</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded-md hover:bg-muted transition-colors"
            aria-label="Close"
          >
            <IconX className="h-4 w-4 text-muted-foreground" />
          </button>
        </div>

        <div className="p-4 overflow-y-auto max-h-[calc(80vh-52px)]">
          <div className="space-y-2">
            {shortcuts.map((shortcut, i) => (
              <div
                key={i}
                className="flex items-center justify-between py-1.5 border-b border-border/40 last:border-0"
              >
                <span className="text-sm text-foreground/80">{shortcut.description}</span>
                <div className="flex items-center gap-1">
                  {shortcut.keys.map((key, j) => (
                    <span key={j}>
                      <kbd className={cn(
                        "px-1.5 py-0.5 text-[10px] font-mono font-medium rounded border",
                        "bg-muted/50 text-muted-foreground border-border/60"
                      )}>
                        {formatKeys([key])}
                      </kbd>
                      {j < shortcut.keys.length - 1 && (
                        <span className="text-muted-foreground/40 mx-0.5">+</span>
                      )}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
})
