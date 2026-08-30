'use client'

import { useState, useEffect } from 'react'
import { cn, Button } from '@sloughgpt/strui'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@sloughgpt/strui'
import { NAV_SECTIONS } from '@/lib/navigation'

interface Shortcut {
  keys: string[]
  description: string
  category: string
}

/** Build navigation shortcuts from shared config */
function buildNavShortcuts(): Shortcut[] {
  const shortcuts: Shortcut[] = []
  for (const section of NAV_SECTIONS) {
    for (const route of section.routes) {
      if (!route.shortcut) continue
      const keys = route.shortcut === 'shift+A'
        ? ['Ctrl', 'Shift', 'A']
        : ['Ctrl', route.shortcut]
      shortcuts.push({
        keys,
        description: route.description || route.path,
        category: 'Navigation',
      })
    }
  }
  return shortcuts
}

const SHORTCUTS: Shortcut[] = [
  // Chat shortcuts
  { keys: ['Enter'], description: 'Send message', category: 'Chat' },
  { keys: ['Shift', 'Enter'], description: 'New line in message', category: 'Chat' },
  { keys: ['Esc'], description: 'Stop streaming / Close dialog', category: 'Chat' },
  { keys: ['Ctrl', 'N'], description: 'New chat', category: 'Chat' },
  { keys: ['Ctrl', 'Shift', 'C'], description: 'Copy last response', category: 'Chat' },
  { keys: ['Ctrl', 'Shift', 'F'], description: 'Search conversations', category: 'Chat' },
  { keys: ['Ctrl', 'Shift', 'R'], description: 'Rename conversation', category: 'Chat' },
  { keys: ['Ctrl', 'Shift', 'E'], description: 'Export as Markdown', category: 'Chat' },
  { keys: ['Ctrl', 'Shift', 'D'], description: 'Duplicate conversation', category: 'Chat' },
  { keys: ['Ctrl', 'Shift', 'B'], description: 'Toggle bookmarks panel', category: 'Chat' },

  // General shortcuts
  { keys: ['?'], description: 'Show keyboard shortcuts', category: 'General' },
  { keys: ['Ctrl', 'K'], description: 'Command palette', category: 'General' },

  // Navigation shortcuts (from shared config)
  ...buildNavShortcuts(),

  // Sidebar shortcuts
  { keys: ['Ctrl', '\\'], description: 'Toggle navigation sidebar', category: 'Sidebar' },
  { keys: ['Ctrl', 'Shift', '\\'], description: 'Toggle conversation sidebar', category: 'Sidebar' },

  // Training shortcuts
  { keys: ['Ctrl', 'Enter'], description: 'Start training', category: 'Training' },
  { keys: ['T'], description: 'Switch to Train tab', category: 'Training' },
  { keys: ['H'], description: 'Switch to History tab', category: 'Training' },
  { keys: ['E'], description: 'Switch to Eval tab', category: 'Training' },
  { keys: ['Ctrl', 'Shift', 'T'], description: 'Open Test Model dialog', category: 'Training' },
]

function KeyboardKey({ k, 'aria-describedby': ariaDescribedBy }: { k: string; 'aria-describedby'?: string }) {
  const isModifier = ['Ctrl', 'Shift', 'Alt', 'Meta', 'Cmd'].includes(k)

  return (
    <kbd
      className={cn(
        "inline-flex h-5 min-w-[1.25rem] items-center justify-center rounded-md border border-border/60 bg-muted/50 px-1.5 py-0.5 text-xs font-medium text-muted-foreground shadow-sm",
        isModifier && "bg-muted/70"
      )}
      aria-describedby={ariaDescribedBy}
    >
      {k}
    </kbd>
  )
}

interface KeyboardShortcutsDialogProps {
  open?: boolean
  onOpenChange?: (open: boolean) => void
}

export function KeyboardShortcutsDialog({ open, onOpenChange }: KeyboardShortcutsDialogProps) {
  const [isOpen, setIsOpen] = useState(open ?? false)

  useEffect(() => {
    if (open !== undefined) {
      setIsOpen(open)
      if (open) {
        document.body.dataset.shortcutsOpen = 'true'
      } else {
        delete document.body.dataset.shortcutsOpen
      }
    }
    return () => {
      delete document.body.dataset.shortcutsOpen
    }
  }, [open])

  const handleOpenChange = (newOpen: boolean) => {
    setIsOpen(newOpen)
    if (newOpen) {
      document.body.dataset.shortcutsOpen = 'true'
    } else {
      delete document.body.dataset.shortcutsOpen
    }
    onOpenChange?.(newOpen)
  }

  // Group shortcuts by category
  const categories = SHORTCUTS.reduce((acc, shortcut) => {
    if (!acc[shortcut.category]) {
      acc[shortcut.category] = []
    }
    acc[shortcut.category].push(shortcut)
    return acc
  }, {} as Record<string, Shortcut[]>)

  return (
    <Dialog open={isOpen} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Keyboard Shortcuts</DialogTitle>
          <DialogDescription>
            Quick keyboard shortcuts to navigate and use the app faster.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          {Object.entries(categories).map(([category, shortcuts]) => (
            <fieldset key={category}>
              <legend className="mb-2 text-sm font-medium text-muted-foreground">{category}</legend>
              <div className="overflow-x-auto">
              <table className="w-full" aria-label="Keyboard shortcuts">
                <thead>
                  <tr className="text-left text-xs text-muted-foreground border-b border-border/30">
                    <th scope="col" className="py-2 pr-4 font-medium">Action</th>
                    <th scope="col" className="py-2 text-right font-medium">Shortcut</th>
                  </tr>
                </thead>
                <tbody>
                  {shortcuts.map((shortcut, index) => (
                    <tr key={index} className="border-b border-border/30 last:border-0">
                      <td className="py-2 pr-4 text-sm">
                        <span id={`shortcut-desc-${index}`}>{shortcut.description}</span>
                      </td>
                      <td className="py-2 text-right">
                        <div className="flex items-center justify-end gap-1" role="group" aria-label={shortcut.description}>
                          {shortcut.keys.map((keyName, keyIndex) => (
                            <span key={keyIndex} className="flex items-center gap-0.5">
                              <KeyboardKey k={keyName} aria-describedby={`shortcut-desc-${index}`} />
                              {keyIndex < shortcut.keys.length - 1 && (
                                <span className="text-muted-foreground/50 text-xs">+</span>
                              )}
                            </span>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              </div>
            </fieldset>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}

// Compact badge for showing "Press ? for shortcuts"
export function ShortcutsHint() {
  const [showModal, setShowModal] = useState(false)

  return (
    <>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setShowModal(true)}
        className="flex items-center gap-1 text-xs"
        title="Keyboard shortcuts"
      >
        <kbd className="rounded border border-border/60 bg-muted/50 px-1.5 py-0.5 text-xs font-medium">?</kbd>
        <span className="hidden sm:inline">Shortcuts</span>
      </Button>
      <KeyboardShortcutsDialog open={showModal} onOpenChange={setShowModal} />
    </>
  )
}

// Hook to show modal with ? key
export function useKeyboardShortcuts() {
  const [showModal, setShowModal] = useState(false)

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Show shortcuts when pressing ?
      if (e.key === '?' && !e.ctrlKey && !e.metaKey) {
        const target = e.target as HTMLElement
        if (target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA') {
          e.preventDefault()
          setShowModal(true)
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  return { showModal, setShowModal }
}
