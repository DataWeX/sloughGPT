'use client'

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { cn } from '@/lib/cn'

interface Shortcut {
  keys: string[]
  description: string
  category: string
}

const SHORTCUTS: Shortcut[] = [
  // Chat shortcuts
  { keys: ['Enter'], description: 'Send message', category: 'Chat' },
  { keys: ['Shift', 'Enter'], description: 'New line in message', category: 'Chat' },
  { keys: ['Esc'], description: 'Stop streaming / Close dialog', category: 'Chat' },
  { keys: ['Ctrl', 'N'], description: 'New chat', category: 'Chat' },
  { keys: ['Ctrl', 'R'], description: 'Regenerate response', category: 'Chat' },
  { keys: ['Ctrl', 'K'], description: 'Toggle settings', category: 'Chat' },
  
  // General shortcuts
  { keys: ['?'], description: 'Show keyboard shortcuts', category: 'General' },
  { keys: ['Ctrl', 'S'], description: 'Save / Export', category: 'General' },
  
  // Navigation shortcuts
  { keys: ['Ctrl', '1'], description: 'Go to Chat', category: 'Navigation' },
  { keys: ['Ctrl', '2'], description: 'Go to Models', category: 'Navigation' },
  { keys: ['Ctrl', '3'], description: 'Go to Datasets', category: 'Navigation' },
  { keys: ['Ctrl', '4'], description: 'Go to Training', category: 'Navigation' },
  { keys: ['Ctrl', '5'], description: 'Go to Monitoring', category: 'Navigation' },
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

interface KeyboardShortcutsModalProps {
  open?: boolean
  onOpenChange?: (open: boolean) => void
}

export function KeyboardShortcutsModal({ open, onOpenChange }: KeyboardShortcutsModalProps) {
  const [isOpen, setIsOpen] = useState(open ?? false)
  
  useEffect(() => {
    if (open !== undefined) {
      setIsOpen(open)
      if (typeof window !== 'undefined') {
        if (open) {
          document.body.dataset.shortcutsOpen = 'true'
        } else {
          delete document.body.dataset.shortcutsOpen
        }
      }
    }
  }, [open])
  
  const handleOpenChange = (newOpen: boolean) => {
    setIsOpen(newOpen)
    if (typeof window !== 'undefined') {
      if (newOpen) {
        document.body.dataset.shortcutsOpen = 'true'
      } else {
        delete document.body.dataset.shortcutsOpen
      }
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
              <table className="w-full">
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
      <KeyboardShortcutsModal open={showModal} onOpenChange={setShowModal} />
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