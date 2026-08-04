'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { cn } from '@sloughgpt/strui'
import { IconCopy, IconCheck, IconRefresh, IconEdit, IconStar, IconTrash } from '@sloughgpt/strui'

interface MessageContextMenuProps {
  messageId: string
  content: string
  role: 'user' | 'assistant'
  isBookmarked?: boolean
  onCopy?: (text: string) => void
  onEdit?: (messageId: string) => void
  onBookmark?: (messageId: string) => void
  onRegenerate?: () => void
  onDelete?: (messageId: string) => void
  onSaveToKnowledge?: (messageId: string, content: string) => void
  children: React.ReactNode
}

interface MenuItem {
  label: string
  icon: React.ReactNode
  onClick: () => void
  variant?: 'default' | 'destructive'
  disabled?: boolean
}

export function MessageContextMenu({
  messageId,
  content,
  role,
  isBookmarked,
  onCopy,
  onEdit,
  onBookmark,
  onRegenerate,
  onDelete,
  onSaveToKnowledge,
  children,
}: MessageContextMenuProps) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState({ x: 0, y: 0 })
  const [copied, setCopied] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    const x = Math.min(e.clientX, window.innerWidth - 200)
    const y = Math.min(e.clientY, window.innerHeight - 280)
    setPos({ x, y })
    setOpen(true)
    setCopied(false)
  }, [])

  useEffect(() => {
    if (!open) return
    const close = () => setOpen(false)
    const handleKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    document.addEventListener('keydown', handleKey)
    document.addEventListener('click', close)
    return () => {
      document.removeEventListener('keydown', handleKey)
      document.removeEventListener('click', close)
    }
  }, [open])

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      onCopy?.(content)
      setTimeout(() => setOpen(false), 400)
    } catch { /* clipboard unavailable */ }
  }, [content, onCopy])

  const items: MenuItem[] = [
    {
      label: copied ? 'Copied!' : 'Copy',
      icon: copied ? <IconCheck className="h-3.5 w-3.5" /> : <IconCopy className="h-3.5 w-3.5" />,
      onClick: handleCopy,
    },
    ...(role === 'user' && onEdit ? [{
      label: 'Edit',
      icon: <IconEdit className="h-3.5 w-3.5" />,
      onClick: () => { onEdit(messageId); setOpen(false) },
    }] : []),
    ...(onBookmark ? [{
      label: isBookmarked ? 'Remove bookmark' : 'Bookmark',
      icon: <IconStar className={cn('h-3.5 w-3.5', isBookmarked && 'fill-current')} />,
      onClick: () => { onBookmark(messageId); setOpen(false) },
    }] : []),
    ...(role === 'assistant' && onRegenerate ? [{
      label: 'Regenerate',
      icon: <IconRefresh className="h-3.5 w-3.5" />,
      onClick: () => { onRegenerate(); setOpen(false) },
    }] : []),
    ...(onSaveToKnowledge ? [{
      label: 'Save to knowledge',
      icon: (
        <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2a7 7 0 0 0-7 7c0 5 7 13 7 13s7-8 7-13a7 7 0 0 0-7-7z" />
        </svg>
      ),
      onClick: () => { onSaveToKnowledge(messageId, content); setOpen(false) },
    }] : []),
    ...(onDelete ? [{
      label: 'Delete',
      icon: <IconTrash className="h-3.5 w-3.5" />,
      onClick: () => { onDelete(messageId); setOpen(false) },
      variant: 'destructive' as const,
    }] : []),
  ]

  return (
    <>
      <div onContextMenu={handleContextMenu}>{children}</div>
      {open && (
        <div
          ref={menuRef}
          className="fixed z-[200] min-w-[160px] rounded-lg border border-border/50 bg-popover shadow-xl py-1 animate-in fade-in zoom-in-95 duration-100"
          style={{ left: pos.x, top: pos.y }}
          role="menu"
          aria-label="Message actions"
        >
          {items.map((item) => (
            <button
              key={item.label}
              role="menuitem"
              disabled={item.disabled}
              onClick={item.onClick}
              className={cn(
                "w-full flex items-center gap-2.5 px-3 py-2 text-sm text-left transition-colors",
                "hover:bg-muted/50 disabled:opacity-40 disabled:pointer-events-none",
                item.variant === 'destructive' && "text-destructive hover:bg-destructive/10"
              )}
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          ))}
        </div>
      )}
    </>
  )
}
