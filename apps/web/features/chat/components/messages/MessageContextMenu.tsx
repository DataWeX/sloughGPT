'use client'

import { useState, useEffect, useCallback, useRef, memo } from 'react'
import { cn } from '@sloughgpt/strui'
import { IconCopy, IconCheck, IconRefresh, IconEdit, IconStar, IconTrash, IconPin } from '@sloughgpt/strui'

interface MessageContextMenuProps {
  messageId: string
  content: string
  role: 'user' | 'assistant'
  isBookmarked?: boolean
  onCopy?: (text: string) => void
  onEdit?: (messageId: string) => void
  onBookmark?: (messageId: string) => void
  onRegenerate?: (messageId: string) => void
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

function simpleMarkdownToHtml(md: string): string {
  return md
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`)
    .replace(/\n{2,}/g, '</p><p>')
    .replace(/\n/g, '<br>')
    .replace(/^/, '<p>')
    .replace(/$/, '</p>')
}

export const MessageContextMenu = memo(function MessageContextMenu({
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
    ...(role === 'assistant' ? [{
      label: 'Copy as HTML',
      icon: <IconCopy className="h-3.5 w-3.5" />,
      onClick: async () => {
        try {
          const html = simpleMarkdownToHtml(content)
          await navigator.clipboard.write([new ClipboardItem({ 'text/html': new Blob([html], { type: 'text/html' }) })])
          setOpen(false)
        } catch { /* clipboard unavailable */ }
      },
    }] : []),
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
      onClick: () => { onRegenerate(messageId); setOpen(false) },
    }] : []),
    ...(onSaveToKnowledge ? [{
      label: 'Save to knowledge',
      icon: <IconPin className="h-3.5 w-3.5" />,
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
          onKeyDown={(e) => {
            const menuItems = menuRef.current?.querySelectorAll('[role="menuitem"]:not([disabled])')
            if (!menuItems?.length) return
            const currentIndex = Array.from(menuItems).indexOf(document.activeElement as Element)
            if (e.key === 'ArrowDown') {
              e.preventDefault()
              const next = currentIndex < menuItems.length - 1 ? currentIndex + 1 : 0
              ;(menuItems[next] as HTMLElement).focus()
            } else if (e.key === 'ArrowUp') {
              e.preventDefault()
              const prev = currentIndex > 0 ? currentIndex - 1 : menuItems.length - 1
              ;(menuItems[prev] as HTMLElement).focus()
            } else if (e.key === 'Escape') {
              e.preventDefault()
              setOpen(false)
            }
          }}
        >
          {items.map((item, idx) => (
            <button
              key={idx}
              type="button"
              role="menuitem"
              tabIndex={0}
              disabled={item.disabled}
              onClick={item.onClick}
              className={cn(
                "w-full flex items-center gap-2.5 px-3 py-2 text-sm text-left transition-colors",
                "hover:bg-muted/50 focus:bg-muted/50 focus:outline-none disabled:opacity-40 disabled:pointer-events-none",
                item.variant === 'destructive' && "text-destructive hover:bg-destructive/10 focus:bg-destructive/10"
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
})
