'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { whatsNewItems, type WhatsNewItem } from '@/lib/whats-new-data'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/tags'
import { cn } from '@/lib/cn'

const SEEN_KEY = 'whatsnew_seen'

function getSeenIds(): Set<string> {
  if (typeof window === 'undefined') return new Set()
  try {
    const raw = localStorage.getItem(SEEN_KEY)
    return new Set(raw ? JSON.parse(raw) : [])
  } catch {
    return new Set()
  }
}

export function getUnseenCount(): number {
  const seen = getSeenIds()
  return whatsNewItems.filter(i => !seen.has(i.id)).length
}

export function markAllSeen() {
  try {
    localStorage.setItem(SEEN_KEY, JSON.stringify(whatsNewItems.map(i => i.id)))
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('whatsnew-updated'))
    }
  } catch {}
}

export function getLatestSeenDate(): string | null {
  const seen = getSeenIds()
  if (seen.size === 0) return null
  const unseen = whatsNewItems.filter(i => !seen.has(i.id))
  if (unseen.length > 0) return null
  const maxId = Math.max(...[...seen].map(id => {
    const item = whatsNewItems.find(i => i.id === id)
    return item ? new Date(item.date).getTime() : 0
  }))
  const last = whatsNewItems.find(i => new Date(i.date).getTime() === maxId)
  return last?.date || null
}

export function WhatsNewDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  const [seen, setSeen] = useState(getSeenIds())
  const markStartedRef = useRef(false)

  useEffect(() => {
    if (!open) {
      markStartedRef.current = false
      return
    }
    if (markStartedRef.current) return
    const unseen = whatsNewItems.some(i => !seen.has(i.id))
    if (!unseen) return
    markStartedRef.current = true
    markAllSeen()
    setSeen(new Set(whatsNewItems.map(i => i.id)))
  }, [open, seen])

  const handleOpen = (v: boolean) => {
    if (v) {
      markAllSeen()
      setSeen(new Set(whatsNewItems.map(i => i.id)))
    }
    onOpenChange(v)
  }

  return (
    <Dialog open={open} onOpenChange={handleOpen}>
      <DialogContent className="max-w-lg max-h-[80vh]">
        <DialogHeader>
          <DialogTitle>What&apos;s New</DialogTitle>
          <DialogDescription>Recent features and improvements across builds.</DialogDescription>
        </DialogHeader>

        <div className="space-y-2 overflow-y-auto pr-1 custom-scrollbar">
          {whatsNewItems.map(item => {
            const isUnseen = !seen.has(item.id)
            return (
              <div
                key={item.id}
                className={cn(
                  "rounded-lg border p-3 transition-colors",
                  isUnseen ? "border-primary/30 bg-primary/[0.03]" : "border-border/40 bg-background"
                )}
              >
                <div className="flex items-start gap-2.5">
                  <span className="text-lg leading-none mt-0.5 shrink-0">{item.icon}</span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      {item.href ? (
                        <Link href={item.href} onClick={() => onOpenChange(false)} className="text-sm font-medium hover:underline">
                          {item.title}
                        </Link>
                      ) : (
                        <span className="text-sm font-medium">{item.title}</span>
                      )}
                      <span className="text-[10px] text-muted-foreground">{item.date}</span>
                      {isUnseen && (
                        <span className="text-[9px] font-medium text-primary bg-primary/10 rounded-full px-1.5 py-0.5">NEW</span>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed">{item.description}</p>
                    {item.tags && item.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1.5">
                        {item.tags.map(tag => (
                          <Badge key={tag} label={tag} variant="default" size="sm" />
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )
          })}
        </div>

        <div className="flex items-center justify-between pt-2 border-t border-border/30">
          <span className="text-xs text-muted-foreground">{whatsNewItems.length} entries</span>
          <Button size="sm" variant="outline" onClick={() => onOpenChange(false)}>Close</Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
