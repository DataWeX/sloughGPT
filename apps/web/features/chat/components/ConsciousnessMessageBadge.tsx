'use client'
import { useState, useEffect } from 'react'
import { getQualiaMood } from '@/hooks/useConsciousnessStatus'
import { onConsciousness, type ConsciousnessEvent } from '@/lib/consciousness-bus'
import { cn } from '@sloughgpt/strui'

interface Props {
  messageId: string
}

export function ConsciousnessMessageBadge({ messageId }: Props) {
  const [event, setEvent] = useState<ConsciousnessEvent | null>(null)

  useEffect(() => {
    const unsub = onConsciousness((e) => {
      if (e.messageId === messageId) setEvent(e)
    })
    return unsub
  }, [messageId])

  if (!event) return null

  const mood = getQualiaMood(event.qualia as any || {})
  const growth = (event as any).growth ?? event.growth_delta ?? 0

  return (
    <span className="inline-flex items-center gap-1 text-[10px] text-muted-foreground">
      {mood && <span className="text-violet-400">{mood}</span>}
      {growth !== 0 && (
        <span className={cn('font-mono', growth > 0 ? 'text-green-400' : 'text-red-400')}>
          {growth > 0 ? '+' : ''}{growth.toFixed(3)}
        </span>
      )}
      <span className="text-muted-foreground/50">L{event.level ?? 0}</span>
    </span>
  )
}
