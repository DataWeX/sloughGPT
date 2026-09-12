'use client'

import { useState, useEffect, memo } from 'react'
import { cn } from '@sloughgpt/strui'
import { IconBrain } from '@sloughgpt/strui'
import { onConsciousness, type ConsciousnessEvent } from '@/lib/consciousness-bus'
import { useLocale } from '@/hooks/useLocale'

interface ConsciousnessToggleProps {
  open: boolean
  onToggle: () => void
}

export const ConsciousnessToggle = memo(function ConsciousnessToggle({ open, onToggle }: ConsciousnessToggleProps) {
  const { t } = useLocale()
  const [level, setLevel] = useState<number | null>(null)

  useEffect(() => {
    const unsub = onConsciousness((e: ConsciousnessEvent) => {
      setLevel(e.level)
    })
    return unsub
  }, [])

  return (
    <button
      type="button"
      onClick={onToggle}
      className={cn(
        'flex items-center gap-1.5 h-7 px-2.5 rounded-lg text-xs transition-colors',
        open
          ? 'bg-primary/15 text-primary'
          : 'text-muted-foreground hover:bg-muted/60 hover:text-foreground',
      )}
      aria-label={t('consciousness_chat.toggle')}
      title={t('consciousness_chat.toggle')}
    >
      <IconBrain className="h-3.5 w-3.5" aria-hidden="true" />
      {level !== null && (
        <span className="text-[9px] font-mono px-1 py-0.5 rounded bg-primary/15 text-primary leading-none">
          L{level}
        </span>
      )}
    </button>
  )
})
