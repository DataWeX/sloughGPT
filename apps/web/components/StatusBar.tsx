'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useApiHealth } from '@/hooks/useApiHealth'
import { soulsController } from '@/lib/souls-controller'
import { cn } from '@/lib/cn'

export function StatusBar() {
  const { state: health } = useApiHealth()
  const [soulName, setSoulName] = useState<string | null>(null)

  useEffect(() => {
    if (health === null || health === 'offline') return
    soulsController.getCurrent().then(s => {
      if (s && 'name' in s) setSoulName((s as any).name)
    }).catch(() => {})
  }, [health])

  const dot = health === null ? 'bg-muted-foreground/50' :
    health === 'offline' ? 'bg-destructive' :
    health.model_loaded ? 'bg-success' : 'bg-warning'

  const label = health === null ? 'Connecting...' :
    health === 'offline' ? 'Offline' :
    health.model_loaded ? health.model_type : 'No model'

  return (
    <Link href="/monitoring" className="flex shrink-0 h-6 sm:h-7 items-center justify-between px-2 sm:px-3 border-t border-border/30 bg-muted/20 text-[10px] text-muted-foreground/70 hover:bg-muted/40 transition-colors" aria-label="System health and model status">
      <div className="flex items-center gap-1.5 sm:gap-3 min-w-0">
        <span className="flex items-center gap-1" aria-live="polite" aria-atomic="true">
          <span className={cn("inline-block h-1.5 w-1.5 rounded-full shrink-0", dot)} aria-hidden="true" />
          <span className="truncate max-w-[100px] sm:max-w-none">{label}</span>
        </span>
        {soulName && (
          <span className="hidden sm:flex items-center gap-1 min-w-0">
            <span className="text-muted-foreground/40 shrink-0">·</span>
            <span className="truncate max-w-[80px]">{soulName}</span>
          </span>
        )}
      </div>
      <div className="flex items-center gap-2 sm:gap-3">
        {health !== null && health !== 'offline' && health.inference_count != null && (
          <span className="hidden sm:inline">{health.inference_count} inference{health.inference_count !== 1 ? 's' : ''}</span>
        )}
      </div>
    </Link>
  )
}
