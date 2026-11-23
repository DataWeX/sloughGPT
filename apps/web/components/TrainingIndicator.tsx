'use client'

import Link from '@/vite/next-compat/link'
import { useRouter } from '@/vite/next-compat/navigation'
import { memo, useEffect, useRef, useState } from 'react'
import { cn } from '@sloughgpt/strui'
import { useTrainingSession } from '@/hooks/useTrainingSession'
import { notify, useBannerStore } from '@/lib/notify'

export const TrainingIndicator = memo(function TrainingIndicator() {
  // Mount gate: training state is rehydrated from localStorage, so the
  // first client render would differ from SSR (null) and break hydration.
  // Render nothing until mounted so both trees match.
  const [mounted, setMounted] = useState(false)
  useEffect(() => {
    setMounted(true)
  }, [])
  const { trainingRunning, phase, progress, loss, method, error } = useTrainingSession()
  const router = useRouter()
  const routerRef = useRef(router)
  // The banner is sticky and outlives renders (single slot) — its action
  // must call the current router, so keep the ref fresh.
  routerRef.current = router

  // A failed run is a journey blocker, so it goes through the ONE global
  // banner surface (rule 16) instead of a sidebar-only label: the banner is
  // impossible to miss and can carry the remedy (→ /training). Same key → a
  // repeated error phase refreshes the row rather than stacking; leaving the
  // error state retires it. The sidebar row itself stays silent on error —
  // the banner IS the failure surface.
  useEffect(() => {
    if (phase !== 'error') {
      const store = useBannerStore.getState()
      const stale = store.banners.find((b) => b.key === 'training-failed')
      if (stale) notify.dismissBanner(stale.id)
      return
    }
    notify.banner({
      key: 'training-failed',
      tone: 'destructive',
      title: 'Training failed',
      message: error || undefined,
      action: {
        label: 'View training',
        onAction: () => routerRef.current.push('/training'),
      },
    })
  }, [phase, error])

  if (!mounted || phase === 'error' || (!trainingRunning && phase !== 'complete')) {
    return null
  }

  return (
    <Link
      href="/training"
      prefetch={false}
      className={cn(
        'flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-[11px] transition-colors',
        'hover:bg-primary/10',
        phase === 'complete' ? 'text-success' : 'text-primary',
      )}
      title="View training status"
    >
      <div className="relative flex h-1.5 w-1.5 shrink-0">
        {trainingRunning && (
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75" />
        )}
        <span
          className={cn(
            'relative inline-flex h-1.5 w-1.5 rounded-full',
            phase === 'complete' ? 'bg-success' : 'bg-primary',
          )}
        />
      </div>
      <div className="min-w-0 flex-1" role="status" aria-live="polite">
        <div className="truncate text-xs font-medium">
          {phase === 'complete'
            ? 'Training complete'
            : `${method === 'turbo' ? 'Turbo' : 'Training'}`}
        </div>
        {trainingRunning && (
          <div className="flex items-center gap-0.5 text-[11px] text-muted-foreground/60 tabular-nums">
            <span>{Math.round(progress ?? 0)}%</span>
            {loss != null && (
              <>
                <span>·</span>
                <span>loss {loss.toFixed(3)}</span>
              </>
            )}
          </div>
        )}
      </div>
    </Link>
  )
})
