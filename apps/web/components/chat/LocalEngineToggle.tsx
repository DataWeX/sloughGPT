'use client'

import { Button } from '@/components/ui/button'
import { IconRefresh } from '@/components/ui'
import { useChatToolbarContext } from '@/contexts/ChatToolbarContext'

function CpuIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <rect x="4" y="4" width="16" height="16" rx="2" strokeWidth={2} />
      <rect x="9" y="9" width="6" height="6" strokeWidth={2} />
      <path d="M15 2v2m-6-2v2m6 16v2m-6-2v2M2 15h2m-2-6h2m16 6h2m-2-6h2" strokeWidth={2} strokeLinecap="round" />
    </svg>
  )
}

function ServerIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <rect x="2" y="2" width="20" height="8" rx="2" strokeWidth={2} />
      <rect x="2" y="14" width="20" height="8" rx="2" strokeWidth={2} />
      <circle cx="6" cy="6" r="1" fill="currentColor" />
      <circle cx="6" cy="18" r="1" fill="currentColor" />
    </svg>
  )
}

export function LocalEngineToggle() {
  const ctx = useChatToolbarContext()
  const { modelUrl, useLocal: useLocalEngine, loading: localEngineLoading, archInfo: localArchInfo, onToggle } = ctx.localEngine
  const visible = !!modelUrl
  if (!visible) return null

  return (
    <Button
      variant={useLocalEngine ? 'default' : 'ghost'}
      size="sm"
      className="h-7 px-2.5 text-xs gap-1.5 rounded-lg"
      disabled={localEngineLoading}
      onClick={onToggle}
      title={localEngineLoading ? 'Loading local engine...' : localArchInfo ? `Local GPU (${localArchInfo})` : useLocalEngine ? 'Running locally on GPU' : 'Running on server'}
      aria-pressed={useLocalEngine}
    >
      {localEngineLoading ? (
        <IconRefresh className="h-3 w-3 animate-spin" />
      ) : useLocalEngine ? <CpuIcon className="h-3 w-3" /> : <ServerIcon className="h-3 w-3" />}
      <span className="hidden sm:inline">{localEngineLoading ? 'Loading' : useLocalEngine ? 'Local' : 'Server'}</span>
    </Button>
  )
}
