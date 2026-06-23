'use client'

import { Button } from '@/components/ui/button'
import { IconRefresh } from '@/components/ui'
import { Cpu, Server } from 'lucide-react'
import { useChatToolbarContext } from '@/contexts/ChatToolbarContext'

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
      ) : useLocalEngine ? <Cpu className="h-3 w-3" /> : <Server className="h-3 w-3" />}
      <span className="hidden sm:inline">{localEngineLoading ? 'Loading' : useLocalEngine ? 'Local' : 'Server'}</span>
    </Button>
  )
}
