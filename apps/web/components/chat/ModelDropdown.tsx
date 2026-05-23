'use client'

import { IconChevronDown, IconCheck, IconRefresh } from '@/components/ui'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/cn'

interface ModelInfo {
  cached?: boolean
  size_gb?: number
}

interface ModelDropdownProps {
  availableModels: string[]
  currentModel: string
  onSelectModel: (model: string) => void | Promise<void>
  modelInfoMap?: Record<string, ModelInfo>
  loadingModel?: string | null
  generating?: boolean
  variant?: 'dropdown' | 'panel'
  panelTitle?: string
}

function shortModelName(m: string): string {
  return m.includes('/') ? m.split('/').pop() || m : m
}

function sizeLabel(info?: ModelInfo): string {
  return info?.size_gb ? `${info.size_gb.toFixed(2)} GB` : ''
}

export function ModelDropdown({
  availableModels,
  currentModel,
  onSelectModel,
  modelInfoMap = {},
  loadingModel,
  generating,
  variant = 'dropdown',
  panelTitle = 'Backend Model',
}: ModelDropdownProps) {
  const isLoading = (m: string) => m === loadingModel
  const isLoaded = (m: string) => m === currentModel

  if (variant === 'panel') {
    return (
      <div className="rounded-lg border border-border/40 bg-muted/10">
        {panelTitle && (
          <div className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider px-2.5 py-1.5 border-b border-border/30">
            {panelTitle}
          </div>
        )}
        <div className="max-h-28 overflow-y-auto p-1.5 space-y-0.5">
          {availableModels.length === 0 ? (
            <div className="px-2 py-3 text-[10px] text-muted-foreground text-center">No models available</div>
          ) : availableModels.map(m => {
            const info = modelInfoMap[m]
            const sl = sizeLabel(info)
            return (
              <button
                key={m}
                onClick={() => onSelectModel(m)}
                className={cn(
                  'w-full text-left px-2 py-1 rounded text-xs transition-colors flex items-center justify-between font-mono',
                  isLoaded(m) ? 'bg-primary/10 text-primary font-medium' : 'hover:bg-muted/80',
                )}
                title={`${m}${sl ? ` — ${sl}` : ''}`}
              >
                <span className="truncate">{shortModelName(m)}</span>
                <span className="text-[10px] text-muted-foreground/60 ml-1 shrink-0">{sl}</span>
                {isLoaded(m) && <IconCheck className="h-3 w-3 shrink-0 ml-1" />}
              </button>
            )
          })}
        </div>
      </div>
    )
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="h-7 px-2.5 font-mono text-xs gap-1.5 rounded-lg border border-transparent hover:border-border/50">
          {generating ? (
            <span className="relative inline-flex h-3 w-3 shrink-0">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/30" />
              <span className="relative inline-flex h-3 w-3 rounded-full bg-primary" />
            </span>
          ) : (
            <span className={cn(
              'inline-block h-1.5 w-1.5 rounded-full shrink-0',
              loadingModel ? 'bg-warning animate-pulse' :
              currentModel ? 'bg-success' : 'bg-muted-foreground/30'
            )} />
          )}
          <span className="truncate max-w-[48px] sm:max-w-[64px]" title={loadingModel || currentModel || 'Select a model to load'}>
            {loadingModel ? shortModelName(loadingModel) : currentModel ? shortModelName(currentModel) : 'Select model'}
          </span>
          <IconChevronDown className="h-2.5 w-2.5 opacity-40 shrink-0" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-[200px] max-h-[300px] overflow-y-auto">
        {loadingModel && (
          <div className="h-0.5 bg-muted rounded-full mx-2 mb-1 overflow-hidden shrink-0">
            <div className="h-full bg-primary rounded-full animate-pulse" style={{ width: '60%' }} />
          </div>
        )}
        {availableModels.map(m => {
          const info = modelInfoMap[m]
          const isCached = info?.cached
          const sl = sizeLabel(info)
          return (
            <DropdownMenuItem
              key={m}
              onSelect={() => onSelectModel(m)}
              disabled={isLoading(m)}
              className="font-mono text-xs"
              title={`${m}${isCached ? ' (cached)' : ' (download)'}${sl ? ` — ${sl}` : ''}`}
            >
              <span className="truncate flex-1">{shortModelName(m)}</span>
              <span className="text-[10px] text-muted-foreground/60 ml-1 shrink-0">{sl}</span>
              {isLoading(m) ? (
                <IconRefresh className="h-3 w-3 animate-spin shrink-0 text-warning ml-1" />
              ) : isLoaded(m) ? (
                <IconCheck className="h-3 w-3 shrink-0 text-success ml-1" />
              ) : isCached ? (
                <span className="text-[9px] text-muted-foreground/40 px-1 ml-1 border border-border/30 rounded leading-none">cached</span>
              ) : (
                <svg className="h-2.5 w-2.5 shrink-0 text-muted-foreground/40 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" /></svg>
              )}
            </DropdownMenuItem>
          )
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
