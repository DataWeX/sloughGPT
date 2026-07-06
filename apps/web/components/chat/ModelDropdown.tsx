'use client'

import { IconChevronDown, IconCheck, IconRefresh } from '@sloughgpt/strui'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { cn } from '@/lib/cn'
import { useChatToolbarContext } from '@/contexts/ChatToolbarContext'

interface ModelDropdownProps {
  variant?: 'dropdown' | 'panel'
  panelTitle?: string
}

interface ModelInfo {
  cached?: boolean
  size_gb?: number
}

function shortModelName(m: string): string {
  return m.includes('/') ? m.split('/').pop() || m : m
}

function sizeLabel(info?: ModelInfo): string {
  return info?.size_gb ? `${info.size_gb.toFixed(2)} GB` : ''
}

export function ModelDropdown({
  variant = 'dropdown',
  panelTitle = 'Backend Model',
}: ModelDropdownProps) {
  const ctx = useChatToolbarContext()
  const {
    availableModels,
    current: currentModel,
    loading: loadingModel,
    generating,
    infoMap: modelInfoMap = {},
    descriptions: modelDescriptions = {},
    downloadProgress,
    onSelect: onSelectModel,
    onUnload: onUnloadModel,
  } = ctx.model
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
                <div className="min-w-0 flex-1">
                  <span className="truncate block">{shortModelName(m)}</span>
                  {modelDescriptions[m] && (
                    <span className="text-[9px] text-muted-foreground/60 block truncate">{modelDescriptions[m]}</span>
                  )}
                </div>
                <span className="text-[10px] text-muted-foreground/60 ml-1 shrink-0">{sl}</span>
                {isLoaded(m) && <IconCheck className="h-3 w-3 shrink-0 ml-1" />}
              </button>
            )
          })}
        </div>
      </div>
    )
  }

  const dlProgress = loadingModel ? downloadProgress[loadingModel] : undefined

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="h-7 px-2.5 font-mono text-xs gap-1.5 rounded-lg border border-transparent hover:border-border/50" aria-label={`Select model. Current: ${currentModel || 'none'}`}>
          {generating ? (
            <span className="relative inline-flex h-3 w-3 shrink-0">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/30" />
              <span className="relative inline-flex h-3 w-3 rounded-full bg-primary" />
            </span>
          ) : (
            <span className={cn(
              'inline-block h-1.5 w-1.5 rounded-full shrink-0',
              loadingModel && dlProgress?.status === 'downloading' ? 'bg-warning animate-pulse' :
              loadingModel ? 'bg-warning animate-pulse' :
              currentModel ? 'bg-success' : 'bg-muted-foreground/30'
            )} />
          )}
          <span className="truncate max-w-[48px] sm:max-w-[64px]" title={loadingModel || currentModel || 'Select a model to load'}>
            {loadingModel ? shortModelName(loadingModel) : currentModel ? shortModelName(currentModel) : 'Select model'}
          </span>
          {dlProgress?.status === 'downloading' && (
            <>
              {dlProgress.eta_seconds != null && dlProgress.eta_seconds > 0 && (
                <span className="text-[9px] text-muted-foreground shrink-0 hidden sm:inline">{Math.round(dlProgress.eta_seconds)}s</span>
              )}
              <span className={cn("text-[10px] font-medium shrink-0 tabular-nums", dlProgress.percentage > 0 ? "text-warning" : "text-muted-foreground")}>
                {dlProgress.percentage > 0 ? `${dlProgress.percentage.toFixed(0)}%` : '...'}
              </span>
            </>
          )}
          <IconChevronDown className="h-2.5 w-2.5 opacity-40 shrink-0" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-[200px] max-h-[300px] overflow-y-auto">
        {loadingModel && dlProgress?.status === 'downloading' && (
          <div className="px-3 pt-2 pb-1.5 space-y-1">
            <div className="h-1.5 bg-muted rounded-full overflow-hidden" role="progressbar" aria-valuenow={dlProgress.percentage || 0} aria-valuemin={0} aria-valuemax={100} aria-label={`Download progress: ${dlProgress.percentage?.toFixed(0) || 0}%`}>
              <div
                className="h-full bg-warning rounded-full transition-all duration-500"
                style={{ width: `${Math.max(2, dlProgress.percentage || 0)}%` }}
              />
            </div>
            <div className="flex items-center justify-between gap-2 text-[10px] text-muted-foreground">
              <span className="tabular-nums">
                {dlProgress.percentage?.toFixed(0) || '0'}%
              </span>
              {dlProgress.speed_mb_per_sec != null && dlProgress.speed_mb_per_sec > 0 && (
                <span className="tabular-nums">{dlProgress.speed_mb_per_sec.toFixed(1)} MB/s</span>
              )}
              {dlProgress.eta_seconds != null && dlProgress.eta_seconds > 0 && (
                <span className="tabular-nums">{Math.round(dlProgress.eta_seconds)}s left</span>
              )}
            </div>
            {dlProgress.current_file && (
              <div className="text-[9px] text-muted-foreground/60 truncate max-w-[220px]" title={dlProgress.current_file}>
                {dlProgress.current_file.split('/').pop()}
              </div>
            )}
            {dlProgress.files_total != null && dlProgress.files_total > 1 && (
              <div className="text-[9px] text-muted-foreground/40 tabular-nums">
                {dlProgress.files_completed || 0} / {dlProgress.files_total} files
              </div>
            )}
          </div>
        )}
        {loadingModel && (
          <div className="h-0.5 bg-muted rounded-full mx-2 mb-1 overflow-hidden shrink-0">
            <div className="h-full bg-primary rounded-full animate-pulse" style={{ width: '60%' }} />
          </div>
        )}
        {currentModel && onUnloadModel && (
          <>
            <DropdownMenuItem
              onSelect={() => onUnloadModel()}
              className="text-destructive focus:text-destructive text-xs gap-2"
            >
              <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
              Remove model
            </DropdownMenuItem>
            <div className="h-px bg-border/50 mx-2 my-1" />
          </>
        )}
        {availableModels.map(m => {
          const info = modelInfoMap[m]
          const isCached = info?.cached
          const sl = sizeLabel(info)
          const modelDl = downloadProgress[m]
          return (
            <DropdownMenuItem
              key={m}
              onSelect={() => onSelectModel(m)}
              disabled={isLoading(m)}
              className="font-mono text-xs"
              title={`${m}${isCached ? ' (cached)' : ' (download)'}${sl ? ` — ${sl}` : ''}`}
            >
              <div className="min-w-0 flex-1">
                <span className="truncate block">{shortModelName(m)}</span>
                {modelDescriptions[m] && (
                  <span className="text-[9px] text-muted-foreground/60 block truncate font-sans">{modelDescriptions[m]}</span>
                )}
              </div>
              <span className="text-[10px] text-muted-foreground/60 ml-1 shrink-0">{sl}</span>
              {isLoading(m) && modelDl?.status === 'downloading' ? (
                <span className="text-[9px] text-warning font-medium ml-1 shrink-0 tabular-nums">
                  {modelDl.percentage > 0 ? `${modelDl.percentage.toFixed(0)}%` : '...'}
                </span>
              ) : isLoading(m) ? (
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
