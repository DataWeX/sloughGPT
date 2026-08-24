'use client'

import { useRouter } from 'next/navigation'
import { cn, IconChevronDown, IconCheck, IconRefresh } from '@sloughgpt/strui'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { useChatToolbarContext } from '@/features/chat/contexts/ChatToolbarContext'

interface ModelDropdownProps {
  variant?: 'dropdown' | 'panel'
  panelTitle?: string
}

interface ModelInfo {
  cached?: boolean
  size_gb?: number
}

interface FineTunedModel {
  name: string
  model?: string
  dataset?: string
  size_mb?: number
}

function shortModelName(m: string): string {
  return m.includes('/') ? m.split('/').pop() || m : m
}

function sizeLabel(info?: ModelInfo): string {
  return info?.size_gb ? `${info.size_gb.toFixed(2)} GB` : ''
}

function fineTunedName(m: string): string {
  // Directory names like "gpt2__dataset_1" -> "gpt2 · dataset_1"
  return m.split('__').join(' · ')
}

export function ModelDropdown({
  variant = 'dropdown',
  panelTitle = 'Backend Model',
}: ModelDropdownProps) {
  const ctx = useChatToolbarContext()
  const router = useRouter()
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
    fineTuned,
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
            <div className="px-2 py-3 text-[10px] text-muted-foreground text-center space-y-2">
              <div>No models available</div>
              <Button size="sm" variant="outline" className="h-7 text-[10px]" onClick={() => router.push('/models')}>
                Browse models
              </Button>
            </div>
          ) : availableModels.map(m => {
            const info = modelInfoMap[m]
            const sl = sizeLabel(info)
            const isCached = info?.cached
            return (
              <button
                key={m}
                onClick={() => onSelectModel(m)}
                className={cn(
                  'w-full text-left px-2 py-1 rounded text-xs transition-colors flex items-center justify-between',
                  isLoaded(m) ? 'bg-primary/[0.08] text-primary font-medium' : 'hover:bg-muted/80',
                )}
                title={`${m}${sl ? ` — ${sl}` : ''}`}
              >
                <div className="min-w-0 flex-1">
                  <span className={cn("truncate block", !isLoaded(m) && "font-mono")}>{shortModelName(m)}</span>
                  {modelDescriptions[m] && (
                    <span className="text-[9px] text-muted-foreground/60 block truncate">{modelDescriptions[m]}</span>
                  )}
                </div>
                <div className="flex items-center gap-1 shrink-0 ml-1">
                  {sl && <span className="text-[10px] text-muted-foreground/60">{sl}</span>}
                  {isCached && !isLoaded(m) && (
                    <span className="text-[9px] px-1 py-0.5 rounded bg-muted text-muted-foreground font-medium">cached</span>
                  )}
                  {isLoaded(m) && <IconCheck className="h-3 w-3" />}
                </div>
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
          {dlProgress?.status === 'downloading' && dlProgress.percentage != null && dlProgress.percentage > 0 && (
            <span className="text-[10px] font-medium shrink-0 tabular-nums text-warning">
              {dlProgress.percentage.toFixed(0)}%
            </span>
          )}
          <IconChevronDown className="h-2.5 w-2.5 opacity-40 shrink-0" aria-hidden="true" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-[200px] max-h-[300px] overflow-y-auto">
        {loadingModel && (
          <div className="h-0.5 bg-muted rounded-full mx-2 mt-2 mb-1 overflow-hidden shrink-0">
            <div
              className="h-full bg-primary rounded-full transition-[width] duration-300"
              style={{ width: dlProgress?.percentage != null && dlProgress.percentage > 0 ? `${dlProgress.percentage}%` : '100%' }}
            />
          </div>
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
                <IconChevronDown className="h-2.5 w-2.5 shrink-0 text-muted-foreground/40 ml-1" />
              )}
            </DropdownMenuItem>
          )
        })}
        {fineTuned && fineTuned.models.length > 0 && (
          <>
            <div className="h-px bg-border/50 mx-2 my-1" />
            <DropdownMenuLabel className="text-[9px] text-muted-foreground/60 font-medium uppercase tracking-wider px-2 py-1">
              Fine-tuned
            </DropdownMenuLabel>
            {fineTuned.models.map(ft => {
              const isLoading = loadingModel === ft.name
              const isLoaded = currentModel === ft.name
              return (
                <DropdownMenuItem
                  key={ft.name}
                  onSelect={() => void fineTuned.onLoad(ft.name)}
                  disabled={isLoading}
                  className="text-xs"
                  title={`${ft.name}${ft.size_mb ? ` — ${ft.size_mb.toFixed(1)} MB` : ''}`}
                >
                  <div className="min-w-0 flex-1">
                    <span className="truncate block">{fineTunedName(ft.name)}</span>
                    {(ft.model || ft.dataset) && (
                      <span className="text-[9px] text-muted-foreground/60 block truncate">
                        {[ft.model, ft.dataset].filter(Boolean).join(' · ')}
                      </span>
                    )}
                  </div>
                  <span className="text-[10px] text-muted-foreground/60 ml-1 shrink-0">
                    {ft.size_mb ? `${ft.size_mb.toFixed(1)} MB` : 'local'}
                  </span>
                  {isLoading ? (
                    <IconRefresh className="h-3 w-3 animate-spin shrink-0 text-warning ml-1" />
                  ) : isLoaded ? (
                    <IconCheck className="h-3 w-3 shrink-0 text-success ml-1" />
                  ) : null}
                </DropdownMenuItem>
              )
            })}
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
