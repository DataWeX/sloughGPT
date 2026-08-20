'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { cn, Card, CardContent, CardHeader, CardTitle, Progress } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { IconStar } from '@sloughgpt/strui'
import { catalogIdMatchesRuntime } from '@/lib/inference-display'
import { generateController } from '@/lib/generate-controller'
import { getJsonItem, setJsonItem } from '@/lib/format-bytes'
import { useLoadModel } from '@/lib/query/api-hooks'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useConversionStatus, formatStage } from '@/hooks/useConversionStatus'
import { logger } from '@/lib/dev-log'

const FAVORITES_KEY = 'sloughgpt-model-favorites'

interface Model {
  id: string; name?: string; source?: string; description?: string; tags?: string[]
  size_mb?: number; size_gb?: number; params?: string; cached?: boolean; thumbnail?: string
}

interface ModelCatalogCardProps {
  models: Model[]
  modelsLoading: boolean
  activeRuntimeId: string | null
  onModelLoaded: () => Promise<void>
}

export default function ModelCatalogCard({ models, modelsLoading, activeRuntimeId, onModelLoaded }: ModelCatalogCardProps) {
  const router = useRouter()
  const addToast = useToastStore(s => s.addToast)
  const { mutateAsync: loadModel } = useLoadModel()
  const [loadingModel, setLoadingModel] = useState<string | null>(null)
  const [warmingModel, setWarmingModel] = useState<string | null>(null)
  const [modelSearch, setModelSearch] = useState('')
  const [trackedModel, setTrackedModel] = useState<string | null>(null)
  const { status: conversionStatus } = useConversionStatus(trackedModel)
  const [loadTimes, setLoadTimes] = useState<Record<string, number>>({})
  const [favorites, setFavorites] = useState<Set<string>>(new Set())

  useEffect(() => {
    let cancelled = false
    getJsonItem<string[]>(FAVORITES_KEY, []).then(list => {
      if (!cancelled) setFavorites(new Set(list))
    })
    return () => { cancelled = true }
  }, [])

  const toggleFavorite = (modelId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setFavorites(prev => {
      const next = new Set(prev)
      if (next.has(modelId)) next.delete(modelId)
      else next.add(modelId)
      setJsonItem(FAVORITES_KEY, [...next]).catch(() => {})
      return next
    })
  }

  const handleLoadModel = async (modelId: string) => {
    setLoadingModel(modelId)
    setTrackedModel(modelId)
    const startTime = Date.now()
    try {
      const data = await loadModel(modelId)
      if (data.error) { addToast(data.error, 'error'); return }
      const loadTime = Date.now() - startTime
      setLoadTimes(prev => ({ ...prev, [modelId]: loadTime }))
      addToast(`${data.model_id ?? modelId} ready (${(loadTime / 1000).toFixed(1)}s)`, 'success')
      setLoadingModel(null)
      setWarmingModel(modelId)
      try { await generateController.generate({ prompt: 'Hello', max_new_tokens: 2 }) } catch {
        logger.warning('ModelCatalogCard: warmup generation failed', { modelId })
      }
      setWarmingModel(null)
    } catch (err) {
      addToast(extractErrorMessage(err), 'error')
    } finally {
      setLoadingModel(null)
      setWarmingModel(null)
      setTrackedModel(null)
      await onModelLoaded()
    }
  }

  const filtered = modelSearch
    ? models.filter(m => (m.name || m.id).toLowerCase().includes(modelSearch.toLowerCase()) || m.id.toLowerCase().includes(modelSearch.toLowerCase()))
    : models

  const sorted = [...filtered].sort((a, b) => {
    const aFav = favorites.has(a.id) ? 0 : 1
    const bFav = favorites.has(b.id) ? 0 : 1
    return aFav - bFav
  })

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Model Catalog</CardTitle></CardHeader>
      <CardContent>
        {modelsLoading ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {[1,2,3,4,5,6].map(i => (
              <div key={i} className="animate-pulse rounded-lg bg-muted/50 border border-border/50 p-4 h-20" />
            ))}
          </div>
        ) : models.length === 0 ? (
          <div className="text-center py-8 text-sm text-muted-foreground">
            No models available
            <div className="mt-2">
              <Button size="sm" variant="outline" className="h-8 text-xs" onClick={() => router.push('/models')}>
                Browse models
              </Button>
            </div>
          </div>
        ) : (
          <>
            {models.length > 3 && (
              <div className="mb-3">
                <input
                  type="text"
                  value={modelSearch}
                  onChange={e => setModelSearch(e.target.value)}
                  placeholder="Search models..."
                  className="h-8 w-full max-w-xs rounded-md border border-border/60 bg-background px-2 text-xs placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/30"
                />
              </div>
            )}
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3 max-h-[60vh] overflow-y-auto overscroll-contain">
              {sorted.map((model) => {
                const isLoading = loadingModel === model.id
                const isLoaded = activeRuntimeId ? catalogIdMatchesRuntime(model.id, activeRuntimeId) : false
                const isWarming = warmingModel === model.id
                const isFav = favorites.has(model.id)
                return (
                  <div key={model.id} className="group">
                  <div
                    className={cn("flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-all duration-200 hover:-translate-y-0.5 hover:shadow-sm",
                      isLoaded ? "border-primary/40 bg-primary/5" : "border-border/60",
                      isLoading && "border-primary/30 bg-primary/[0.08]")}
                    onClick={() => router.push(`/model/${encodeURIComponent(model.id)}`)}
                    onKeyDown={(e) => { if (e.key === 'Enter') router.push(`/model/${encodeURIComponent(model.id)}`) }}
                    tabIndex={0}
                    role="button"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={(e) => toggleFavorite(model.id, e)}
                          className={cn(
                            "h-4 w-4 flex items-center justify-center rounded shrink-0 transition-all",
                            isFav ? "opacity-100 text-warning" : "opacity-0 group-hover:opacity-60 hover:!opacity-100 text-muted-foreground/40"
                          )}
                          aria-label={isFav ? 'Remove from favorites' : 'Add to favorites'}
                        >
                          <IconStar className="h-3 w-3" filled={isFav} />
                        </button>
                        <div className="text-sm font-medium truncate">{model.name || model.id}</div>
                        {isLoading && (
                          <span className="relative flex h-2 w-2 shrink-0">
                            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
                            <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
                          </span>
                        )}
                        {isWarming && (
                          <span className="relative flex h-2 w-2 shrink-0">
                            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-warning/60" />
                            <span className="relative inline-flex h-2 w-2 rounded-full bg-warning" />
                          </span>
                        )}
                      </div>
                      {model.description && (
                        <div className="text-[10px] text-muted-foreground/70 truncate mt-0.5">{model.description}</div>
                      )}
                      <div className="flex items-center gap-1.5 mt-1 flex-wrap">
                        {model.params && <span className="text-[10px] text-muted-foreground font-mono">{model.params}</span>}
                        {model.size_gb && <span className="text-[10px] text-muted-foreground">{(model.size_gb).toFixed(1)} GB</span>}
                        {model.source && model.source !== 'local' && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">{model.source}</span>
                        )}
                        {isLoaded && <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">Loaded</span>}
                        {!isLoaded && model.cached && <span className="text-[10px] px-1.5 py-0.5 rounded bg-success/10 text-success font-medium">Cached</span>}
                        {loadTimes[model.id] && <span className="text-[10px] text-muted-foreground/60">{(loadTimes[model.id] / 1000).toFixed(1)}s</span>}
                        {model.tags && model.tags.slice(0, 2).map(tag => (
                          <span key={tag} className="text-[10px] px-1.5 py-0.5 rounded bg-secondary text-secondary-foreground font-medium">{tag}</span>
                        ))}
                      </div>
                      {isLoading && (
                        <p className="text-[10px] text-primary mt-1">
                          {conversionStatus && conversionStatus.stage !== 'idle'
                            ? `Converting: ${conversionStatus.stage} (${Math.round(conversionStatus.progress * 100)}%)`
                            : 'Loading model...'}
                        </p>
                      )}
                      {isWarming && (
                        <p className="text-[10px] text-warning mt-1">Warming up model...</p>
                      )}
                    </div>
                    <div className="flex gap-1 shrink-0 ml-2">
                      {model.source === 'local' ? (
                        <span className="text-[10px] text-muted-foreground">Local</span>
                      ) : (
                        <Button size="sm" variant={isLoaded ? 'outline' : 'default'} className="h-7 text-xs px-3"
                          disabled={isLoading || warmingModel === model.id}
                          onClick={(e) => { e.stopPropagation(); handleLoadModel(model.id) }}>
                          {isLoading
                            ? (conversionStatus && conversionStatus.stage !== 'idle'
                                ? `${Math.round(conversionStatus.progress * 100)}%`
                                : <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />)
                            : warmingModel === model.id ? 'Warming…' : isLoaded ? 'Loaded' : 'Load'}
                        </Button>
                      )}
                    </div>
                  </div>
                  {/* Conversion progress bar */}
                  {isLoading && conversionStatus && conversionStatus.stage !== 'idle' && conversionStatus.stage !== 'ready' && (
                    <div key={`${model.id}-progress`} className="col-span-full mt-1">
                      <Progress
                        value={conversionStatus.progress * 100}
                        size="xs"
                        variant={conversionStatus.stage === 'error' ? 'error' : 'default'}
                        label={formatStage(conversionStatus.stage)}
                        showValue
                      />
                      <span className="text-[9px] text-muted-foreground ml-auto">{conversionStatus.elapsed_s.toFixed(0)}s</span>
                    </div>
                  )}
                  </div>
                )
              })}
            </div>
            {sorted.length === 0 && modelSearch && (
              <div className="text-center py-4 text-xs text-muted-foreground">No models matching &quot;{modelSearch}&quot;</div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}
