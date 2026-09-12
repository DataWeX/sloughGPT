'use client'

import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { consciousnessController } from '@/lib/consciousness-controller'
import {
  Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, Skeleton,
  Input, Select, SelectTrigger, SelectValue, SelectContent, SelectItem,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'
import { useSelection } from '@/hooks/useSelection'

interface Episode {
  input: string
  response: string
  narrative: string
  qualia: Record<string, number>
  growth_delta: number
  rating: number
  timestamp: string
}

const QUALIA_DIMS = [
  { key: 'valence', color: '#8b5cf6' },
  { key: 'arousal', color: '#ef4444' },
  { key: 'novelty', color: '#f59e0b' },
  { key: 'coherence', color: '#22c55e' },
  { key: 'salience', color: '#3b82f6' },
  { key: 'certainty', color: '#06b6d4' },
  { key: 'complexity', color: '#d946ef' },
]

const RATING_COLORS = ['#ef4444', '#f97316', '#eab308', '#84cc16', '#22c55e']

const PAGE_SIZE = 20

function formatTimestamp(ts: string): string {
  try {
    const d = new Date(ts)
    if (isNaN(d.getTime())) return ts
    return d.toLocaleString()
  } catch {
    return ts
  }
}

function truncate(s: string, max: number): string {
  if (!s) return ''
  return s.length > max ? s.slice(0, max) + '...' : s
}

function StarRating({ rating, onRate, disabled }: { rating: number; onRate: (r: number) => void; disabled: boolean }) {
  return (
    <div className="flex gap-0.5">
      {[1, 2, 3, 4, 5].map((s) => (
        <button
          key={s}
          onClick={() => onRate(s)}
          disabled={disabled}
          className={`text-sm transition-colors ${s <= rating ? 'text-yellow-400' : 'text-muted-foreground/40'} ${disabled ? 'cursor-not-allowed' : 'cursor-pointer hover:text-yellow-300'}`}
        >
          {s <= rating ? '\u2605' : '\u2606'}
        </button>
      ))}
    </div>
  )
}

export default function ConsciousnessHistoryPage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()
  const { selectedIds, toggleSelect, clearSelection } = useSelection()
  const [loading, setLoading] = useState(true)
  const [episodes, setEpisodes] = useState<Episode[]>([])
  const [totalEpisodes, setTotalEpisodes] = useState(0)
  const [offset, setOffset] = useState(0)
  const [hasMore, setHasMore] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [ratingEpisode, setRatingEpisode] = useState<number | null>(null)

  const [filterRating, setFilterRating] = useState<string>('all')
  const [filterGrowth, setFilterGrowth] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [sortOrder, setSortOrder] = useState<string>('newest')
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null)

  const [exporting, setExporting] = useState(false)
  const [importing, setImporting] = useState(false)

  const fetchEpisodes = useCallback(async (off: number, append: boolean) => {
    try {
      const data = await consciousnessController.getEpisodeHistory(PAGE_SIZE) as any
      const newEps: Episode[] = data.episodes ?? []
      const total: number = data.total ?? 0
      if (append) {
        setEpisodes(prev => [...prev, ...newEps])
      } else {
        setEpisodes(newEps)
      }
      setTotalEpisodes(total)
      setHasMore(off + newEps.length < total)
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }, [addToast])

  const initialLoad = useCallback(async () => {
    setLoading(true)
    await fetchEpisodes(0, false)
    setLoading(false)
  }, [fetchEpisodes])

  useEffect(() => { initialLoad() }, [initialLoad])

  const handleLoadMore = useCallback(async () => {
    setLoadingMore(true)
    const newOffset = offset + PAGE_SIZE
    await fetchEpisodes(newOffset, true)
    setOffset(newOffset)
    setLoadingMore(false)
  }, [offset, fetchEpisodes])

  const handleFeedback = useCallback(async (episodeIndex: number, rating: number) => {
    setRatingEpisode(episodeIndex)
    try {
      await consciousnessController.submitFeedback({ episode_index: episodeIndex, rating })
      addToast(`Rated ${rating}/5`, 'success')
      setEpisodes(prev => prev.map((ep, i) => i === episodeIndex ? { ...ep, rating } : ep))
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setRatingEpisode(null)
    }
  }, [addToast])

  const filteredEpisodes = useMemo(() => {
    let result = [...episodes]
    if (filterRating !== 'all') {
      const r = parseInt(filterRating)
      result = result.filter(ep => ep.rating === r)
    }
    if (filterGrowth === 'positive') {
      result = result.filter(ep => ep.growth_delta >= 0)
    } else if (filterGrowth === 'negative') {
      result = result.filter(ep => ep.growth_delta < 0)
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase()
      result = result.filter(ep =>
        (ep.input && ep.input.toLowerCase().includes(q)) ||
        (ep.response && ep.response.toLowerCase().includes(q)) ||
        (ep.narrative && ep.narrative.toLowerCase().includes(q))
      )
    }
    if (sortOrder === 'oldest') {
      result.reverse()
    }
    return result
  }, [episodes, filterRating, filterGrowth, searchQuery, sortOrder])

  const stats = useMemo(() => {
    const filtered = filteredEpisodes
    if (episodes.length === 0) return null
    const avgRating = episodes.filter(e => e.rating > 0).reduce((s, e) => s + e.rating, 0) / Math.max(1, episodes.filter(e => e.rating > 0).length)
    const avgGrowth = episodes.reduce((s, e) => s + e.growth_delta, 0) / episodes.length
    const dist = [0, 0, 0, 0, 0]
    episodes.forEach(e => { if (e.rating >= 1 && e.rating <= 5) dist[e.rating - 1]++ })
    return {
      total: episodes.length,
      filtered: filtered.length,
      avgRating: isNaN(avgRating) ? 0 : avgRating,
      avgGrowth,
      dist,
    }
  }, [episodes, filteredEpisodes])

  const handleExport = useCallback(async () => {
    setExporting(true)
    try {
      const blob = new Blob([JSON.stringify(episodes, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `consciousness-history-${Date.now()}.json`
      a.click()
      URL.revokeObjectURL(url)
      addToast('Exported history', 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setExporting(false)
    }
  }, [episodes, addToast])

  const handleImport = useCallback(async () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return
      setImporting(true)
      try {
        const text = await file.text()
        const data = JSON.parse(text)
        addToast(`Imported ${Array.isArray(data) ? data.length : 0} episodes`, 'success')
        await initialLoad()
      } catch {
        addToast('Invalid JSON file', 'error')
      } finally {
        setImporting(false)
      }
    }
    input.click()
  }, [addToast, initialLoad])

  if (loading) {
    return (
      <PageContainer title={t('consciousness_history.page_title')}>
        <div className="space-y-6 p-6">
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
          </div>
          <Skeleton className="h-12" />
          <Skeleton className="h-72" />
          <Skeleton className="h-72" />
        </div>
      </PageContainer>
    )
  }

  return (
    <PageContainer title={t('consciousness_history.page_title')}>
      <div className="space-y-6 p-6">
        {stats && (
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">{t('consciousness_history.total_episodes')}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stats.total}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">{t('consciousness_history.filtered_count')}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stats.filtered}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">{t('consciousness_history.avg_rating')}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stats.avgRating.toFixed(1)} / 5</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">{t('consciousness_history.avg_growth')}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className={`text-2xl font-bold ${stats.avgGrowth >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {stats.avgGrowth >= 0 ? '+' : ''}{(stats.avgGrowth * 100).toFixed(1)}%
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {stats && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-muted-foreground">{t('consciousness_history.rating_distribution')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-end gap-2 h-12">
                {stats.dist.map((count, i) => {
                  const maxCount = Math.max(1, ...stats.dist)
                  const h = (count / maxCount) * 100
                  return (
                    <div key={i} className="flex-1 flex flex-col items-center gap-1">
                      <div
                        className="w-full rounded-sm transition-all"
                        style={{
                          height: `${Math.max(4, h)}%`,
                          backgroundColor: RATING_COLORS[i],
                          opacity: 0.8,
                        }}
                      />
                      <span className="text-[10px] text-muted-foreground">{count}</span>
                    </div>
                  )
                })}
              </div>
              <div className="flex gap-2 mt-1">
                {['1', '2', '3', '4', '5'].map((r) => (
                  <span key={r} className="flex-1 text-center text-[10px] text-muted-foreground">{r}\u2605</span>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader className="pb-2">
            <CardTitle>{t('consciousness_history.filters_title')}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-3 items-center">
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">{t('consciousness_history.sort')}:</span>
                <Select value={sortOrder} onValueChange={setSortOrder}>
                  <SelectTrigger className="w-[140px] h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="newest">{t('consciousness_history.newest')}</SelectItem>
                    <SelectItem value="oldest">{t('consciousness_history.oldest')}</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">{t('consciousness_history.rating')}:</span>
                <Select value={filterRating} onValueChange={setFilterRating}>
                  <SelectTrigger className="w-[120px] h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">{t('consciousness_history.all')}</SelectItem>
                    <SelectItem value="1">1 \u2605</SelectItem>
                    <SelectItem value="2">2 \u2605</SelectItem>
                    <SelectItem value="3">3 \u2605</SelectItem>
                    <SelectItem value="4">4 \u2605</SelectItem>
                    <SelectItem value="5">5 \u2605</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">{t('consciousness_history.growth')}:</span>
                <Select value={filterGrowth} onValueChange={setFilterGrowth}>
                  <SelectTrigger className="w-[130px] h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">{t('consciousness_history.all')}</SelectItem>
                    <SelectItem value="positive">{t('consciousness_history.positive')}</SelectItem>
                    <SelectItem value="negative">{t('consciousness_history.negative')}</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Input
                placeholder={t('consciousness_history.search_placeholder')}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="max-w-[200px] h-8"
              />
              <div className="flex-1" />
              <Button size="sm" variant="outline" onClick={handleExport} disabled={exporting}>
                {exporting ? t('consciousness_history.exporting') : t('consciousness_history.export')}
              </Button>
              <Button size="sm" variant="outline" onClick={handleImport} disabled={importing}>
                {importing ? t('consciousness_history.importing') : t('consciousness_history.import')}
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {t('consciousness_history.timeline_title')}
              <Badge variant="outline" className="text-xs">{filteredEpisodes.length} shown</Badge>
            </CardTitle>
            <CardDescription>{t('consciousness_history.timeline_desc')}</CardDescription>
          </CardHeader>
          <CardContent>
            {filteredEpisodes.length > 0 ? (
              <div className="relative">
                <div className="absolute left-4 top-0 bottom-0 w-px bg-border" />
                <div className="space-y-4">
                  {selectedIds.size > 0 && (
                    <div className="flex items-center gap-2 py-2">
                      <span className="text-sm text-muted-foreground">{selectedIds.size} selected</span>
                      <Button size="sm" variant="ghost" onClick={clearSelection}>Clear</Button>
                    </div>
                  )}
                  {filteredEpisodes.map((ep, i) => {
                    const realIndex = sortOrder === 'newest' ? episodes.length - 1 - i : i
                    const isExpanded = expandedIndex === i
                    const isSelected = selectedIds.has(String(realIndex))
                    return (
                      <div key={i} className="relative pl-10">
                        <div className={`absolute left-2.5 top-3 h-3 w-3 rounded-full border-2 border-background ${ep.growth_delta >= 0 ? 'bg-green-500' : 'bg-red-500'}`} />
                        <div className={`rounded-lg border p-3 text-sm cursor-pointer hover:bg-muted/50 transition-colors ${isSelected ? 'bg-primary/10 border-primary/30' : ''}`} onClick={() => setExpandedIndex(isExpanded ? null : i)}>
                          <div className="flex items-start justify-between gap-3">
                            <div className="flex items-center gap-2">
                              <input
                                type="checkbox"
                                checked={isSelected}
                                onChange={(e) => { e.stopPropagation(); toggleSelect(String(realIndex)) }}
                                className="h-4 w-4 rounded border-gray-300"
                                onClick={(e) => e.stopPropagation()}
                              />
                              <div className="flex-1 min-w-0">
                                <div className="text-xs text-muted-foreground mb-1">{formatTimestamp(ep.timestamp)}</div>
                                <div className="text-sm font-medium truncate">{truncate(ep.input, 120)}</div>
                                <div className="flex items-center gap-2 mt-1.5">
                                  <Badge variant={ep.growth_delta >= 0 ? 'default' : 'destructive'} className="text-[10px] px-1.5 py-0">
                                    {ep.growth_delta >= 0 ? '+' : ''}{(ep.growth_delta * 100).toFixed(1)}%
                                  </Badge>
                                </div>
                              </div>
                            </div>
                            <div className="flex items-center gap-2 shrink-0" onClick={(e) => e.stopPropagation()}>
                              <StarRating rating={ep.rating} onRate={(r) => handleFeedback(realIndex, r)} disabled={ratingEpisode === realIndex} />
                            </div>
                          </div>
                          {isExpanded && (
                            <div className="mt-3 space-y-3 border-t pt-3">
                              {ep.response && (
                                <div>
                                  <div className="text-xs font-medium text-muted-foreground mb-1">{t('consciousness_history.response')}</div>
                                  <div className="text-sm rounded-md bg-muted p-2 whitespace-pre-wrap">{ep.response}</div>
                                </div>
                              )}
                              {ep.narrative && (
                                <div>
                                  <div className="text-xs font-medium text-muted-foreground mb-1">{t('consciousness_history.narrative')}</div>
                                  <div className="text-sm italic text-muted-foreground">{ep.narrative}</div>
                                </div>
                              )}
                              {ep.qualia && Object.keys(ep.qualia).length > 0 && (
                                <div>
                                  <div className="text-xs font-medium text-muted-foreground mb-2">{t('consciousness_history.qualia_breakdown')}</div>
                                  <div className="grid grid-cols-2 gap-2">
                                    {QUALIA_DIMS.map(({ key, color }) => {
                                      const val = (ep.qualia as any)[key]
                                      if (val === undefined) return null
                                      return (
                                        <div key={key} className="flex items-center gap-2">
                                          <span className="text-xs capitalize w-20">{key}</span>
                                          <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                                            <div
                                              className="h-full rounded-full transition-all duration-500"
                                              style={{
                                                width: `${Math.max(0, Math.min(100, ((val as number) + 1) / 2 * 100))}%`,
                                                backgroundColor: color,
                                              }}
                                            />
                                          </div>
                                          <span className="text-[10px] text-muted-foreground w-8 text-right">
                                            {(val as number).toFixed(2)}
                                          </span>
                                        </div>
                                      )
                                    })}
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            ) : (
              <div className="flex h-[100px] items-center justify-center text-sm text-muted-foreground">
                {t('consciousness_history.no_episodes')}
              </div>
            )}
          </CardContent>
        </Card>

        {hasMore && !searchQuery && filterRating === 'all' && filterGrowth === 'all' && (
          <div className="flex justify-center">
            <Button variant="outline" onClick={handleLoadMore} disabled={loadingMore}>
              {loadingMore ? t('consciousness_history.loading') : t('consciousness_history.load_more')}
            </Button>
          </div>
        )}
      </div>
    </PageContainer>
  )
}
