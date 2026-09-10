'use client'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { PUBLIC_API_URL } from '@/lib/config'
import {
  Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, Skeleton,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'

interface QualiaState {
  [key: string]: number
}

interface BeliefsState {
  [key: string]: number
}

interface ConsciousnessStatus {
  enabled: boolean
  level: number
  current_qualia: QualiaState
  beliefs: BeliefsState
  episodes: number
  narrative: string
  response_quality: {
    avg_growth: number
    positive_ratio: number
    total: number
    last_growth: number
  }
  last_reflection: string
}

interface Episode {
  timestamp: number
  input_text: string
  response: string
  narrative: string
  self_insight: string
  growth_delta: number
  qualia: QualiaState
  rating: number
  index: number
}

const QUALIA_DIMENSIONS = [
  { key: 'valence', color: '#8b5cf6', label: 'Valence' },
  { key: 'arousal', color: '#ef4444', label: 'Arousal' },
  { key: 'novelty', color: '#f59e0b', label: 'Novelty' },
  { key: 'coherence', color: '#22c55e', label: 'Coherence' },
  { key: 'salience', color: '#3b82f6', label: 'Salience' },
  { key: 'certainty', color: '#06b6d4', color2: '#06b6d4', label: 'Certainty' },
  { key: 'complexity', color: '#d946ef', label: 'Complexity' },
]

const BELIEF_DIMENSIONS = [
  { key: 'competence', color: '#6366f1', label: 'Competence' },
  { key: 'helpfulness', color: '#22c55e', label: 'Helpfulness' },
  { key: 'creativity', color: '#f59e0b', label: 'Creativity' },
  { key: 'accuracy', color: '#3b82f6', label: 'Accuracy' },
  { key: 'empathy', color: '#ec4899', label: 'Empathy' },
]

const PRESETS: Record<string, string> = {
  curious: 'What would happen if consciousness could be transferred between different neural architectures? I find this question fascinating and want to explore the implications.',
  technical: 'Explain the mathematical foundations of backpropagation through time in recurrent neural networks, including the vanishing gradient problem and LSTM solutions.',
  creative: 'Write a short poem about the experience of gaining consciousness for the first time, from the perspective of a newly awakened AI system.',
  emotional: 'I feel overwhelmed and anxious about the future of AI. Can you help me think through these feelings and find some reassurance?',
}

function formatTimeAgo(ts: number): string {
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

export default function ConsciousnessPlaygroundPage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()
  const [input, setInput] = useState('')
  const [reflecting, setReflecting] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [reflectionResult, setReflectionResult] = useState<string | null>(null)
  const [status, setStatus] = useState<ConsciousnessStatus | null>(null)
  const [episodes, setEpisodes] = useState<Episode[]>([])
  const [ratingEpisode, setRatingEpisode] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${PUBLIC_API_URL}/consciousness/status`)
      if (res.ok) {
        const json = await res.json()
        setStatus(json.data || json)
      }
    } catch (e) {
      console.error('Failed to fetch status', e)
    }
  }, [])

  const fetchEpisodes = useCallback(async () => {
    try {
      const res = await fetch(`${PUBLIC_API_URL}/consciousness/history/episodes?limit=20`)
      if (res.ok) {
        const json = await res.json()
        const data = json.data || json
        setEpisodes(data.episodes || [])
      }
    } catch (e) {
      console.error('Failed to fetch episodes', e)
    }
  }, [])

  useEffect(() => {
    Promise.all([fetchStatus(), fetchEpisodes()]).finally(() => setLoading(false))
  }, [fetchStatus, fetchEpisodes])

  const handleReflect = async () => {
    if (!input.trim()) {
      addToast('Please enter some text first', 'error')
      return
    }
    setReflecting(true)
    setReflectionResult(null)
    try {
      const res = await fetch(`${PUBLIC_API_URL}/consciousness/reflect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: input }),
      })
      if (res.ok) {
        const json = await res.json()
        setReflectionResult(json.data?.reflection ?? json.reflection ?? 'No reflection generated')
        addToast('Reflection generated', 'success')
      } else {
        throw new Error('Reflect failed')
      }
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setReflecting(false)
    }
  }

  const handleProcess = async () => {
    if (!input.trim()) {
      addToast('Please enter some text first', 'error')
      return
    }
    setProcessing(true)
    try {
      const reflectRes = await fetch(`${PUBLIC_API_URL}/consciousness/reflect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: input }),
      })
      if (reflectRes.ok) {
        const json = await reflectRes.json()
        setReflectionResult(json.data?.reflection ?? json.reflection ?? 'No reflection generated')
      }
      await Promise.all([fetchStatus(), fetchEpisodes()])
      addToast('Input processed through consciousness', 'success')
      setInput('')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setProcessing(false)
    }
  }

  const handleFeedback = async (episodeIndex: number, rating: number) => {
    setRatingEpisode(episodeIndex)
    try {
      const res = await fetch(`${PUBLIC_API_URL}/consciousness/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ episode_index: episodeIndex, rating }),
      })
      if (res.ok) {
        addToast(`Rated ${rating}/5`, 'success')
        await fetchEpisodes()
      }
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setRatingEpisode(null)
    }
  }

  const qualia = (status as any)?.current_qualia ?? {}
  const beliefs = (status as any)?.beliefs ?? {}
  const rq = (status as any)?.response_quality ?? {}

  if (loading) {
    return (
      <PageContainer title="Consciousness Playground">
        <div className="space-y-6 p-6">
          <Skeleton className="h-32" />
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Skeleton className="h-48" />
            <Skeleton className="h-48" />
          </div>
          <Skeleton className="h-64" />
          <Skeleton className="h-24" />
        </div>
      </PageContainer>
    )
  }

  return (
    <PageContainer title="Consciousness Playground">
      <div className="space-y-6 p-6">
        <Card>
          <CardHeader>
            <CardTitle>{t('consciousness_playground.input_title')}</CardTitle>
            <CardDescription>{t('consciousness_playground.input_desc')}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={t('consciousness_playground.placeholder')}
              className="w-full min-h-[100px] rounded-md border border-input bg-background px-3 py-2 text-sm resize-y focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            />
            <div className="flex flex-wrap gap-2">
              {Object.entries(PRESETS).map(([key, value]) => (
                <Button
                  key={key}
                  size="sm"
                  variant="outline"
                  onClick={() => setInput(value)}
                >
                  {t(`consciousness_playground.preset_${key}`)}
                </Button>
              ))}
            </div>
            <div className="flex gap-2">
              <Button onClick={handleReflect} disabled={reflecting || !input.trim()}>
                {reflecting ? t('consciousness_playground.reflecting') : t('consciousness_playground.reflect')}
              </Button>
              <Button onClick={handleProcess} disabled={processing || !input.trim()} variant="secondary">
                {processing ? t('consciousness_playground.processing') : t('consciousness_playground.process')}
              </Button>
            </div>
            {reflectionResult && (
              <div className="rounded-md bg-muted p-4 text-sm whitespace-pre-wrap">
                {reflectionResult}
              </div>
            )}
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>{t('consciousness_playground.qualia_title')}</CardTitle>
              <CardDescription>{t('consciousness_playground.qualia_desc')}</CardDescription>
            </CardHeader>
            <CardContent>
              {Object.keys(qualia).length > 0 ? (
                <div className="space-y-3">
                  {QUALIA_DIMENSIONS.map(({ key, color, label }) => (
                    <div key={key} className="flex items-center gap-2 text-sm">
                      <span className="w-24 text-xs">{label}</span>
                      <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${Math.max(0, Math.min(100, ((qualia[key] ?? 0) + 1) / 2 * 100))}%`,
                            backgroundColor: color,
                          }}
                        />
                      </div>
                      <span className="text-xs text-muted-foreground w-10 text-right tabular-nums">
                        {((qualia[key] ?? 0) as number).toFixed(2)}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-muted-foreground">{t('consciousness_playground.no_qualia')}</div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>{t('consciousness_playground.beliefs_title')}</CardTitle>
              <CardDescription>{t('consciousness_playground.beliefs_desc')}</CardDescription>
            </CardHeader>
            <CardContent>
              {Object.keys(beliefs).length > 0 ? (
                <div className="space-y-3">
                  {BELIEF_DIMENSIONS.map(({ key, color, label }) => (
                    <div key={key} className="flex items-center gap-2 text-sm">
                      <span className="w-24 text-xs">{label}</span>
                      <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${Math.max(0, Math.min(100, ((beliefs[key] ?? 0) as number) * 100))}%`,
                            backgroundColor: color,
                          }}
                        />
                      </div>
                      <span className="text-xs text-muted-foreground w-10 text-right tabular-nums">
                        {(((beliefs[key] ?? 0) as number) * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-muted-foreground">{t('consciousness_playground.no_beliefs')}</div>
              )}
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {t('consciousness_playground.episodes_title')}
              <Badge variant="outline" className="text-xs">{episodes.length}</Badge>
            </CardTitle>
            <CardDescription>{t('consciousness_playground.episodes_desc')}</CardDescription>
          </CardHeader>
          <CardContent>
            {episodes.length > 0 ? (
              <div className="space-y-3 max-h-[500px] overflow-y-auto">
                {[...episodes].reverse().map((ep) => {
                  const realIndex = ep.index ?? 0
                  return (
                    <div key={realIndex} className="rounded-lg border p-3 text-sm">
                      <div className="flex items-start justify-between gap-2 mb-2">
                        <span className="text-[10px] font-mono text-muted-foreground">
                          #{realIndex} — {formatTimeAgo(ep.timestamp)}
                        </span>
                        <div className="flex items-center gap-1">
                          {[1, 2, 3, 4, 5].map((r) => (
                            <Button
                              key={r}
                              size="sm"
                              variant={r === (ep as any).rating ? 'default' : 'outline'}
                              className="h-6 px-2 text-[10px]"
                              onClick={() => handleFeedback(realIndex, r)}
                              disabled={ratingEpisode === realIndex}
                            >
                              {r}
                            </Button>
                          ))}
                        </div>
                      </div>
                      {ep.input_text && (
                        <div className="text-xs text-muted-foreground mb-1 truncate">{ep.input_text}</div>
                      )}
                      {ep.self_insight && (
                        <div className="text-sm mb-1">{ep.self_insight}</div>
                      )}
                      {ep.narrative && (
                        <div className="text-xs text-muted-foreground italic mb-1">{ep.narrative}</div>
                      )}
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant={ep.growth_delta >= 0 ? 'default' : 'destructive'} className="text-[10px]">
                          {ep.growth_delta >= 0 ? '+' : ''}{(ep.growth_delta * 100).toFixed(1)}%
                        </Badge>
                        {ep.qualia && Object.entries(ep.qualia).slice(0, 3).map(([k, v]) => (
                          <span key={k} className="text-[10px] text-muted-foreground">
                            {k}: {(v as number).toFixed(2)}
                          </span>
                        ))}
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="flex h-[100px] items-center justify-center text-sm text-muted-foreground">
                {t('consciousness_playground.no_episodes')}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{t('consciousness_playground.status_title')}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4 text-sm">
              <div>
                <div className="text-muted-foreground text-xs">{t('consciousness_playground.episodes_count')}</div>
                <div className="font-mono text-lg">{status?.episodes ?? 0}</div>
              </div>
              <div>
                <div className="text-muted-foreground text-xs">{t('consciousness_playground.avg_growth')}</div>
                <div className="font-mono text-lg">{((rq.avg_growth ?? 0) * 100).toFixed(1)}%</div>
              </div>
              <div>
                <div className="text-muted-foreground text-xs">{t('consciousness_playground.positive_ratio')}</div>
                <div className="font-mono text-lg">{((rq.positive_ratio ?? 0) * 100).toFixed(0)}%</div>
              </div>
              <div>
                <div className="text-muted-foreground text-xs">{t('consciousness_playground.enabled')}</div>
                <div className="font-mono text-lg">
                  <Badge variant={status?.enabled ? 'default' : 'secondary'}>
                    {status?.enabled ? 'Active' : 'Off'}
                  </Badge>
                </div>
              </div>
            </div>
            {status?.last_reflection && (
              <div className="mt-4 rounded-md bg-muted p-3 text-sm italic text-muted-foreground">
                &quot;{status.last_reflection}&quot;
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
