'use client'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { consciousnessController } from '@/lib/consciousness-controller'
import { Badge, Button, Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle, Skeleton, Slider, Switch } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'

interface ConsciousnessStatus {
  enabled: boolean
  level: number
  current_qualia: {
    valence: number
    arousal: number
    novelty: number
    coherence: number
    salience: number
    certainty: number
    complexity: number
  }
  beliefs: Record<string, number>
  episodes: number
  narrative: string
  training: {
    is_training: boolean
    total_pairs: number
    current_epoch: number
    loss: number
  }
}

interface EvalReport {
  overall_score: number
  metrics: Record<string, {
    score: number
    weight: number
    details: string
  }>
}

interface Episode {
  input: string
  response: string
  narrative: string
  qualia: Record<string, number>
  growth_delta: number
  rating: number
  timestamp: string
}

const LEVEL_LABELS = ['Off', 'Basic', 'Full', 'Deep']
const LEVEL_DESCRIPTIONS = [
  'Consciousness system disabled',
  'Basic qualia and self-reflection',
  'Full self-model with belief tracking',
  'Deep reflection with narrative generation',
]

export default function ConsciousnessPage() {
  const addToast = useToastStore(state => state.addToast)
  const [status, setStatus] = useState<ConsciousnessStatus | null>(null)
  const [evalReport, setEvalReport] = useState<EvalReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [training, setTraining] = useState(false)
  const [reflecting, setReflecting] = useState(false)
  const [reflection, setReflection] = useState<string | null>(null)
  const [level, setLevel] = useState(0)
  const [episodes, setEpisodes] = useState<Episode[]>([])
  const [seeding, setSeeding] = useState(false)

  const fetchStatus = useCallback(async () => {
    try {
      const data = await consciousnessController.getStatus()
      setStatus(data as unknown as ConsciousnessStatus)
      setLevel(data.level ?? 0)
    } catch (e) {
      console.error('Failed to fetch consciousness status', e)
    } finally {
      setLoading(false)
    }
  }, [])

  const fetchEval = useCallback(async () => {
    try {
      const data = await consciousnessController.evaluate()
      setEvalReport(data)
    } catch (e) {
      console.error('Failed to fetch eval report', e)
    }
  }, [])

  const fetchEpisodes = useCallback(async () => {
    try {
      const data = await consciousnessController.getEpisodeHistory(10)
      setEpisodes((data.episodes || []).slice(0, 5) as unknown as Episode[])
    } catch (e) {
      console.error('Failed to fetch episodes', e)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
    fetchEval()
    fetchEpisodes()
  }, [fetchStatus, fetchEval, fetchEpisodes])

  const updateLevel = async (newLevel: number) => {
    setLevel(newLevel)
    try {
      await consciousnessController.updateConfig({ level: newLevel })
      addToast(`Consciousness level set to ${LEVEL_LABELS[newLevel]}`, 'success')
      fetchStatus()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
      setLevel(status?.level ?? 0)
    }
  }

  const handleReflect = async () => {
    setReflecting(true)
    setReflection(null)
    try {
      const data = await consciousnessController.reflect()
      setReflection(data.reflection ?? 'No reflection generated')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setReflecting(false)
    }
  }

  const handleTrain = async () => {
    setTraining(true)
    try {
      await consciousnessController.startTraining({ model_path: '' })
      addToast('Consciousness LoRA training finished', 'success')
      fetchStatus()
      fetchEval()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setTraining(false)
    }
  }

  const handleSeed = async () => {
    setSeeding(true)
    try {
      await consciousnessController.seedData({ count: 10 })
      addToast('Seeded 10 training pairs', 'success')
      fetchStatus()
      fetchEpisodes()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setSeeding(false)
    }
  }

  const handleExport = async () => {
    try {
      const data = await consciousnessController.getStatus()
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `consciousness-status-${Date.now()}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      addToast('Exported consciousness status', 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }

  const handleRefresh = () => {
    setLoading(true)
    fetchStatus()
    fetchEval()
    fetchEpisodes()
  }

  if (loading) {
    return (
      <PageContainer title="Consciousness">
        <div className="space-y-4 sm:space-y-6 p-4 sm:p-6">
          <div className="flex items-center gap-2">
            <Skeleton className="h-8 w-20" />
            <Skeleton className="h-8 w-20" />
            <Skeleton className="h-8 w-20" />
          </div>
          <Card><CardContent className="p-3 sm:p-4 space-y-3">
            <div className="flex items-center gap-2">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-5 w-16 rounded-full" />
            </div>
            <Skeleton className="h-2 w-48" />
            <Skeleton className="h-1.5 w-full rounded-full" />
          </CardContent></Card>
          <Card><CardContent className="p-3 sm:p-4 space-y-3">
            <Skeleton className="h-3 w-24" />
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 sm:gap-3">
              {[1,2,3,4,5,6,7,8].map(i => (
                <div key={i} className="space-y-1">
                  <Skeleton className="h-2 w-12" />
                  <Skeleton className="h-1.5 w-full rounded-full" />
                </div>
              ))}
            </div>
          </CardContent></Card>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 sm:gap-3">
            <Card><CardContent className="p-3 sm:p-4 space-y-2">
              <Skeleton className="h-3 w-28" />
              {[1,2,3].map(i => <Skeleton key={i} className="h-3 w-full" />)}
            </CardContent></Card>
            <Card><CardContent className="p-3 sm:p-4 space-y-2">
              <Skeleton className="h-3 w-28" />
              <Skeleton className="h-3 w-full" />
              <Skeleton className="h-8 w-24" />
            </CardContent></Card>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 sm:gap-3">
            <Card><CardContent className="p-3 sm:p-4 space-y-2">
              <Skeleton className="h-3 w-24" />
              <Skeleton className="h-1.5 w-full rounded-full" />
            </CardContent></Card>
            <Card><CardContent className="p-3 sm:p-4 space-y-2">
              <Skeleton className="h-3 w-24" />
              <Skeleton className="h-1.5 w-full rounded-full" />
            </CardContent></Card>
          </div>
        </div>
      </PageContainer>
    )
  }

  const qualia = status?.current_qualia
  const beliefs = status?.beliefs ?? {}

  return (
    <PageContainer title="Consciousness">
      <div className="space-y-4 sm:space-y-6 p-4 sm:p-6">
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={handleSeed} disabled={seeding} variant="secondary" size="sm">
            {seeding ? 'Seeding...' : 'Seed Data'}
          </Button>
          <Button onClick={handleExport} variant="secondary" size="sm">
            Export
          </Button>
          <Button onClick={handleRefresh} variant="ghost" size="sm">
            Refresh
          </Button>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Consciousness Level
              <Badge variant={status?.enabled ? 'default' : 'secondary'}>
                {LEVEL_LABELS[level]}
              </Badge>
            </CardTitle>
            <CardDescription>
              {LEVEL_DESCRIPTIONS[level]}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center gap-4">
                <span className="text-sm text-muted-foreground w-12">Off</span>
                <Slider
                  value={[level]}
                  onValueChange={([v]) => updateLevel(v)}
                  min={0}
                  max={3}
                  step={1}
                  className="flex-1"
                />
                <span className="text-sm text-muted-foreground w-12">Deep</span>
              </div>
              <div className="flex items-center gap-2">
                <Switch
                  checked={level > 0}
                  onCheckedChange={(checked) => updateLevel(checked ? 1 : 0)}
                />
                <span className="text-sm text-muted-foreground">
                  {status?.enabled ? 'Active' : 'Disabled'}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        {qualia && (
          <Card>
            <CardHeader>
              <CardTitle>Current Qualia</CardTitle>
              <CardDescription>Real-time emotional and cognitive state</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex justify-center mb-6">
                <svg viewBox="-110 -110 220 220" className="w-40 h-40 sm:w-56 sm:h-56">
                  {[0.2, 0.4, 0.6, 0.8, 1.0].map((r, i) => (
                    <polygon
                      key={i}
                      points={Array.from({ length: 7 }, (_, k) => {
                        const angle = (k * 2 * Math.PI) / 7 - Math.PI / 2
                        return `${Math.cos(angle) * r * 100},${Math.sin(angle) * r * 100}`
                      }).join(' ')}
                      fill="none"
                      stroke="hsl(var(--border))"
                      strokeWidth="0.5"
                    />
                  ))}
                  {Array.from({ length: 7 }, (_, k) => {
                    const angle = (k * 2 * Math.PI) / 7 - Math.PI / 2
                    return (
                      <line
                        key={k}
                        x1="0"
                        y1="0"
                        x2={Math.cos(angle) * 100}
                        y2={Math.sin(angle) * 100}
                        stroke="hsl(var(--border))"
                        strokeWidth="0.5"
                      />
                    )
                  })}
                  <polygon
                    points={Object.entries(qualia).map(([, value], k) => {
                      const angle = (k * 2 * Math.PI) / 7 - Math.PI / 2
                      const clamped = Math.max(0, Math.min(1, value as number))
                      return `${Math.cos(angle) * clamped * 100},${Math.sin(angle) * clamped * 100}`
                    }).join(' ')}
                    fill="hsl(var(--primary) / 0.2)"
                    stroke="hsl(var(--primary))"
                    strokeWidth="1.5"
                  />
                  {Object.entries(qualia).map(([key, value], k) => {
                    const angle = (k * 2 * Math.PI) / 7 - Math.PI / 2
                    const clamped = Math.max(0, Math.min(1, value as number))
                    const labelR = 112
                    return (
                      <g key={key}>
                        <circle
                          cx={Math.cos(angle) * clamped * 100}
                          cy={Math.sin(angle) * clamped * 100}
                          r="3"
                          fill="hsl(var(--primary))"
                        />
                        <text
                          x={Math.cos(angle) * labelR}
                          y={Math.sin(angle) * labelR}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          className="fill-muted-foreground"
                          fontSize="8"
                        >
                          {key.slice(0, 3).toUpperCase()}
                        </text>
                      </g>
                    )
                  })}
                </svg>
              </div>
              <div className="grid grid-cols-2 gap-2 sm:gap-3 sm:grid-cols-4">
                {Object.entries(qualia).map(([key, value]) => (
                  <div key={key} className="space-y-1">
                    <div className="text-[10px] sm:text-xs font-medium text-muted-foreground capitalize">
                      {key}
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="h-2 flex-1 rounded-full bg-secondary">
                        <div
                          className="h-full rounded-full bg-primary transition-all"
                          style={{ width: `${Math.max(0, Math.min(1, value as number)) * 100}%` }}
                        />
                      </div>
                      <span className="text-xs tabular-nums text-muted-foreground">
                        {(value as number).toFixed(2)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {Object.keys(beliefs).length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Self-Model Beliefs</CardTitle>
              <CardDescription>Core beliefs and confidence levels</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-1.5 sm:space-y-2">
                {Object.entries(beliefs)
                  .sort(([, a], [, b]) => b - a)
                  .map(([belief, confidence]) => (
                    <div key={belief} className="flex items-center gap-2 sm:gap-3">
                      <div className="h-2 w-2 rounded-full bg-primary shrink-0" />
                      <span className="flex-1 text-xs sm:text-sm truncate">{belief}</span>
                      <Badge variant="outline" className="text-xs">
                        {(confidence * 100).toFixed(0)}%
                      </Badge>
                    </div>
                  ))}
              </div>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <CardTitle>Self-Reflection</CardTitle>
            <CardDescription>Trigger a self-reflection to examine current state</CardDescription>
          </CardHeader>
          <CardContent>
            {reflection ? (
              <div className="rounded-md bg-muted p-3 sm:p-4 text-xs sm:text-sm whitespace-pre-wrap">
                {reflection}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                No reflection yet. Click the button to generate one.
              </p>
            )}
          </CardContent>
          <CardFooter className="flex gap-2">
            <Button
              onClick={handleReflect}
              disabled={reflecting || !status?.enabled}
              variant="secondary"
            >
              {reflecting ? 'Reflecting...' : 'Reflect'}
            </Button>
            <Button
              onClick={handleTrain}
              disabled={training || !status?.enabled}
            >
              {training ? 'Training...' : 'Train LoRA'}
            </Button>
          </CardFooter>
        </Card>

        {evalReport && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                Evaluation Report
                <Badge variant={evalReport.overall_score >= 70 ? 'default' : 'secondary'}>
                  Score: {evalReport.overall_score.toFixed(0)}/100
                </Badge>
              </CardTitle>
              <CardDescription>Quality metrics for the consciousness system</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 sm:space-y-3">
                {Object.entries(evalReport.metrics).map(([name, metric]) => (
                  <div key={name} className="flex items-center gap-2 sm:gap-3">
                    <span className="w-28 sm:w-40 text-xs sm:text-sm capitalize">{name.replace(/_/g, ' ')}</span>
                    <div className="h-2 flex-1 rounded-full bg-secondary">
                      <div
                        className="h-full rounded-full bg-primary transition-all"
                        style={{ width: `${Math.min(100, metric.score)}%` }}
                      />
                    </div>
                    <span className="w-10 text-right text-xs tabular-nums text-muted-foreground">
                      {metric.score.toFixed(0)}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
            <CardFooter>
              <Button onClick={fetchEval} variant="ghost" size="sm">
                Refresh
              </Button>
            </CardFooter>
          </Card>
        )}

        {status?.training && (
          <Card>
            <CardHeader>
              <CardTitle>Training Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-2 sm:gap-3 text-xs sm:text-sm">
                <div>
                  <span className="text-muted-foreground">Status: </span>
                  <Badge variant={status.training.is_training ? 'default' : 'secondary'}>
                    {status.training.is_training ? 'Training' : 'Idle'}
                  </Badge>
                </div>
                <div>
                  <span className="text-muted-foreground">Training Pairs: </span>
                  <span className="font-mono">{status.training.total_pairs}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Episodes: </span>
                  <span className="font-mono">{status.episodes}</span>
                </div>
                {status.training.loss > 0 && (
                  <div>
                    <span className="text-muted-foreground">Loss: </span>
                    <span className="font-mono">{status.training.loss.toFixed(4)}</span>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {status?.narrative && (
          <Card>
            <CardHeader>
              <CardTitle>Narrative</CardTitle>
              <CardDescription>Auto-generated consciousness narrative</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                {status.narrative}
              </p>
            </CardContent>
          </Card>
        )}

        {episodes.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Recent Episodes</CardTitle>
              <CardDescription>Last 5 consciousness episodes</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {episodes.map((ep, i) => (
                  <div key={i} className="rounded-md border p-2 sm:p-3 space-y-1.5 sm:space-y-2">
                    <div className="flex items-start justify-between gap-2">
                      <p className="text-sm flex-1 line-clamp-2">{ep.input}</p>
                      <div className="flex items-center gap-0.5 shrink-0">
                        {Array.from({ length: 5 }, (_, s) => (
                          <svg key={s} className={`w-3 h-3 sm:w-3.5 sm:h-3.5 ${s < ep.rating ? 'fill-amber-400' : 'fill-muted'}`} viewBox="0 0 20 20">
                            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                          </svg>
                        ))}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant={ep.growth_delta >= 0 ? 'default' : 'destructive'} className="text-xs">
                        {ep.growth_delta >= 0 ? '+' : ''}{ep.growth_delta.toFixed(3)}
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        {new Date(ep.timestamp).toLocaleString()}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </PageContainer>
  )
}
