'use client'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { PUBLIC_API_URL } from '@/lib/config'
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

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${PUBLIC_API_URL}/consciousness/status`)
      const json = await res.json()
      setStatus(json.data || json)
      setLevel(json.data?.level ?? json.level ?? 0)
    } catch (e) {
      console.error('Failed to fetch consciousness status', e)
    } finally {
      setLoading(false)
    }
  }, [])

  const fetchEval = useCallback(async () => {
    try {
      const res = await fetch(`${PUBLIC_API_URL}/consciousness/evaluate`)
      const json = await res.json()
      setEvalReport(json.data || json)
    } catch (e) {
      console.error('Failed to fetch eval report', e)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
    fetchEval()
  }, [fetchStatus, fetchEval])

  const updateLevel = async (newLevel: number) => {
    setLevel(newLevel)
    try {
      const res = await fetch(`${PUBLIC_API_URL}/consciousness/config`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ level: newLevel }),
      })
      if (!res.ok) throw new Error('Failed to update level')
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
      const res = await fetch(`${PUBLIC_API_URL}/consciousness/reflect`, { method: 'POST' })
      const json = await res.json()
      setReflection(json.data?.reflection ?? json.reflection ?? 'No reflection generated')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setReflecting(false)
    }
  }

  const handleTrain = async () => {
    setTraining(true)
    try {
      const res = await fetch(`${PUBLIC_API_URL}/consciousness/train/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: '' }),
      })
      if (!res.ok) {
        const json = await res.json()
        throw new Error(json.detail?.message || 'Training failed')
      }
      addToast('Consciousness LoRA training finished', 'success')
      fetchStatus()
      fetchEval()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setTraining(false)
    }
  }

  if (loading) {
    return (
      <PageContainer title="Consciousness">
        <div className="space-y-6 p-6">
          {/* Level Control skeleton */}
          <Card><CardContent className="p-4 space-y-3">
            <div className="flex items-center gap-2">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-5 w-16 rounded-full" />
            </div>
            <Skeleton className="h-2 w-48" />
            <Skeleton className="h-1.5 w-full rounded-full" />
          </CardContent></Card>
          {/* Qualia State skeleton: 2x4 grid */}
          <Card><CardContent className="p-4 space-y-3">
            <Skeleton className="h-3 w-24" />
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[1,2,3,4,5,6,7,8].map(i => (
                <div key={i} className="space-y-1">
                  <Skeleton className="h-2 w-12" />
                  <Skeleton className="h-1.5 w-full rounded-full" />
                </div>
              ))}
            </div>
          </CardContent></Card>
          {/* Beliefs + Reflection skeleton: 2-col */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card><CardContent className="p-4 space-y-2">
              <Skeleton className="h-3 w-28" />
              {[1,2,3].map(i => <Skeleton key={i} className="h-3 w-full" />)}
            </CardContent></Card>
            <Card><CardContent className="p-4 space-y-2">
              <Skeleton className="h-3 w-28" />
              <Skeleton className="h-3 w-full" />
              <Skeleton className="h-8 w-24" />
            </CardContent></Card>
          </div>
          {/* Training Status skeleton: 2x2 */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card><CardContent className="p-4 space-y-2">
              <Skeleton className="h-3 w-24" />
              <Skeleton className="h-1.5 w-full rounded-full" />
            </CardContent></Card>
            <Card><CardContent className="p-4 space-y-2">
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
      <div className="space-y-6 p-6">
        {/* Level Control */}
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

        {/* Qualia State */}
        {qualia && (
          <Card>
            <CardHeader>
              <CardTitle>Current Qualia</CardTitle>
              <CardDescription>Real-time emotional and cognitive state</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                {Object.entries(qualia).map(([key, value]) => (
                  <div key={key} className="space-y-1">
                    <div className="text-xs font-medium text-muted-foreground capitalize">
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

        {/* Self-Model Beliefs */}
        {Object.keys(beliefs).length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Self-Model Beliefs</CardTitle>
              <CardDescription>Core beliefs and confidence levels</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {Object.entries(beliefs)
                  .sort(([, a], [, b]) => b - a)
                  .map(([belief, confidence]) => (
                    <div key={belief} className="flex items-center gap-3">
                      <div className="h-2 w-2 rounded-full bg-primary" />
                      <span className="flex-1 text-sm">{belief}</span>
                      <Badge variant="outline" className="text-xs">
                        {(confidence * 100).toFixed(0)}%
                      </Badge>
                    </div>
                  ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Self-Reflection */}
        <Card>
          <CardHeader>
            <CardTitle>Self-Reflection</CardTitle>
            <CardDescription>Trigger a self-reflection to examine current state</CardDescription>
          </CardHeader>
          <CardContent>
            {reflection ? (
              <div className="rounded-md bg-muted p-4 text-sm whitespace-pre-wrap">
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

        {/* Evaluation Report */}
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
              <div className="space-y-3">
                {Object.entries(evalReport.metrics).map(([name, metric]) => (
                  <div key={name} className="flex items-center gap-3">
                    <span className="w-40 text-sm capitalize">{name.replace(/_/g, ' ')}</span>
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

        {/* Training Status */}
        {status?.training && (
          <Card>
            <CardHeader>
              <CardTitle>Training Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4 text-sm">
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

        {/* Narrative */}
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
      </div>
    </PageContainer>
  )
}
