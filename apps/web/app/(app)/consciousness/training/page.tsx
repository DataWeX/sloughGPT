'use client'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { PUBLIC_API_URL } from '@/lib/config'
import {
  Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, Skeleton,
  Input,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'

interface TrainStatus {
  pairs_collected: number
  min_pairs: number
  should_train: boolean
  is_training: boolean
  training_runs: TrainingRun[]
  last_result: TrainingResult | null
}

interface TrainingRun {
  timestamp: string
  status: string
  pairs_count: number
  loss: number
  adapter_path: string
  elapsed_seconds: number
}

interface TrainingResult {
  status: string
  loss: number
  adapter_path: string
  elapsed_seconds: number
}

interface EvalReport {
  overall_score: number
  narrative_coherence: number
  qualia_richness: number
  belief_stability: number
  self_reflection_depth: number
  growth_trajectory: number
  curiosity_engagement: number
  feedback_alignment: number
  episode_count: number
  diagnostics: string[]
}

export default function ConsciousnessTrainingPage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()
  const [loading, setLoading] = useState(true)
  const [trainStatus, setTrainStatus] = useState<TrainStatus | null>(null)
  const [evalReport, setEvalReport] = useState<EvalReport | null>(null)
  const [training, setTraining] = useState(false)
  const [modelPath, setModelPath] = useState('')

  const fetchTrainStatus = useCallback(async () => {
    try {
      const res = await fetch(`${PUBLIC_API_URL}/consciousness/train/status`)
      const json = await res.json()
      setTrainStatus(json.data || json)
    } catch (e) {
      console.error('Failed to fetch train status', e)
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
    Promise.all([fetchTrainStatus(), fetchEval()]).finally(() => setLoading(false))
  }, [fetchTrainStatus, fetchEval])

  const handleStartTraining = async () => {
    setTraining(true)
    try {
      const res = await fetch(`${PUBLIC_API_URL}/consciousness/train/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: modelPath }),
      })
      if (!res.ok) {
        const json = await res.json()
        throw new Error(json.detail?.message || 'Training failed')
      }
      addToast(t('consciousness_training.toast_started'), 'success')
      await Promise.all([fetchTrainStatus(), fetchEval()])
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setTraining(false)
    }
  }

  const canTrain = trainStatus && !trainStatus.is_training && trainStatus.pairs_collected >= trainStatus.min_pairs

  const getScoreColor = (score: number) => {
    if (score > 70) return 'text-green-500'
    if (score > 40) return 'text-yellow-500'
    return 'text-red-500'
  }

  const getScoreBg = (score: number) => {
    if (score > 70) return 'bg-green-500'
    if (score > 40) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  const getBarColor = (score: number) => {
    if (score > 70) return 'bg-green-500'
    if (score > 40) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  if (loading) {
    return (
      <PageContainer title={t('consciousness_training.page_title')}>
        <div className="space-y-6 p-6">
          <Card><CardContent className="p-4 space-y-3">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-1.5 w-full rounded-full" />
            <Skeleton className="h-4 w-48" />
          </CardContent></Card>
          <Card><CardContent className="p-4 space-y-3">
            <Skeleton className="h-4 w-40" />
            <Skeleton className="h-1.5 w-full rounded-full" />
          </CardContent></Card>
          <Card><CardContent className="p-4 space-y-2">
            <Skeleton className="h-4 w-36" />
            {[1, 2, 3].map(i => <Skeleton key={i} className="h-3 w-full" />)}
          </CardContent></Card>
        </div>
      </PageContainer>
    )
  }

  const pairsProgress = trainStatus
    ? Math.min(100, (trainStatus.pairs_collected / Math.max(1, trainStatus.min_pairs)) * 100)
    : 0

  const evalMetrics = evalReport ? [
    { key: 'narrative_coherence', value: evalReport.narrative_coherence },
    { key: 'qualia_richness', value: evalReport.qualia_richness },
    { key: 'belief_stability', value: evalReport.belief_stability },
    { key: 'self_reflection_depth', value: evalReport.self_reflection_depth },
    { key: 'growth_trajectory', value: evalReport.growth_trajectory },
    { key: 'curiosity_engagement', value: evalReport.curiosity_engagement },
    { key: 'feedback_alignment', value: evalReport.feedback_alignment },
  ] : []

  return (
    <PageContainer title={t('consciousness_training.page_title')}>
      <div className="space-y-6 p-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {t('consciousness_training.status_title')}
              {trainStatus?.is_training && (
                <Badge variant="default">{t('consciousness_training.training_active')}</Badge>
              )}
            </CardTitle>
            <CardDescription>{t('consciousness_training.status_desc')}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">{t('consciousness_training.pairs_collected')}</span>
                <span className="font-mono">
                  {trainStatus?.pairs_collected ?? 0} / {trainStatus?.min_pairs ?? 0}
                </span>
              </div>
              <div className="h-2 rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full rounded-full bg-primary transition-all duration-500"
                  style={{ width: `${pairsProgress}%` }}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">{t('consciousness_training.training_runs')}: </span>
                <span className="font-mono">{trainStatus?.training_runs?.length ?? 0}</span>
              </div>
              <div>
                <span className="text-muted-foreground">{t('consciousness_training.episodes')}: </span>
                <span className="font-mono">{evalReport?.episode_count ?? 0}</span>
              </div>
            </div>

            {trainStatus?.last_result && (
              <div className="rounded-md bg-muted p-3 text-sm space-y-1">
                <div className="font-medium">{t('consciousness_training.last_result')}</div>
                <div className="flex gap-4">
                  <span className="text-muted-foreground">{t('consciousness_training.status')}: </span>
                  <Badge variant={trainStatus.last_result.status === 'success' ? 'default' : 'secondary'}>
                    {trainStatus.last_result.status}
                  </Badge>
                </div>
                {trainStatus.last_result.loss > 0 && (
                  <div>
                    <span className="text-muted-foreground">{t('consciousness_training.loss')}: </span>
                    <span className="font-mono">{trainStatus.last_result.loss.toFixed(4)}</span>
                  </div>
                )}
                {trainStatus.last_result.adapter_path && (
                  <div>
                    <span className="text-muted-foreground">{t('consciousness_training.adapter')}: </span>
                    <span className="font-mono text-xs">{trainStatus.last_result.adapter_path}</span>
                  </div>
                )}
                {trainStatus.last_result.elapsed_seconds > 0 && (
                  <div>
                    <span className="text-muted-foreground">{t('consciousness_training.elapsed')}: </span>
                    <span className="font-mono">{trainStatus.last_result.elapsed_seconds.toFixed(1)}s</span>
                  </div>
                )}
              </div>
            )}

            <div className="space-y-2">
              <label className="text-sm text-muted-foreground">{t('consciousness_training.model_path')}</label>
              <Input
                value={modelPath}
                onChange={(e) => setModelPath(e.target.value)}
                placeholder={t('consciousness_training.model_path_placeholder')}
                disabled={training}
              />
            </div>

            <Button
              onClick={handleStartTraining}
              disabled={!canTrain || training}
              className="w-full"
            >
              {training ? t('consciousness_training.training_in_progress') : t('consciousness_training.start_training')}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {t('consciousness_training.eval_title')}
              {evalReport && (
                <span className={`text-2xl font-bold ${getScoreColor(evalReport.overall_score)}`}>
                  {evalReport.overall_score.toFixed(0)}
                </span>
              )}
            </CardTitle>
            <CardDescription>{t('consciousness_training.eval_desc')}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {evalReport && (
              <>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">{t('consciousness_training.overall_score')}</span>
                    <span className={`font-mono font-bold ${getScoreColor(evalReport.overall_score)}`}>
                      {evalReport.overall_score.toFixed(0)}/100
                    </span>
                  </div>
                  <div className="h-3 rounded-full bg-muted overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${getScoreBg(evalReport.overall_score)}`}
                      style={{ width: `${evalReport.overall_score}%` }}
                    />
                  </div>
                </div>

                <div className="space-y-3">
                  {evalMetrics.map(({ key, value }) => (
                    <div key={key} className="space-y-1">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground capitalize">{key.replace(/_/g, ' ')}</span>
                        <span className="font-mono text-xs">{value.toFixed(0)}</span>
                      </div>
                      <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${getBarColor(value)}`}
                          style={{ width: `${Math.min(100, value)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                {evalReport.diagnostics && evalReport.diagnostics.length > 0 && (
                  <div className="space-y-2">
                    <div className="text-sm font-medium">{t('consciousness_training.diagnostics')}</div>
                    <div className="space-y-1">
                      {evalReport.diagnostics.map((d: string, i: number) => (
                        <div key={i} className="text-xs text-muted-foreground rounded-md bg-muted p-2">{d}</div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </CardContent>
          <CardContent>
            <Button onClick={fetchEval} variant="ghost" size="sm">
              {t('consciousness_training.refresh')}
            </Button>
          </CardContent>
        </Card>

        {trainStatus?.training_runs && trainStatus.training_runs.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>{t('consciousness_training.history_title')}</CardTitle>
              <CardDescription>{t('consciousness_training.history_desc')}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b text-left text-muted-foreground">
                      <th className="pb-2 font-medium">{t('consciousness_training.col_timestamp')}</th>
                      <th className="pb-2 font-medium">{t('consciousness_training.col_status')}</th>
                      <th className="pb-2 font-medium">{t('consciousness_training.col_pairs')}</th>
                      <th className="pb-2 font-medium">{t('consciousness_training.col_loss')}</th>
                      <th className="pb-2 font-medium">{t('consciousness_training.col_elapsed')}</th>
                      <th className="pb-2 font-medium">{t('consciousness_training.col_adapter')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trainStatus.training_runs.map((run, i) => (
                      <tr key={i} className="border-b last:border-0">
                        <td className="py-2 text-xs text-muted-foreground">
                          {new Date(run.timestamp).toLocaleString()}
                        </td>
                        <td className="py-2">
                          <Badge variant={run.status === 'success' ? 'default' : 'secondary'}>
                            {run.status}
                          </Badge>
                        </td>
                        <td className="py-2 font-mono">{run.pairs_count}</td>
                        <td className="py-2 font-mono">{run.loss > 0 ? run.loss.toFixed(4) : '—'}</td>
                        <td className="py-2 font-mono">{run.elapsed_seconds > 0 ? `${run.elapsed_seconds.toFixed(1)}s` : '—'}</td>
                        <td className="py-2 font-mono text-xs max-w-[200px] truncate">{run.adapter_path || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </PageContainer>
  )
}
