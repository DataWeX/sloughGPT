'use client'

import { useState, useEffect, useCallback, useMemo } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { consciousnessController } from '@/lib/consciousness-controller'
import {
  Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, Skeleton,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'
import { IconSparkle, IconRefresh, IconDownload } from '@/components/icons/NavIcons'

interface Insight {
  id: string
  title: string
  severity: 'info' | 'warning' | 'success'
  description: string
  suggestion: string
  detail: string
  score: number
}

const QUALIA_DIMS = ['valence', 'arousal', 'novelty', 'coherence'] as const
const BELIEF_DIMS = ['competence', 'helpfulness', 'creativity', 'accuracy', 'empathy'] as const

function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)) }

export default function ConsciousnessInsightsPage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [expandedCard, setExpandedCard] = useState<string | null>(null)

  const [episodes, setEpisodes] = useState<any[]>([])
  const [healthData, setHealthData] = useState<any>(null)
  const [evalData, setEvalData] = useState<any>(null)
  const [personalityData, setPersonalityData] = useState<any>(null)
  const [statsData, setStatsData] = useState<any>(null)

  const fetchAll = useCallback(async () => {
    try {
      const [epResult, healthResult, evalResult, personalityResult, statsResult] = await Promise.allSettled([
        consciousnessController.getEpisodeHistory(100),
        consciousnessController.healthCheck(),
        consciousnessController.evaluate(),
        consciousnessController.getPersonality(),
        (await import('@/lib/http-client')).apiGet('/consciousness/stats') as any,
      ])

      if (epResult.status === 'fulfilled') {
        setEpisodes((epResult.value as any)?.episodes ?? [])
      }
      if (healthResult.status === 'fulfilled') {
        setHealthData(healthResult.value)
      }
      if (evalResult.status === 'fulfilled') {
        setEvalData(evalResult.value)
      }
      if (personalityResult.status === 'fulfilled') {
        setPersonalityData(personalityResult.value)
      }
      if (statsResult.status === 'fulfilled') {
        setStatsData(statsResult.value)
      }
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [addToast])

  useEffect(() => { fetchAll() }, [fetchAll])

  const handleRefresh = useCallback(() => {
    setRefreshing(true)
    fetchAll()
  }, [fetchAll])

  const computedInsights = useMemo<Insight[]>(() => {
    const insights: Insight[] = []

    if (episodes.length >= 2) {
      const growths = episodes.map((e: any) => e.growth_delta)
      const half = Math.floor(growths.length / 2)
      const firstHalf = growths.slice(0, half).reduce((a: number, b: number) => a + b, 0) / (half || 1)
      const secondHalf = growths.slice(half).reduce((a: number, b: number) => a + b, 0) / ((growths.length - half) || 1)
      const improving = secondHalf > firstHalf + 0.005
      const declining = secondHalf < firstHalf - 0.005
      const trend = improving ? 'improving' : declining ? 'declining' : 'stable'

      insights.push({
        id: 'growth-trend',
        title: t('consciousness_insights.growth_trend'),
        severity: improving ? 'success' : declining ? 'warning' : 'info',
        description: improving
          ? t('consciousness_insights.growth_improving')
          : declining
          ? t('consciousness_insights.growth_declining')
          : t('consciousness_insights.growth_stable'),
        suggestion: improving
          ? t('consciousness_insights.growth_improving_suggest')
          : declining
          ? t('consciousness_insights.growth_declining_suggest')
          : t('consciousness_insights.growth_stable_suggest'),
        detail: `${t('consciousness_insights.growth_first_half')}: ${(firstHalf * 100).toFixed(2)}% → ${t('consciousness_insights.growth_second_half')}: ${(secondHalf * 100).toFixed(2)}%`,
        score: improving ? 90 : declining ? 40 : 70,
      })
    }

    if (healthData) {
      const qualia = healthData.qualia ?? {}
      const activeDims = QUALIA_DIMS.filter((d: string) => Math.abs(qualia[d] ?? 0) > 0.3)
      const dormantDims = QUALIA_DIMS.filter((d: string) => Math.abs(qualia[d] ?? 0) < 0.1)

      insights.push({
        id: 'qualia-balance',
        title: t('consciousness_insights.qualia_balance'),
        severity: dormantDims.length > 2 ? 'warning' : 'success',
        description: dormantDims.length > 0
          ? `${t('consciousness_insights.qualia_dormant')}: ${dormantDims.join(', ')}`
          : t('consciousness_insights.qualia_all_active'),
        suggestion: dormantDims.length > 0
          ? t('consciousness_insights.qualia_suggest')
          : t('consciousness_insights.qualia_maintain'),
        detail: `${t('consciousness_insights.qualia_active')}: ${activeDims.join(', ') || '—'} | ${t('consciousness_insights.qualia_dormant')}: ${dormantDims.join(', ') || '—'}`,
        score: dormantDims.length > 2 ? 45 : 85,
      })
    }

    if (healthData) {
      const beliefs = healthData.qualia ?? {}
      const avgBelief = BELIEF_DIMS.reduce((sum: number, d: string) => sum + (beliefs[d] ?? 0.5), 0) / BELIEF_DIMS.length
      const variance = BELIEF_DIMS.reduce((sum: number, d: string) => sum + Math.pow((beliefs[d] ?? 0.5) - avgBelief, 2), 0) / BELIEF_DIMS.length
      const stable = variance < 0.02

      insights.push({
        id: 'belief-stability',
        title: t('consciousness_insights.belief_stability'),
        severity: stable ? 'success' : 'warning',
        description: stable
          ? t('consciousness_insights.beliefs_stable')
          : t('consciousness_insights.beliefs_fluctuating'),
        suggestion: stable
          ? t('consciousness_insights.beliefs_stable_suggest')
          : t('consciousness_insights.beliefs_fluctuating_suggest'),
        detail: `${t('consciousness_insights.beliefs_avg')}: ${(avgBelief * 100).toFixed(1)}% | ${t('consciousness_insights.beliefs_variance')}: ${(variance * 1000).toFixed(2)}`,
        score: stable ? 85 : 55,
      })
    }

    if (personalityData && evalData) {
      const evalScore = evalData.overall_score ?? 50
      const aligned = evalScore > 60

      insights.push({
        id: 'personality-alignment',
        title: t('consciousness_insights.personality_alignment'),
        severity: aligned ? 'success' : 'warning',
        description: aligned
          ? t('consciousness_insights.personality_aligned')
          : t('consciousness_insights.personality_misaligned'),
        suggestion: aligned
          ? t('consciousness_insights.personality_aligned_suggest')
          : t('consciousness_insights.personality_misaligned_suggest'),
        detail: `${t('consciousness_insights.personality_eval_score')}: ${evalScore.toFixed(0)}/100`,
        score: aligned ? 80 : 45,
      })
    }

    if (episodes.length > 0) {
      const rated = episodes.filter((e: any) => e.rating > 0)
      if (rated.length > 0) {
        const avgRating = rated.reduce((s: number, e: any) => s + e.rating, 0) / rated.length
        const growths = rated.map((e: any) => e.growth_delta)
        const avgGrowth = growths.reduce((a: number, b: number) => a + b, 0) / growths.length
        const consistent = (avgRating > 3 && avgGrowth > 0) || (avgRating < 3 && avgGrowth < 0)

        insights.push({
          id: 'feedback-quality',
          title: t('consciousness_insights.feedback_quality'),
          severity: consistent ? 'success' : 'warning',
          description: consistent
            ? t('consciousness_insights.feedback_consistent')
            : t('consciousness_insights.feedback_inconsistent'),
          suggestion: consistent
            ? t('consciousness_insights.feedback_consistent_suggest')
            : t('consciousness_insights.feedback_inconsistent_suggest'),
          detail: `${t('consciousness_insights.feedback_avg_rating')}: ${avgRating.toFixed(1)}/5 | ${t('consciousness_insights.feedback_avg_growth')}: ${(avgGrowth * 100).toFixed(1)}%`,
          score: consistent ? 80 : 50,
        })
      }
    }

    if (episodes.length > 0) {
      const inputs = episodes.map((e: any) => (e.input_text ?? '').toLowerCase().slice(0, 50))
      const unique = new Set(inputs).size
      const diversity = unique / inputs.length

      insights.push({
        id: 'episode-diversity',
        title: t('consciousness_insights.episode_diversity'),
        severity: diversity > 0.7 ? 'success' : diversity > 0.4 ? 'info' : 'warning',
        description: diversity > 0.7
          ? t('consciousness_insights.diversity_high')
          : diversity > 0.4
          ? t('consciousness_insights.diversity_moderate')
          : t('consciousness_insights.diversity_low'),
        suggestion: diversity < 0.7
          ? t('consciousness_insights.diversity_suggest')
          : t('consciousness_insights.diversity_maintain'),
        detail: `${t('consciousness_insights.diversity_unique')}: ${unique}/${inputs.length} (${(diversity * 100).toFixed(0)}%)`,
        score: clamp(diversity * 100, 0, 100),
      })
    }

    const lowScoreInsights = insights.filter(i => i.score < 60)
    if (lowScoreInsights.length > 0) {
      insights.push({
        id: 'recommendation',
        title: t('consciousness_insights.recommendation'),
        severity: lowScoreInsights.length > 2 ? 'warning' : 'info',
        description: t('consciousness_insights.recommendation_desc'),
        suggestion: lowScoreInsights.map(i => i.suggestion).join('; '),
        detail: lowScoreInsights.map(i => `${i.title}: ${i.score}/100`).join(' | '),
        score: Math.max(...lowScoreInsights.map(i => i.score)),
      })
    }

    return insights
  }, [episodes, healthData, evalData, personalityData, t])

  const handleExport = useCallback(() => {
    try {
      const data = {
        episodes: episodes.length,
        health: healthData,
        evaluation: evalData,
        personality: personalityData,
        stats: statsData,
        insights: computedInsights,
        exportedAt: new Date().toISOString(),
      }
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `consciousness-insights-${Date.now()}.json`
      a.click()
      URL.revokeObjectURL(url)
      addToast(t('consciousness_insights.toast_exported'), 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }, [episodes, healthData, evalData, personalityData, statsData, computedInsights, addToast, t])

  const overallScore = useMemo(() => {
    if (computedInsights.length === 0) return 0
    return Math.round(computedInsights.reduce((s, i) => s + i.score, 0) / computedInsights.length)
  }, [computedInsights])

  const scoreTrend = useMemo(() => {
    if (episodes.length < 4) return 'stable'
    const growths = episodes.map((e: any) => e.growth_delta)
    const half = Math.floor(growths.length / 2)
    const firstHalf = growths.slice(0, half).reduce((a: number, b: number) => a + b, 0) / (half || 1)
    const secondHalf = growths.slice(half).reduce((a: number, b: number) => a + b, 0) / ((growths.length - half) || 1)
    return secondHalf > firstHalf + 0.005 ? 'up' : secondHalf < firstHalf - 0.005 ? 'down' : 'stable'
  }, [episodes])

  if (loading) {
    return (
      <PageContainer title={t('consciousness_insights.page_title')}>
        <div className="space-y-6 p-6">
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Skeleton className="h-40" />
            <Skeleton className="h-40" />
            <Skeleton className="h-40" />
            <Skeleton className="h-40" />
          </div>
        </div>
      </PageContainer>
    )
  }

  return (
    <PageContainer title={t('consciousness_insights.page_title')}>
      <div className="space-y-6 p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <Card className="px-4 py-2">
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">{t('consciousness_insights.overall_score')}</span>
                <span className={`text-2xl font-bold ${
                  overallScore > 70 ? 'text-green-500' : overallScore > 40 ? 'text-yellow-500' : 'text-red-500'
                }`}>{overallScore}</span>
                <span className="text-xs text-muted-foreground">/100</span>
                <span className={`text-xs ${
                  scoreTrend === 'up' ? 'text-green-500' : scoreTrend === 'down' ? 'text-red-500' : 'text-muted-foreground'
                }`}>
                  {scoreTrend === 'up' ? '↑' : scoreTrend === 'down' ? '↓' : '→'}
                </span>
              </div>
            </Card>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={handleRefresh} disabled={refreshing}>
              <IconRefresh className={`h-3 w-3 mr-1 ${refreshing ? 'animate-spin' : ''}`} />
              {t('consciousness_insights.refresh')}
            </Button>
            <Button variant="outline" size="sm" onClick={handleExport}>
              <IconDownload className="h-3 w-3 mr-1" />
              {t('consciousness_insights.export')}
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {computedInsights.map(insight => (
            <Card
              key={insight.id}
              className="cursor-pointer transition-colors hover:bg-muted/50"
              onClick={() => setExpandedCard(expandedCard === insight.id ? null : insight.id)}
            >
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2">
                    <IconSparkle className="h-4 w-4 text-muted-foreground" />
                    <CardTitle className="text-sm">{insight.title}</CardTitle>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge
                      variant={insight.severity === 'success' ? 'default' : insight.severity === 'warning' ? 'outline' : 'secondary'}
                      className="text-[10px]"
                    >
                      {insight.severity}
                    </Badge>
                    <span className="text-xs font-mono text-muted-foreground">{insight.score}</span>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="text-xs text-muted-foreground">{insight.description}</div>
                  <div className="text-xs font-medium">{insight.suggestion}</div>
                  {expandedCard === insight.id && (
                    <div className="mt-2 rounded border border-border/30 bg-muted/10 p-2 text-[10px] text-muted-foreground font-mono">
                      {insight.detail}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {computedInsights.length === 0 && (
          <Card>
            <CardContent className="flex h-32 items-center justify-center text-sm text-muted-foreground">
              {t('consciousness_insights.no_data')}
            </CardContent>
          </Card>
        )}
      </div>
    </PageContainer>
  )
}
