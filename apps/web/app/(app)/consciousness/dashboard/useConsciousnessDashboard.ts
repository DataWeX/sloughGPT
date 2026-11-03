'use client'

// Stateful core of the consciousness dashboard, extracted from page.tsx by the
// page-decomposition refactor: all state, fetching, live-event folding, and
// action handlers live here — the page and its components stay presentational.

import { useState, useEffect, useCallback } from 'react'
import { consciousnessController } from '@/lib/consciousness-controller'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useConsciousnessLive } from '@/hooks/useConsciousnessLive'
import {
  formatTime,
  type Episode,
  type QualiaPoint,
  type BeliefsPoint,
  type EvalReport,
  type BeliefsChartPoint,
  type QualiaChartPoint,
  type GrowthChartPoint,
} from './dashboard-data'

export function useConsciousnessDashboard() {
  const addToast = useToastStore((state) => state.addToast)
  const { isLive, lastUpdate, toggleLive, latestEvent } = useConsciousnessLive()
  const [loading, setLoading] = useState(true)
  const [episodes, setEpisodes] = useState<Episode[]>([])
  const [qualiaHistory, setQualiaHistory] = useState<QualiaPoint[]>([])
  const [beliefsHistory, setBeliefsHistory] = useState<BeliefsPoint[]>([])
  const [evalReport, setEvalReport] = useState<EvalReport | null>(null)
  const [totalEpisodes, setTotalEpisodes] = useState(0)
  const [currentBeliefs, setCurrentBeliefs] = useState<Record<string, number>>({})
  const [currentQualia, setCurrentQualia] = useState<Record<string, number>>({})
  const [seeding, setSeeding] = useState(false)
  const [reflecting, setReflecting] = useState(false)
  const [ratingEpisode, setRatingEpisode] = useState<number | null>(null)
  const [selectedEpisode, setSelectedEpisode] = useState<Episode | null>(null)
  const [consciousnessLevel, setConsciousnessLevel] = useState(1)
  const [exporting, setExporting] = useState(false)
  const [importing, setImporting] = useState(false)

  const fetchAll = useCallback(async () => {
    try {
      const [epResult, qResult, bResult, eResult, sResult, qNowResult, statusResult] =
        await Promise.allSettled([
          consciousnessController.getEpisodeHistory(50),
          consciousnessController.getQualiaHistory(100),
          consciousnessController.getBeliefsHistory(),
          consciousnessController.evaluate(),
          consciousnessController.getSelfModel(),
          consciousnessController.getQualia(),
          consciousnessController.getStatus(),
        ])

      if (epResult.status === 'fulfilled') {
        setEpisodes((epResult.value.episodes ?? []) as unknown as Episode[])
        setTotalEpisodes(epResult.value.total ?? 0)
      }
      if (qResult.status === 'fulfilled') {
        setQualiaHistory((qResult.value.history ?? []) as unknown as QualiaPoint[])
      }
      if (bResult.status === 'fulfilled') {
        setBeliefsHistory((bResult.value.beliefs ?? []) as unknown as BeliefsPoint[])
        const last = bResult.value.beliefs?.[bResult.value.beliefs.length - 1]
        if (last) {
          const { timestamp, step, ...rest } = last
          setCurrentBeliefs(rest)
        }
      }
      if (eResult.status === 'fulfilled') {
        setEvalReport(eResult.value ?? null)
      }
      if (sResult.status === 'fulfilled') {
        // Wire field is `beliefs` (routers/consciousness.py get_self_model
        // returns `beliefs: dict(sm.self_beliefs)`); `self_beliefs` never
        // existed on the response, so this branch silently never ran.
        if (sResult.value.beliefs) setCurrentBeliefs(sResult.value.beliefs)
      }
      if (qNowResult.status === 'fulfilled') {
        // The endpoint wraps the flat map: {current: {valence…}, narrative,
        // history_count}. Storing the whole envelope here crashed
        // CurrentStateFold (v.toFixed on an object) and blanked the page.
        if (qNowResult.value) setCurrentQualia(qNowResult.value.current ?? {})
      }
      if (statusResult.status === 'fulfilled') {
        if (statusResult.value.level !== undefined) setConsciousnessLevel(statusResult.value.level)
      }
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    fetchAll()
  }, [fetchAll])

  useEffect(() => {
    if (!isLive || !latestEvent) return
    if (latestEvent.qualia) setCurrentQualia(latestEvent.qualia)
    if (latestEvent.beliefs) {
      setCurrentBeliefs(latestEvent.beliefs)
      setBeliefsHistory((prev) => {
        const point: BeliefsPoint = {
          timestamp: Date.now() / 1000,
          step: (prev[prev.length - 1]?.step ?? 0) + 1,
          competence: (latestEvent.beliefs as any).competence ?? 0,
          helpfulness: (latestEvent.beliefs as any).helpfulness ?? 0,
          creativity: (latestEvent.beliefs as any).creativity ?? 0,
          accuracy: (latestEvent.beliefs as any).accuracy ?? 0,
          empathy: (latestEvent.beliefs as any).empathy ?? 0,
        }
        return [...prev, point]
      })
    }
    if (latestEvent.qualia) {
      setQualiaHistory((prev) => {
        const point: QualiaPoint = {
          timestamp: Date.now() / 1000,
          valence: latestEvent.qualia.valence ?? 0,
          arousal: latestEvent.qualia.arousal ?? 0,
          novelty: latestEvent.qualia.novelty ?? 0,
          coherence: latestEvent.qualia.coherence ?? 0,
        }
        return [...prev, point]
      })
    }
    if (latestEvent.level !== undefined) setConsciousnessLevel(latestEvent.level)
    addToast('Consciousness state updated', 'success')
  }, [latestEvent, isLive, addToast])

  const handleSeed = async () => {
    setSeeding(true)
    try {
      await consciousnessController.seedData({ count: 30 })
      addToast('30 episodes added', 'success')
      await fetchAll()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setSeeding(false)
    }
  }

  const handleReflect = async () => {
    setReflecting(true)
    try {
      await consciousnessController.reflect()
      addToast('New episode recorded', 'success')
      await fetchAll()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setReflecting(false)
    }
  }

  const handleFeedback = async (episodeIndex: number, rating: number) => {
    setRatingEpisode(episodeIndex)
    try {
      await consciousnessController.submitFeedback({ episode_index: episodeIndex, rating })
      addToast(`Rated ${rating}/5`, 'success')
      await fetchAll()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setRatingEpisode(null)
    }
  }

  const handleLevelChange = async (newLevel: number) => {
    try {
      await consciousnessController.updateConfig({ level: newLevel })
      setConsciousnessLevel(newLevel)
      addToast(`Level set to ${newLevel}`, 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }

  const handleExport = async () => {
    setExporting(true)
    try {
      const data = {
        episodes,
        qualiaHistory,
        beliefsHistory,
        currentBeliefs,
        currentQualia,
        evalReport,
        exportedAt: new Date().toISOString(),
      }
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `consciousness-export-${Date.now()}.json`
      a.click()
      URL.revokeObjectURL(url)
      addToast('Exported consciousness data', 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setExporting(false)
    }
  }

  const handleImport = async () => {
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
        addToast(`Imported ${data.episodes?.length ?? 0} episodes`, 'success')
        await fetchAll()
      } catch (err) {
        addToast('Invalid JSON file', 'error')
      } finally {
        setImporting(false)
      }
    }
    input.click()
  }

  // Chart data prepared with time labels (was inline in the page component)
  const qualiaData: QualiaChartPoint[] = qualiaHistory.map((q) => ({
    ...q,
    time: formatTime(q.timestamp),
  }))

  const beliefsData: BeliefsChartPoint[] = beliefsHistory.map((b) => ({
    ...b,
    time: formatTime(b.timestamp),
  }))

  const growthData: GrowthChartPoint[] = episodes.map((ep) => ({
    time: formatTime(ep.timestamp),
    growth: ep.growth_delta,
    cumulative: episodes
      .slice(0, episodes.indexOf(ep) + 1)
      .reduce((sum, e) => sum + e.growth_delta, 0),
  }))

  const avgGrowth =
    episodes.length > 0
      ? `${((episodes.reduce((s, e) => s + e.growth_delta, 0) / episodes.length) * 100).toFixed(1)}%`
      : '—'

  const score = evalReport ? evalReport.overall_score : null

  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- wire shape predates this card's types (see qualia-crash card for the fix)
  const lastReflection: string | null =
    (evalReport as any)?.last_reflection || episodes[episodes.length - 1]?.self_insight || null

  return {
    loading,
    episodes,
    totalEpisodes,
    currentBeliefs,
    currentQualia,
    evalReport,
    qualiaData,
    beliefsData,
    growthData,
    avgGrowth,
    score,
    lastReflection,
    seeding,
    reflecting,
    ratingEpisode,
    selectedEpisode,
    consciousnessLevel,
    exporting,
    importing,
    isLive,
    lastUpdate,
    toggleLive,
    setSelectedEpisode,
    handleSeed,
    handleReflect,
    handleFeedback,
    handleLevelChange,
    handleExport,
    handleImport,
  }
}
