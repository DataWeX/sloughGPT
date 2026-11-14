'use client'

/**
 * Stateful core of the Consciousness Dashboard.
 *
 * This hook owns IO and state only: it fetches the dashboard's seven endpoints,
 * applies live-stream deltas, and exposes the action handlers. Deriving chart
 * series and rendering stay in `dashboard-panels.tsx`, so the seam is
 * deliberately "hook returns raw data, panels shape and paint it" — that keeps
 * the chart-specific reshaping next to the charts it serves.
 *
 * Extracted from app/(app)/consciousness/dashboard/page.tsx (874 lines) to keep
 * that page a composition root.
 */
import { notify } from '@/lib/notify'
import { useState, useEffect, useCallback } from 'react'
import { consciousnessController } from '@/lib/consciousness-controller'
import { extractErrorMessage } from '@/lib/error-utils'
import { useConsciousnessLive } from '@/hooks/useConsciousnessLive'

export interface Episode {
  timestamp: number
  input_text: string
  self_insight: string
  growth_delta: number
  qualia: Record<string, number>
}

export interface QualiaPoint {
  timestamp: number
  valence: number
  arousal: number
  novelty: number
  coherence: number
}

export interface BeliefsPoint {
  timestamp: number
  step: number
  competence: number
  helpfulness: number
  creativity: number
  accuracy: number
  empathy: number
}

export interface EvalReport {
  overall_score: number
  diagnostics: string[]
  [key: string]: unknown
}

export function useConsciousnessDashboard() {
  const addToast = notify.toast
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
          // Keep only the belief metrics — timestamp/step are bookkeeping.
          const rest: Record<string, number> = {}
          for (const [key, value] of Object.entries(last)) {
            if (key !== 'timestamp' && key !== 'step') rest[key] = value
          }
          setCurrentBeliefs(rest)
        }
      }
      if (eResult.status === 'fulfilled') {
        setEvalReport(eResult.value ?? null)
      }
      if (sResult.status === 'fulfilled') {
        if (sResult.value.beliefs) setCurrentBeliefs(sResult.value.beliefs)
      }
      if (qNowResult.status === 'fulfilled') {
        if (qNowResult.value) setCurrentQualia(qNowResult.value.current)
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

  // Live stream: fold each event into current state and append to history so
  // the charts advance without a refetch.
  useEffect(() => {
    if (!isLive || !latestEvent) return
    if (latestEvent.qualia) setCurrentQualia(latestEvent.qualia)
    if (latestEvent.beliefs) {
      setCurrentBeliefs(latestEvent.beliefs)
      setBeliefsHistory((prev) => {
        const point: BeliefsPoint = {
          timestamp: Date.now() / 1000,
          step: (prev[prev.length - 1]?.step ?? 0) + 1,
          competence: latestEvent.beliefs.competence ?? 0,
          helpfulness: latestEvent.beliefs.helpfulness ?? 0,
          creativity: latestEvent.beliefs.creativity ?? 0,
          accuracy: latestEvent.beliefs.accuracy ?? 0,
          empathy: latestEvent.beliefs.empathy ?? 0,
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

  const handleImport = () => {
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
      } catch {
        addToast('Invalid JSON file', 'error')
      } finally {
        setImporting(false)
      }
    }
    input.click()
  }

  const closeSelectedEpisode = useCallback(() => setSelectedEpisode(null), [])

  return {
    loading,
    episodes,
    totalEpisodes,
    qualiaHistory,
    beliefsHistory,
    evalReport,
    currentBeliefs,
    currentQualia,
    consciousnessLevel,
    isLive,
    lastUpdate,
    toggleLive,
    seeding,
    reflecting,
    ratingEpisode,
    exporting,
    importing,
    selectedEpisode,
    setSelectedEpisode,
    closeSelectedEpisode,
    handleSeed,
    handleReflect,
    handleFeedback,
    handleLevelChange,
    handleExport,
    handleImport,
    fetchAll,
  }
}
