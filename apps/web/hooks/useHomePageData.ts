'use client'

import { useEffect, useState } from 'react'
import { modelController } from '@/lib/model-controller'
import type { HealthStatus } from '@/lib/model-controller'
import { soulsController, type Soul } from '@/lib/souls-controller'
import { sessionController } from '@/lib/session-controller'
import { trainingController } from '@/lib/training-controller'
import { knowledgeController } from '@/lib/knowledge-controller'
import { feedbackController, type FeedbackStats } from '@/lib/feedback-controller'
import { datasetController } from '@/lib/dataset-controller'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'
import { useApiReady } from '@/hooks/useLiveStatus'

export interface HomePageData {
  modelCount: number | null
  checkpointCount: number
  modelStatus: { loaded: boolean; model: string | null }
  currentSoul: { name: string; description: string; traits: string[] } | null
  recentSessions: Array<{ id: string; name: string; updated_at: string; message_count?: number; pinned?: boolean; starred?: boolean }>
  runningTraining: { name: string; status_message: string } | null
  knowledgeCount: number
  recentJobs: Array<{ id: string; name: string; status: string; created_at?: string }>
  recentDatasets: Array<{ id: string; name: string; updated_at?: string; size?: number; samples?: number }>
  testRunning: boolean
  testResponse: string | null
  setTestRunning: (v: boolean) => void
  setTestResponse: (v: string | null) => void
  setKnowledgeCount: React.Dispatch<React.SetStateAction<number>>
  inferenceCount: number | null
  healthSummary: string | null
  feedbackStats: FeedbackStats | null
  /** Per-section error flags — true if that section's fetch failed */
  errors: {
    models: boolean
    soul: boolean
    sessions: boolean
    training: boolean
    knowledge: boolean
    feedback: boolean
    datasets: boolean
  }
}

export function useHomePageData(health: ApiHealthSnapshot): HomePageData {
  const [modelCount, setModelCount] = useState<number | null>(null)
  const [checkpointCount, setCheckpointCount] = useState<number>(0)
  const [modelStatus, setModelStatus] = useState<{ loaded: boolean; model: string | null }>({ loaded: false, model: null })
  const [currentSoul, setCurrentSoul] = useState<{ name: string; description: string; traits: string[] } | null>(null)
  const [recentSessions, setRecentSessions] = useState<Array<{ id: string; name: string; updated_at: string; message_count?: number; pinned?: boolean; starred?: boolean }>>([])
  const [runningTraining, setRunningTraining] = useState<{ name: string; status_message: string } | null>(null)
  const [knowledgeCount, setKnowledgeCount] = useState<number>(0)
  const [recentJobs, setRecentJobs] = useState<Array<{ id: string; name: string; status: string; created_at?: string }>>([])
  const [recentDatasets, setRecentDatasets] = useState<Array<{ id: string; name: string; updated_at?: string; size?: number; samples?: number }>>([])
  const [testRunning, setTestRunning] = useState(false)
  const [testResponse, setTestResponse] = useState<string | null>(null)
  const [feedbackStats, setFeedbackStats] = useState<FeedbackStats | null>(null)
  const [errors, setErrors] = useState({
    models: false,
    soul: false,
    sessions: false,
    training: false,
    knowledge: false,
    feedback: false,
    datasets: false,
  })

  const ready = useApiReady()

  const inferenceCount = health && health !== 'offline' ? health.inference_count ?? 0 : null
  const healthSummary = health && health !== 'offline' ? health.model_type ?? null : null
  const apiStatus = health === null ? 'loading' : health === 'offline' ? 'offline' : 'online'

  useEffect(() => {
    if (!ready) return
    const cancelled = { current: false }
    modelController.status().then(status => {
      if (!cancelled.current) setModelStatus({ loaded: status.loaded, model: status.model_type })
    }).catch(e => { console.warn('[home] model status failed:', e?.message || e); if (!cancelled.current) setErrors(p => ({ ...p, models: true })) })
    soulsController.list().then(data => {
      if (!cancelled.current) {
        const active = data.current_soul
          ? data.souls?.find((s: Soul) => s.name === data.current_soul)
          : null
        setCurrentSoul(active || null)
      }
    }).catch(e => { console.warn('[home] souls list failed:', e?.message || e); if (!cancelled.current) setErrors(p => ({ ...p, soul: true })) })
    modelController.list().then(models => {
      if (!cancelled.current) setModelCount(models.length)
    }).catch(e => { console.warn('[home] model list failed:', e?.message || e); if (!cancelled.current) setErrors(p => ({ ...p, models: true })) })
    sessionController.list().then(sessions => {
      if (!cancelled.current) {
        const sorted = [...sessions]
          .filter(s => s.name || s.id)
          .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
          .slice(0, 5)
          .map(s => ({
            id: s.id,
            name: s.name,
            updated_at: s.updated_at,
            message_count: s.messages?.length,
            pinned: s.pinned,
            starred: s.starred,
          }))
        setRecentSessions(sorted)
      }
    }).catch(e => { console.warn('[home] sessions list failed:', e?.message || e); if (!cancelled.current) setErrors(p => ({ ...p, sessions: true })) })
    trainingController.list().then(jobs => {
      if (!cancelled.current) {
        const running = jobs.find(j => j.status === 'running')
        setRunningTraining(running ? { name: running.name || running.id, status_message: running.status_message || 'Training...' } : null)
        const recent = [...jobs]
          .sort((a, b) => new Date(b.created_at || 0).getTime() - new Date(a.created_at || 0).getTime())
          .slice(0, 3)
        setRecentJobs(recent)
      }
    }).catch(e => { console.warn('[home] training list failed:', e?.message || e); if (!cancelled.current) setErrors(p => ({ ...p, training: true })) })
    knowledgeController.stats().then(s => {
      if (!cancelled.current) setKnowledgeCount(s.total_items)
    }).catch(e => { console.warn('[home] knowledge stats failed:', e?.message || e); if (!cancelled.current) setErrors(p => ({ ...p, knowledge: true })) })
    feedbackController.getFeedbackStats().then(s => {
      if (!cancelled.current) setFeedbackStats(s)
    }).catch(e => { console.warn('[home] feedback stats failed:', e?.message || e); if (!cancelled.current) setErrors(p => ({ ...p, feedback: true })) })
    datasetController.list().then(list => {
      if (!cancelled.current) {
        const sorted = [...list]
          .sort((a, b) => new Date(b.updated_at || b.created_at || 0).getTime() - new Date(a.updated_at || a.created_at || 0).getTime())
          .slice(0, 5)
          .map(ds => ({
            id: ds.id,
            name: ds.name,
            updated_at: ds.updated_at || ds.created_at,
            size: ds.size,
            samples: ds.samples,
          }))
        setRecentDatasets(sorted)
      }
    }).catch(e => { console.warn('[home] datasets list failed:', e?.message || e); if (!cancelled.current) setErrors(p => ({ ...p, datasets: true })) })
    return () => { cancelled.current = true }
  }, [ready])

  return {
    modelCount, checkpointCount,
    modelStatus, currentSoul,
    recentSessions, runningTraining,
    knowledgeCount, recentJobs, recentDatasets,
    testRunning, testResponse,
    setTestRunning, setTestResponse,
    setKnowledgeCount, inferenceCount,
    healthSummary, feedbackStats,
    errors,
  }
}
