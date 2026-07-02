'use client'

import { useEffect, useState } from 'react'
import { modelController } from '@/lib/model-controller'
import type { HealthStatus } from '@/lib/model-controller'
import { soulsController } from '@/lib/souls-controller'
import { sessionController } from '@/lib/session-controller'
import { trainingController } from '@/lib/training-controller'
import { knowledgeController } from '@/lib/knowledge-controller'
import { feedbackController, type FeedbackStats } from '@/lib/feedback-controller'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'

export interface HomePageData {
  modelCount: number | null
  checkpointCount: number
  modelStatus: { loaded: boolean; model: string | null }
  currentSoul: { name: string; description: string; traits: string[] } | null
  recentSessions: Array<{ id: string; name: string; updated_at: string }>
  runningTraining: { name: string; status_message: string } | null
  knowledgeCount: number
  recentJobs: Array<{ id: string; name: string; status: string; created_at?: string }>
  testRunning: boolean
  testResponse: string | null
  setTestRunning: (v: boolean) => void
  setTestResponse: (v: string | null) => void
  setKnowledgeCount: React.Dispatch<React.SetStateAction<number>>
  inferenceCount: number | null
  healthSummary: string | null
  feedbackStats: FeedbackStats | null
}

export function useHomePageData(health: ApiHealthSnapshot): HomePageData {
  const [modelCount, setModelCount] = useState<number | null>(null)
  const [checkpointCount, setCheckpointCount] = useState<number>(0)
  const [modelStatus, setModelStatus] = useState<{ loaded: boolean; model: string | null }>({ loaded: false, model: null })
  const [currentSoul, setCurrentSoul] = useState<{ name: string; description: string; traits: string[] } | null>(null)
  const [recentSessions, setRecentSessions] = useState<Array<{ id: string; name: string; updated_at: string }>>([])
  const [runningTraining, setRunningTraining] = useState<{ name: string; status_message: string } | null>(null)
  const [knowledgeCount, setKnowledgeCount] = useState<number>(0)
  const [recentJobs, setRecentJobs] = useState<Array<{ id: string; name: string; status: string; created_at?: string }>>([])
  const [testRunning, setTestRunning] = useState(false)
  const [testResponse, setTestResponse] = useState<string | null>(null)
  const [feedbackStats, setFeedbackStats] = useState<FeedbackStats | null>(null)

  const inferenceCount = health && health !== 'offline' ? (health as HealthStatus).inference_count ?? 0 : null
  const healthSummary = health && health !== 'offline' ? (health as HealthStatus).model_type ?? null : null
  const apiStatus = health === null ? 'loading' : health === 'offline' ? 'offline' : 'online'

  useEffect(() => {
    const cancelled = { current: false }
    modelController.status().then(status => {
      if (!cancelled.current) setModelStatus({ loaded: status.loaded, model: status.model_type })
    }).catch(() => {})
    soulsController.list().then(data => {
      if (!cancelled.current) {
        const active = data.current_soul
          ? data.souls?.find((s: any) => s.name === data.current_soul)
          : null
        setCurrentSoul(active || null)
      }
    }).catch(() => {})
    modelController.list().then(models => {
      if (!cancelled.current) setModelCount(models.length)
    }).catch(() => {})
    sessionController.list().then(sessions => {
      if (!cancelled.current) {
        const sorted = [...sessions]
          .filter(s => s.name || s.id)
          .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
          .slice(0, 5)
        setRecentSessions(sorted as any)
      }
    }).catch(() => {})
    trainingController.list().then(jobs => {
      if (!cancelled.current) {
        const running = jobs.find((j: any) => j.status === 'running')
        setRunningTraining(running ? { name: running.name || running.id, status_message: running.status_message || 'Training...' } : null)
        const recent = [...jobs]
          .sort((a: any, b: any) => new Date(b.created_at || 0).getTime() - new Date(a.created_at || 0).getTime())
          .slice(0, 3)
        setRecentJobs(recent)
      }
    }).catch(() => {})
    knowledgeController.stats().then(s => {
      if (!cancelled.current) setKnowledgeCount(s.total_items)
    }).catch(() => {})
    feedbackController.getFeedbackStats().then(s => {
      if (!cancelled.current) setFeedbackStats(s)
    }).catch(() => {})
    return () => { cancelled.current = true }
  }, [])

  return {
    modelCount, checkpointCount,
    modelStatus, currentSoul,
    recentSessions, runningTraining,
    knowledgeCount, recentJobs,
    testRunning, testResponse,
    setTestRunning, setTestResponse,
    setKnowledgeCount, inferenceCount,
    healthSummary, feedbackStats,
  }
}
