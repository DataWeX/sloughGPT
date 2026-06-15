'use client'

import { useEffect, useState, useCallback } from 'react'
import { modelController } from '@/lib/model-controller'
import { soulsController } from '@/lib/souls-controller'
import { sessionController } from '@/lib/session-controller'
import { trainingController } from '@/lib/training-controller'
import { knowledgeController } from '@/lib/knowledge-controller'
import { PUBLIC_API_URL } from '@/lib/config'
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

  const inferenceCount = health && health !== 'offline' ? (health as any).inference_count ?? 0 : null
  const apiStatus = health === null ? 'loading' : health === 'offline' ? 'offline' : 'online'

  useEffect(() => {
    if (apiStatus !== 'online') return
    modelController.status().then(status => {
      setModelStatus({ loaded: status.loaded, model: status.model_type })
    }).catch(() => {})
    soulsController.list().then(data => {
      const active = data.souls?.find((s: any) => s.name === data.current_soul)
      if (active) setCurrentSoul(active)
    }).catch(() => {})
  }, [apiStatus])

  useEffect(() => {
    if (health === null || health === 'offline') { setModelCount(null); return }
    let cancelled = false
    modelController.list().then(models => { if (!cancelled) setModelCount(models.length) }).catch(() => {})
    sessionController.list().then(sessions => {
      if (!cancelled) {
        const sorted = [...sessions]
          .filter(s => s.name)
          .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
          .slice(0, 5)
        setRecentSessions(sorted)
      }
    }).catch(() => {})
    trainingController.list().then(jobs => {
      if (!cancelled) {
        const running = jobs.find((j: any) => j.status === 'running')
        setRunningTraining(running ? { name: running.name || running.id, status_message: running.status_message || 'Training...' } : null)
        const recent = [...jobs]
          .sort((a: any, b: any) => new Date(b.created_at || 0).getTime() - new Date(a.created_at || 0).getTime())
          .slice(0, 3)
        setRecentJobs(recent)
      }
    }).catch(() => {})
    knowledgeController.stats().then(s => {
      if (!cancelled) setKnowledgeCount(s.total_items)
    }).catch(() => {})
    return () => { cancelled = true }
  }, [health])

  return {
    modelCount, checkpointCount,
    modelStatus, currentSoul,
    recentSessions, runningTraining,
    knowledgeCount, recentJobs,
    testRunning, testResponse,
    setTestRunning, setTestResponse,
    setKnowledgeCount, inferenceCount,
  }
}
