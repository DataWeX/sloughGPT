'use client'

// Home-page data hook — derives every landing-section value from the
// /bootstrap payload (one request per app load) instead of the historical
// per-section fan-out (sessions + training + knowledge + feedback +
// datasets + models + souls). Model status / inference count still come
// from the SSE health snapshot — streams stay streams by contract.

import { useEffect, useState } from 'react'
import type { Soul } from '@/lib/souls-controller'
import { buildFeedbackStats, type FeedbackStats } from '@/lib/feedback-controller'
import type { ApiHealthSnapshot } from '@/hooks/useApiHealth'
import { useBootstrap } from '@/contexts/BootstrapContext'

export interface HomePageData {
  modelCount: number | null
  checkpointCount: number
  modelStatus: { loaded: boolean; model: string | null }
  currentSoul: { name: string; description: string; traits: string[] } | null
  recentSessions: Array<{
    id: string
    name: string
    updated_at: string
    message_count?: number
    pinned?: boolean
    starred?: boolean
  }>
  runningTraining: { name: string; status_message: string } | null
  knowledgeCount: number
  recentJobs: Array<{ id: string; name: string; status: string; created_at?: string }>
  recentDatasets: Array<{
    id: string
    name: string
    updated_at?: string
    size?: number
    samples?: number
  }>
  testRunning: boolean
  testResponse: string | null
  setTestRunning: (v: boolean) => void
  setTestResponse: (v: string | null) => void
  setKnowledgeCount: React.Dispatch<React.SetStateAction<number>>
  inferenceCount: number | null
  healthSummary: string | null
  feedbackStats: FeedbackStats | null
  loading: boolean
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
  const [checkpointCount] = useState<number>(0)
  const [modelStatus, setModelStatus] = useState<{ loaded: boolean; model: string | null }>({
    loaded: false,
    model: null,
  })
  const [currentSoul, setCurrentSoul] = useState<{
    name: string
    description: string
    traits: string[]
  } | null>(null)
  const [recentSessions, setRecentSessions] = useState<
    Array<{
      id: string
      name: string
      updated_at: string
      message_count?: number
      pinned?: boolean
      starred?: boolean
    }>
  >([])
  const [runningTraining, setRunningTraining] = useState<{
    name: string
    status_message: string
  } | null>(null)
  const [knowledgeCount, setKnowledgeCount] = useState<number>(0)
  const [recentJobs, setRecentJobs] = useState<
    Array<{ id: string; name: string; status: string; created_at?: string }>
  >([])
  const [recentDatasets, setRecentDatasets] = useState<
    Array<{
      id: string
      name: string
      updated_at?: string
      size?: number
      samples?: number
    }>
  >([])
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

  const { data: bootstrap, error: bootstrapError } = useBootstrap()

  // Derive model count from the bootstrap slice (was the shared models
  // query cache — same normalization, one less consumer of that call).
  useEffect(() => {
    const models = bootstrap?.members.models
    if (models) setModelCount(models.length)
  }, [bootstrap])

  // Current soul from the souls slice.
  useEffect(() => {
    const soulsData = bootstrap?.members.souls
    if (soulsData) {
      const active = soulsData.current_soul
        ? (soulsData.souls || [])?.find((s: Soul) => s.name === soulsData.current_soul)
        : null
      setCurrentSoul(active || null)
    }
  }, [bootstrap])

  // Use health snapshot for model status instead of a separate API call.
  useEffect(() => {
    if (health && health !== 'offline') {
      setModelStatus({ loaded: health.model_loaded, model: health.model_type })
    }
  }, [health])

  const inferenceCount = health && health !== 'offline' ? (health.inference_count ?? 0) : null
  const healthSummary = health && health !== 'offline' ? (health.model_type ?? null) : null

  // The remaining landing sections arrive as members of the same payload.
  // A null member with errors[key] === 'not_ready' is just the server still
  // booting (members appear in later payloads) — only a real error entry
  // raises the section's error flag.
  useEffect(() => {
    if (!bootstrap) return
    const { members, errors: memberErrors } = bootstrap
    const failed = (key: string) => {
      const err = memberErrors[key]
      return err !== undefined && err !== 'not_ready'
    }

    if (members.sessions) {
      const sorted = [...members.sessions]
        .filter((s) => s.name || s.id)
        .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
        .slice(0, 5)
        .map((s) => ({
          id: s.id,
          name: s.name,
          updated_at: s.updated_at,
          message_count: s.messages?.length,
          pinned: s.pinned,
          starred: s.starred,
        }))
      setRecentSessions(sorted)
    }

    if (members.training_jobs) {
      const jobs = members.training_jobs
      const running = jobs.find((j) => j.status === 'running')
      setRunningTraining(
        running
          ? {
              name: running.name || running.id,
              status_message:
                (running as { status_message?: string }).status_message || 'Training...',
            }
          : null,
      )
      const recent = [...jobs]
        .sort(
          (a, b) => new Date(b.created_at || 0).getTime() - new Date(a.created_at || 0).getTime(),
        )
        .slice(0, 3)
      setRecentJobs(recent)
    }

    if (members.knowledge_stats) setKnowledgeCount(members.knowledge_stats.total_items)

    if (members.feedback_stats) {
      setFeedbackStats(buildFeedbackStats(members.feedback_stats, members.meta_weights_stats ?? {}))
    }

    if (members.datasets) {
      const list = members.datasets.datasets
      const sorted = [...list]
        .sort(
          (a, b) =>
            new Date(b.updated_at || b.created_at || 0).getTime() -
            new Date(a.updated_at || a.created_at || 0).getTime(),
        )
        .slice(0, 5)
        .map((ds) => ({
          id: ds.id,
          name: ds.name,
          updated_at: ds.updated_at || ds.created_at,
          size: ds.size,
          samples: ds.samples,
        }))
      setRecentDatasets(sorted)
    }

    setErrors({
      models: failed('models'),
      soul: failed('souls'),
      sessions: failed('sessions'),
      training: failed('training_jobs'),
      knowledge: failed('knowledge_stats'),
      feedback: failed('feedback_stats'),
      datasets: failed('datasets'),
    })
  }, [bootstrap])

  // The whole payload failed (server unreachable at boot): every section
  // failed — same flags the old per-section catch blocks would raise.
  useEffect(() => {
    if (bootstrapError && !bootstrap) {
      setErrors({
        models: true,
        soul: true,
        sessions: true,
        training: true,
        knowledge: true,
        feedback: true,
        datasets: true,
      })
    }
  }, [bootstrapError, bootstrap])

  const loading = bootstrap === null || health === null

  return {
    modelCount,
    checkpointCount,
    modelStatus,
    currentSoul,
    recentSessions,
    runningTraining,
    knowledgeCount,
    recentJobs,
    recentDatasets,
    testRunning,
    testResponse,
    setTestRunning,
    setTestResponse,
    setKnowledgeCount,
    inferenceCount,
    healthSummary,
    feedbackStats,
    loading,
    errors,
  }
}
