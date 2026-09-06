'use client'

import { useState, useCallback } from 'react'
import { trainingJobsController } from '@/lib/controllers'
import type { Checkpoint } from '@/lib/souls-controller'
import type { TrainingBuild, TrainingJob } from '@/lib/training-controller'
import { logger, trackEvent } from '@/lib/dev-log'

export interface UseTrainingCheckpointsReturn {
  checkpoints: Checkpoint[]
  loadingCheckpoints: boolean
  activeCheckpoint: string | null
  builds: TrainingBuild[]
  loadingBuilds: boolean
  jobs: TrainingJob[]
  loadingJobs: boolean
  setActiveCheckpoint: (c: string | null) => void
  setCheckpoints: (c: Checkpoint[]) => void
  fetchCheckpoints: () => Promise<void>
  fetchBuilds: () => Promise<void>
  fetchJobs: () => Promise<void>
  handleLoadCheckpoint: (name: string, addToast: (msg: string, type?: 'success' | 'error' | 'info') => void) => Promise<void>
  handleDeleteCheckpoint: (name: string, addToast: (msg: string, type?: 'success' | 'error' | 'info') => void) => Promise<void>
}

export function useTrainingCheckpoints(): UseTrainingCheckpointsReturn {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [loadingCheckpoints, setLoadingCheckpoints] = useState(true)
  const [activeCheckpoint, setActiveCheckpoint] = useState<string | null>(null)
  const [builds, setBuilds] = useState<TrainingBuild[]>([])
  const [loadingBuilds, setLoadingBuilds] = useState(true)
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [loadingJobs, setLoadingJobs] = useState(true)

  const fetchCheckpoints = useCallback(async () => {
    setLoadingCheckpoints(true)
    try {
      const data = await trainingJobsController.listCheckpoints()
      setCheckpoints(data)
    } catch (e) { logger.warning('Could not checkpoints fetch', { exception: String(e instanceof Error ? e.message : e) }) }
    finally { setLoadingCheckpoints(false) }
  }, [])

  const fetchBuilds = useCallback(async () => {
    try {
      const data = await trainingJobsController.listBuilds()
      setBuilds(data ?? [])
    } catch (e) { logger.warning('Could not checkpoints builds fetch', { exception: String(e instanceof Error ? e.message : e) }) }
    finally { setLoadingBuilds(false) }
  }, [])

  const fetchJobs = useCallback(async () => {
    try { setJobs((await trainingJobsController.list()) ?? []) }
    catch (e) { logger.warning('Could not checkpoints jobs fetch', { exception: String(e instanceof Error ? e.message : e) }) }
    finally { setLoadingJobs(false) }
  }, [])

  const handleLoadCheckpoint = useCallback(async (name: string, addToast: (msg: string, type?: 'success' | 'error' | 'info') => void) => {
    try {
      await trainingJobsController.loadCheckpoint?.(name)
      setActiveCheckpoint(name)
      trackEvent('checkpoint_loaded', { name })
      addToast(`Loaded trained version: ${name}`, 'success')
    } catch { addToast('Could not load trained version', 'error') }
  }, [])

  const handleDeleteCheckpoint = useCallback(async (name: string, addToast: (msg: string, type?: 'success' | 'error' | 'info') => void) => {
    if (!confirm(`Delete trained version "${name}"?`)) return
    try {
      await trainingJobsController.deleteCheckpoint?.(name)
      setCheckpoints(prev => prev.filter(c => c.name !== name))
      setActiveCheckpoint(prev => prev === name ? null : prev)
      trackEvent('checkpoint_deleted', { name })
      addToast(`Deleted ${name}`, 'success')
    } catch { addToast('Could not delete trained version', 'error') }
  }, [])

  return {
    checkpoints, loadingCheckpoints, activeCheckpoint, builds, loadingBuilds, jobs, loadingJobs,
    setActiveCheckpoint, setCheckpoints,
    fetchCheckpoints, fetchBuilds, fetchJobs,
    handleLoadCheckpoint, handleDeleteCheckpoint,
  }
}
