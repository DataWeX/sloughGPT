'use client'

import { useState, useCallback } from 'react'
import { trainingJobsController, modelController } from '@/lib/controllers'
import type { Checkpoint } from '@/lib/souls-controller'
import type { TrainingBuild } from '@/lib/training-controller'

export interface UseTrainingCheckpointsReturn {
  checkpoints: Checkpoint[]
  activeCheckpoint: string | null
  builds: TrainingBuild[]
  loadingBuilds: boolean
  jobs: any[]
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
  const [activeCheckpoint, setActiveCheckpoint] = useState<string | null>(null)
  const [builds, setBuilds] = useState<TrainingBuild[]>([])
  const [loadingBuilds, setLoadingBuilds] = useState(true)
  const [jobs, setJobs] = useState<any[]>([])
  const [loadingJobs, setLoadingJobs] = useState(true)

  const fetchCheckpoints = useCallback(async () => {
    try {
      const data = await trainingJobsController.listCheckpoints()
      setCheckpoints(data)
    } catch { /* ignore */ }
  }, [])

  const fetchBuilds = useCallback(async () => {
    try {
      const data = await trainingJobsController.listBuilds()
      setBuilds(data)
    } catch { /* ignore */ }
    finally { setLoadingBuilds(false) }
  }, [])

  const fetchJobs = useCallback(async () => {
    try { setJobs(await trainingJobsController.list()) }
    catch { /* ignore */ }
    finally { setLoadingJobs(false) }
  }, [])

  const handleLoadCheckpoint = useCallback(async (name: string, addToast: (msg: string, type?: 'success' | 'error' | 'info') => void) => {
    try {
      await trainingJobsController.loadCheckpoint?.(name)
      setActiveCheckpoint(name)
      addToast(`Loaded trained version: ${name}`, 'success')
    } catch { addToast('Failed to load trained version', 'error') }
  }, [])

  const handleDeleteCheckpoint = useCallback(async (name: string, addToast: (msg: string, type?: 'success' | 'error' | 'info') => void) => {
    if (!confirm(`Delete trained version "${name}"?`)) return
    try {
      await trainingJobsController.deleteCheckpoint?.(name)
      setCheckpoints(prev => prev.filter(c => c.name !== name))
      setActiveCheckpoint(prev => prev === name ? null : prev)
      addToast(`Deleted ${name}`, 'success')
    } catch { addToast('Failed to delete trained version', 'error') }
  }, [])

  return {
    checkpoints, activeCheckpoint, builds, loadingBuilds, jobs, loadingJobs,
    setActiveCheckpoint, setCheckpoints,
    fetchCheckpoints, fetchBuilds, fetchJobs,
    handleLoadCheckpoint, handleDeleteCheckpoint,
  }
}
