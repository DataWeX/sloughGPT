'use client'

import { useState, useEffect, useCallback } from 'react'
import { settingsController } from '@/lib/settings-controller'
import type { TrainingRun } from '@/components/training/training-shared'

export interface UseTrainingRunsReturn {
  runs: TrainingRun[]
  loading: boolean
  filterModel: string
  filterMethod: string
  setFilterModel: (v: string) => void
  setFilterMethod: (v: string) => void
  fetchRuns: () => Promise<void>
  deleteRun: (runId: string) => Promise<void>
  duplicateRun: (runId: string) => Promise<void>
  toggleBookmark: (runId: string) => Promise<void>
  addTag: (runId: string, tag: string) => Promise<TrainingRun | null>
  removeTag: (runId: string, tag: string) => Promise<TrainingRun | null>
  saveNotes: (runId: string, notes: string) => Promise<TrainingRun | null>
  bulkDelete: (runIds: string[]) => Promise<void>
  bulkTag: (runIds: string[], tag: string) => Promise<void>
  bulkBookmark: (runIds: string[], bookmarked: boolean) => Promise<void>
}

export function useTrainingRuns(): UseTrainingRunsReturn {
  const [runs, setRuns] = useState<TrainingRun[]>([])
  const [loading, setLoading] = useState(true)
  const [filterModel, setFilterModel] = useState('')
  const [filterMethod, setFilterMethod] = useState('')

  const fetchRuns = useCallback(async () => {
    setLoading(true)
    try {
      const params: Record<string, string | number> = { limit: 200 }
      if (filterModel) params.model = filterModel
      if (filterMethod) params.method = filterMethod
      const resp = await settingsController.filterTrainingRuns(params)
      setRuns((resp.runs as unknown as TrainingRun[]) || [])
    } catch (err) {
      console.error('Failed to fetch training runs:', err)
    } finally {
      setLoading(false)
    }
  }, [filterModel, filterMethod])

  useEffect(() => { void fetchRuns() }, [fetchRuns])

  const applyUpdated = useCallback((updated: unknown) => {
    const run = updated as unknown as TrainingRun
    setRuns(prev => prev.map(r => (r.run_id === run.run_id ? run : r)))
    return run
  }, [])

  const deleteRun = useCallback(async (runId: string) => {
    await settingsController.deleteTrainingRun(runId)
    setRuns(prev => prev.filter(r => r.run_id !== runId))
  }, [])

  const duplicateRun = useCallback(async (runId: string) => {
    const newRun = await settingsController.duplicateTrainingRun(runId)
    setRuns(prev => [(newRun as unknown as TrainingRun), ...prev])
  }, [])

  const toggleBookmark = useCallback(async (runId: string) => {
    const updated = await settingsController.toggleBookmark(runId)
    applyUpdated(updated)
  }, [applyUpdated])

  const addTag = useCallback(async (runId: string, tag: string) => {
    const updated = await settingsController.addRunTag(runId, tag)
    return applyUpdated(updated)
  }, [applyUpdated])

  const removeTag = useCallback(async (runId: string, tag: string) => {
    const updated = await settingsController.removeRunTag(runId, tag)
    return applyUpdated(updated)
  }, [applyUpdated])

  const saveNotes = useCallback(async (runId: string, notes: string) => {
    const updated = await settingsController.setRunNotes(runId, notes)
    return applyUpdated(updated)
  }, [applyUpdated])

  const bulkDelete = useCallback(async (runIds: string[]) => {
    await settingsController.bulkDeleteRuns(runIds)
    setRuns(prev => prev.filter(r => !runIds.includes(r.run_id)))
  }, [])

  const bulkTag = useCallback(async (runIds: string[], tag: string) => {
    await settingsController.bulkAddTag(runIds, tag)
    await fetchRuns()
  }, [fetchRuns])

  const bulkBookmark = useCallback(async (runIds: string[], bookmarked: boolean) => {
    await settingsController.bulkBookmark(runIds, bookmarked)
    await fetchRuns()
  }, [fetchRuns])

  return {
    runs, loading, filterModel, filterMethod, setFilterModel, setFilterMethod,
    fetchRuns, deleteRun, duplicateRun, toggleBookmark,
    addTag, removeTag, saveNotes, bulkDelete, bulkTag, bulkBookmark,
  }
}
