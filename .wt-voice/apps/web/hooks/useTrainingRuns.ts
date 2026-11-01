'use client'

import { useState, useEffect, useCallback } from 'react'
import { settingsController } from '@/lib/settings-controller'
import { downloadBlob, type TrainingRun } from '@/components/training/training-run'

export interface UseTrainingRunsReturn {
  runs: TrainingRun[]
  loading: boolean
  filteredRuns: TrainingRun[]
  models: string[]
  methods: string[]
  summary: { total: number; converged: number; avgQuality: number; avgLoss: number }
  searchQuery: string
  setSearchQuery: (v: string) => void
  filterModel: string
  setFilterModel: (v: string) => void
  filterMethod: string
  setFilterMethod: (v: string) => void
  format: 'json' | 'csv'
  setFormat: (v: 'json' | 'csv') => void
  handleExport: () => Promise<void>
  selectedIds: Set<string>
  toggleSelect: (runId: string) => void
  toggleSelectAll: () => void
  clearSelection: () => void
  bulkTag: string
  setBulkTag: (v: string) => void
  handleBulkTag: () => Promise<void>
  handleBulkBookmark: (bookmarked: boolean) => Promise<void>
  handleBulkDelete: () => Promise<void>
  deleting: string | null
  selectedRun: TrainingRun | null
  selectRun: (run: TrainingRun) => void
  closeSelection: () => void
  handleBookmark: (runId: string) => Promise<void>
  handleDuplicate: (runId: string) => Promise<void>
  handleDelete: (runId: string) => Promise<void>
  handleExportSingle: (runId: string) => Promise<void>
  newTag: string
  setNewTag: (v: string) => void
  handleAddTag: () => Promise<void>
  handleRemoveTag: (tag: string) => Promise<void>
  editingNotes: boolean
  startEditNotes: () => void
  setEditingNotes: (v: boolean) => void
  notesValue: string
  setNotesValue: (v: string) => void
  handleSaveNotes: () => Promise<void>
  compareRunId: string
  changeCompareTarget: (v: string) => void
  compareResult: Record<string, unknown> | null
  handleCompare: () => Promise<void>
}

export function useTrainingRuns(): UseTrainingRunsReturn {
  const [runs, setRuns] = useState<TrainingRun[]>([])
  const [loading, setLoading] = useState(true)
  const [format, setFormat] = useState<'json' | 'csv'>('json')
  const [searchQuery, setSearchQuery] = useState('')
  const [filterModel, setFilterModel] = useState('')
  const [filterMethod, setFilterMethod] = useState('')
  const [selectedRun, setSelectedRun] = useState<TrainingRun | null>(null)
  const [deleting, setDeleting] = useState<string | null>(null)
  const [newTag, setNewTag] = useState('')
  const [editingNotes, setEditingNotes] = useState(false)
  const [notesValue, setNotesValue] = useState('')
  const [compareRunId, setCompareRunId] = useState('')
  const [compareResult, setCompareResult] = useState<Record<string, unknown> | null>(null)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [bulkTag, setBulkTag] = useState('')

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

  useEffect(() => {
    fetchRuns()
  }, [fetchRuns])

  const handleExport = async () => {
    try {
      const resp = await settingsController.exportTrainingHistory(format, 500)
      const content = format === 'csv' ? resp.content : JSON.stringify(resp, null, 2)
      const blob = new Blob([content ?? ''], {
        type: format === 'csv' ? 'text/csv' : 'application/json',
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `training-history.${format}`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Export failed:', err)
    }
  }

  const handleDelete = async (runId: string) => {
    if (!confirm('Delete this training run?')) return
    setDeleting(runId)
    try {
      await settingsController.deleteTrainingRun(runId)
      setRuns((prev) => prev.filter((r) => r.run_id !== runId))
      if (selectedRun?.run_id === runId) setSelectedRun(null)
    } catch (err) {
      console.error('Delete failed:', err)
    } finally {
      setDeleting(null)
    }
  }

  const handleAddTag = async () => {
    if (!selectedRun || !newTag.trim()) return
    try {
      const updated = await settingsController.addRunTag(selectedRun.run_id, newTag.trim())
      setSelectedRun(updated as unknown as TrainingRun)
      setRuns((prev) =>
        prev.map((r) =>
          r.run_id === selectedRun.run_id ? (updated as unknown as TrainingRun) : r,
        ),
      )
      setNewTag('')
    } catch (err) {
      console.error('Add tag failed:', err)
    }
  }

  const handleRemoveTag = async (tag: string) => {
    if (!selectedRun) return
    try {
      const updated = await settingsController.removeRunTag(selectedRun.run_id, tag)
      setSelectedRun(updated as unknown as TrainingRun)
      setRuns((prev) =>
        prev.map((r) =>
          r.run_id === selectedRun.run_id ? (updated as unknown as TrainingRun) : r,
        ),
      )
    } catch (err) {
      console.error('Remove tag failed:', err)
    }
  }

  const handleSaveNotes = async () => {
    if (!selectedRun) return
    try {
      const updated = await settingsController.setRunNotes(selectedRun.run_id, notesValue)
      setSelectedRun(updated as unknown as TrainingRun)
      setRuns((prev) =>
        prev.map((r) =>
          r.run_id === selectedRun.run_id ? (updated as unknown as TrainingRun) : r,
        ),
      )
      setEditingNotes(false)
    } catch (err) {
      console.error('Save notes failed:', err)
    }
  }

  const handleExportSingle = async (runId: string) => {
    try {
      const resp = await settingsController.exportTrainingRun(runId, 'json')
      downloadBlob(resp.content, `training-run-${runId}.json`, 'application/json')
    } catch (err) {
      console.error('Export failed:', err)
    }
  }

  const handleCompare = async () => {
    if (!selectedRun || !compareRunId) return
    try {
      const result = await settingsController.compareTrainingRuns(selectedRun.run_id, compareRunId)
      setCompareResult(result)
    } catch (err) {
      console.error('Compare failed:', err)
    }
  }

  const handleBookmark = async (runId: string) => {
    try {
      const updated = await settingsController.toggleBookmark(runId)
      setSelectedRun((prev) =>
        prev?.run_id === runId ? (updated as unknown as TrainingRun) : prev,
      )
      setRuns((prev) =>
        prev.map((r) =>
          r.run_id === runId
            ? { ...r, bookmarked: (updated as unknown as TrainingRun).bookmarked }
            : r,
        ),
      )
    } catch (err) {
      console.error('Bookmark failed:', err)
    }
  }

  const handleDuplicate = async (runId: string) => {
    try {
      const newRun = await settingsController.duplicateTrainingRun(runId)
      setRuns((prev) => [newRun as unknown as TrainingRun, ...prev])
    } catch (err) {
      console.error('Duplicate failed:', err)
    }
  }

  // Bulk operations
  const toggleSelect = (runId: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(runId)) next.delete(runId)
      else next.add(runId)
      return next
    })
  }

  const toggleSelectAll = () => {
    if (selectedIds.size === filteredRuns.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(filteredRuns.map((r) => r.run_id)))
    }
  }

  const clearSelection = () => setSelectedIds(new Set())

  const handleBulkDelete = async () => {
    if (!confirm(`Delete ${selectedIds.size} training runs?`)) return
    try {
      await settingsController.bulkDeleteRuns(Array.from(selectedIds))
      setRuns((prev) => prev.filter((r) => !selectedIds.has(r.run_id)))
      setSelectedIds(new Set())
      if (selectedRun && selectedIds.has(selectedRun.run_id)) setSelectedRun(null)
    } catch (err) {
      console.error('Bulk delete failed:', err)
    }
  }

  const handleBulkTag = async () => {
    if (!bulkTag.trim() || selectedIds.size === 0) return
    try {
      await settingsController.bulkAddTag(Array.from(selectedIds), bulkTag.trim())
      setBulkTag('')
      fetchRuns()
    } catch (err) {
      console.error('Bulk tag failed:', err)
    }
  }

  const handleBulkBookmark = async (bookmarked: boolean) => {
    if (selectedIds.size === 0) return
    try {
      await settingsController.bulkBookmark(Array.from(selectedIds), bookmarked)
      fetchRuns()
    } catch (err) {
      console.error('Bulk bookmark failed:', err)
    }
  }

  // The compare dropdown both retargets and clears the previous result
  // (matching the original page's combined onChange behavior).
  const changeCompareTarget = (v: string) => {
    setCompareRunId(v)
    setCompareResult(null)
  }

  const selectRun = (run: TrainingRun) => {
    setSelectedRun(run)
    setEditingNotes(false)
    setNotesValue(run.notes || '')
  }

  const closeSelection = () => setSelectedRun(null)

  const startEditNotes = () => {
    setEditingNotes(true)
    setNotesValue(selectedRun?.notes || '')
  }

  const filteredRuns = runs.filter((run) => {
    if (!searchQuery) return true
    const q = searchQuery.toLowerCase()
    return (
      (run.model || '').toLowerCase().includes(q) ||
      (run.method || '').toLowerCase().includes(q) ||
      (run.dataset || '').toLowerCase().includes(q) ||
      run.run_id.toLowerCase().includes(q) ||
      (run.tags || []).some((t) => t.toLowerCase().includes(q)) ||
      (run.notes || '').toLowerCase().includes(q)
    )
  })

  const models = [...new Set(runs.map((r) => r.model).filter(Boolean))] as string[]
  const methods = [...new Set(runs.map((r) => r.method).filter(Boolean))] as string[]

  const summary = {
    total: runs.length,
    converged: runs.filter((r) => r.converged).length,
    avgQuality:
      runs.length > 0 ? runs.reduce((s, r) => s + (r.quality_score || 0), 0) / runs.length : 0,
    avgLoss:
      runs.filter((r) => (r.final_loss || 0) > 0).length > 0
        ? runs.filter((r) => (r.final_loss || 0) > 0).reduce((s, r) => s + (r.final_loss || 0), 0) /
          runs.filter((r) => (r.final_loss || 0) > 0).length
        : 0,
  }

  return {
    runs,
    loading,
    filteredRuns,
    models,
    methods,
    summary,
    searchQuery,
    setSearchQuery,
    filterModel,
    setFilterModel,
    filterMethod,
    setFilterMethod,
    format,
    setFormat,
    handleExport,
    selectedIds,
    toggleSelect,
    toggleSelectAll,
    clearSelection,
    bulkTag,
    setBulkTag,
    handleBulkTag,
    handleBulkBookmark,
    handleBulkDelete,
    deleting,
    selectedRun,
    selectRun,
    closeSelection,
    handleBookmark,
    handleDuplicate,
    handleDelete,
    handleExportSingle,
    newTag,
    setNewTag,
    handleAddTag,
    handleRemoveTag,
    editingNotes,
    startEditNotes,
    setEditingNotes,
    notesValue,
    setNotesValue,
    handleSaveNotes,
    compareRunId,
    changeCompareTarget,
    compareResult,
    handleCompare,
  }
}
