'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, Button, Badge, Input } from '@sloughgpt/strui'
import { settingsController } from '@/lib/settings-controller'
import { Clock, Download, BarChart3, Trash2, Search, X, Tag, Plus, GitCompare, Star, Copy, CheckSquare, Square } from 'lucide-react'

interface TrainingRun {
  run_id: string
  timestamp: number
  model?: string
  method?: string
  epochs?: number
  final_loss?: number
  best_loss?: number
  perplexity?: number
  quality_score?: number
  training_time_s?: number
  converged?: boolean
  early_stopped?: boolean
  dataset?: string
  tags?: string[]
  notes?: string
  bookmarked?: boolean
}

function formatDate(ts: number) {
  if (!ts) return '-'
  try { return new Date(ts * 1000).toLocaleString() } catch { return String(ts) }
}

function formatDuration(secs?: number) {
  if (!secs) return '-'
  const m = Math.floor(secs / 60)
  const s = Math.round(secs % 60)
  return m > 0 ? `${m}m ${s}s` : `${s}s`
}

function qualityBadge(score?: number) {
  if (score === undefined || score === null) return null
  const color = score >= 0.8 ? 'bg-green-100 text-green-800' :
                score >= 0.6 ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'
  return <Badge className={color}>{Math.round(score * 100)}%</Badge>
}

function downloadBlob(content: string, filename: string, mime: string) {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export default function TrainingRunsPage() {
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

  useEffect(() => { fetchRuns() }, [fetchRuns])

  const handleExport = async () => {
    try {
      const resp = await settingsController.exportTrainingHistory(format, 500)
      const content = format === 'csv' ? resp.content : JSON.stringify(resp, null, 2)
      const blob = new Blob(
        [content ?? ''],
        { type: format === 'csv' ? 'text/csv' : 'application/json' }
      )
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
      setRuns(prev => prev.filter(r => r.run_id !== runId))
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
      setRuns(prev => prev.map(r => r.run_id === selectedRun.run_id ? updated as unknown as TrainingRun : r))
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
      setRuns(prev => prev.map(r => r.run_id === selectedRun.run_id ? updated as unknown as TrainingRun : r))
    } catch (err) {
      console.error('Remove tag failed:', err)
    }
  }

  const handleSaveNotes = async () => {
    if (!selectedRun) return
    try {
      const updated = await settingsController.setRunNotes(selectedRun.run_id, notesValue)
      setSelectedRun(updated as unknown as TrainingRun)
      setRuns(prev => prev.map(r => r.run_id === selectedRun.run_id ? updated as unknown as TrainingRun : r))
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
      setSelectedRun(prev => prev?.run_id === runId ? updated as unknown as TrainingRun : prev)
      setRuns(prev => prev.map(r => r.run_id === runId ? { ...r, bookmarked: (updated as unknown as TrainingRun).bookmarked } : r))
    } catch (err) {
      console.error('Bookmark failed:', err)
    }
  }

  const handleDuplicate = async (runId: string) => {
    try {
      const newRun = await settingsController.duplicateTrainingRun(runId)
      setRuns(prev => [newRun as unknown as TrainingRun, ...prev])
    } catch (err) {
      console.error('Duplicate failed:', err)
    }
  }

  // Bulk operations
  const toggleSelect = (runId: string) => {
    setSelectedIds(prev => {
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
      setSelectedIds(new Set(filteredRuns.map(r => r.run_id)))
    }
  }

  const handleBulkDelete = async () => {
    if (!confirm(`Delete ${selectedIds.size} training runs?`)) return
    try {
      await settingsController.bulkDeleteRuns(Array.from(selectedIds))
      setRuns(prev => prev.filter(r => !selectedIds.has(r.run_id)))
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

  const filteredRuns = runs.filter(run => {
    if (!searchQuery) return true
    const q = searchQuery.toLowerCase()
    return (
      (run.model || '').toLowerCase().includes(q) ||
      (run.method || '').toLowerCase().includes(q) ||
      (run.dataset || '').toLowerCase().includes(q) ||
      run.run_id.toLowerCase().includes(q) ||
      (run.tags || []).some(t => t.toLowerCase().includes(q)) ||
      (run.notes || '').toLowerCase().includes(q)
    )
  })

  const models = [...new Set(runs.map(r => r.model).filter(Boolean))]
  const methods = [...new Set(runs.map(r => r.method).filter(Boolean))]

  const summary = {
    total: runs.length,
    converged: runs.filter(r => r.converged).length,
    avgQuality: runs.length > 0
      ? (runs.reduce((s, r) => s + (r.quality_score || 0), 0) / runs.length)
      : 0,
    avgLoss: runs.filter(r => (r.final_loss || 0) > 0).length > 0
      ? (runs.filter(r => (r.final_loss || 0) > 0).reduce((s, r) => s + (r.final_loss || 0), 0) /
         runs.filter(r => (r.final_loss || 0) > 0).length)
      : 0,
  }

  return (
    <PageContainer title="Training Runs">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Training Runs" />}
        right={
          <div className="flex items-center gap-2">
            <select
              value={format}
              onChange={(e) => setFormat(e.target.value as 'json' | 'csv')}
              className="text-sm border rounded px-2 py-1"
            >
              <option value="json">JSON</option>
              <option value="csv">CSV</option>
            </select>
            <Button size="sm" variant="outline" onClick={handleExport}>
              <Download className="h-4 w-4 mr-1" /> Export
            </Button>
          </div>
        }
      />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <Card>
          <CardContent className="pt-4 text-center">
            <div className="text-2xl font-bold">{summary.total}</div>
            <div className="text-xs text-muted-foreground">Total Runs</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 text-center">
            <div className="text-2xl font-bold text-green-600">{summary.converged}</div>
            <div className="text-xs text-muted-foreground">Converged</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 text-center">
            <div className="text-2xl font-bold">{Math.round(summary.avgQuality * 100)}%</div>
            <div className="text-xs text-muted-foreground">Avg Quality</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 text-center">
            <div className="text-2xl font-bold">{summary.avgLoss.toFixed(4)}</div>
            <div className="text-xs text-muted-foreground">Avg Loss</div>
          </CardContent>
        </Card>
      </div>

      <div className="flex flex-wrap items-center gap-2 mb-4">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search runs, tags, notes..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-8"
          />
          {searchQuery && (
            <button onClick={() => setSearchQuery('')} className="absolute right-2 top-2.5">
              <X className="h-4 w-4 text-muted-foreground" />
            </button>
          )}
        </div>
        <select
          value={filterModel}
          onChange={(e) => setFilterModel(e.target.value)}
          className="text-sm border rounded px-2 py-1.5"
        >
          <option value="">All Models</option>
          {models.map(m => <option key={m} value={m}>{m}</option>)}
        </select>
        <select
          value={filterMethod}
          onChange={(e) => setFilterMethod(e.target.value)}
          className="text-sm border rounded px-2 py-1.5"
        >
          <option value="">All Methods</option>
          {methods.map(m => <option key={m} value={m}>{m}</option>)}
        </select>
      </div>

      {selectedIds.size > 0 && (
        <div className="flex items-center gap-2 mb-4 p-3 bg-muted rounded-lg">
          <span className="text-sm font-medium">{selectedIds.size} selected</span>
          <div className="flex items-center gap-1 ml-2">
            <Input
              placeholder="Tag name..."
              value={bulkTag}
              onChange={(e) => setBulkTag(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleBulkTag()}
              className="w-32 text-xs h-7"
            />
            <Button size="sm" variant="outline" className="h-7 text-xs" onClick={handleBulkTag} disabled={!bulkTag.trim()}>
              <Tag className="h-3 w-3 mr-1" /> Tag
            </Button>
          </div>
          <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => handleBulkBookmark(true)}>
            <Star className="h-3 w-3 mr-1" /> Bookmark
          </Button>
          <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => handleBulkBookmark(false)}>
            Unbookmark
          </Button>
          <Button size="sm" variant="destructive" className="h-7 text-xs" onClick={handleBulkDelete}>
            <Trash2 className="h-3 w-3 mr-1" /> Delete
          </Button>
          <Button size="sm" variant="ghost" className="h-7 text-xs ml-auto" onClick={() => setSelectedIds(new Set())}>
            Clear Selection
          </Button>
        </div>
      )}

      <div className="flex gap-4">
        <div className="flex-1 space-y-2">
          {loading ? (
            Array.from({ length: 5 }).map((_, i) => (
              <Card key={i} className="animate-pulse">
                <CardContent className="py-4">
                  <div className="h-4 bg-muted rounded w-1/3 mb-2" />
                  <div className="h-3 bg-muted rounded w-2/3" />
                </CardContent>
              </Card>
            ))
          ) : filteredRuns.length === 0 ? (
            <Card>
              <CardContent className="py-12 text-center text-muted-foreground">
                <BarChart3 className="h-12 w-12 mx-auto mb-3 opacity-30" />
                <p>No training runs found.</p>
              </CardContent>
            </Card>
          ) : (
            <>
              <div className="flex items-center gap-2 mb-2">
                <button onClick={toggleSelectAll} className="text-muted-foreground hover:text-foreground">
                  {selectedIds.size === filteredRuns.length && filteredRuns.length > 0
                    ? <CheckSquare className="h-4 w-4" />
                    : <Square className="h-4 w-4" />
                  }
                </button>
                <span className="text-xs text-muted-foreground">Select all</span>
              </div>
              {filteredRuns.map((run) => (
                <Card
                  key={run.run_id}
                  className={`hover:shadow-sm transition-shadow cursor-pointer ${selectedRun?.run_id === run.run_id ? 'ring-2 ring-primary' : ''} ${selectedIds.has(run.run_id) ? 'bg-muted/30' : ''}`}
                  onClick={() => { setSelectedRun(run); setEditingNotes(false); setNotesValue(run.notes || '') }}
                >
                  <CardContent className="py-3 flex items-center gap-4">
                    <button
                      onClick={(e) => { e.stopPropagation(); toggleSelect(run.run_id) }}
                      className="flex-shrink-0 text-muted-foreground hover:text-foreground"
                    >
                      {selectedIds.has(run.run_id)
                        ? <CheckSquare className="h-4 w-4" />
                        : <Square className="h-4 w-4" />
                      }
                    </button>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-medium text-sm truncate">
                          {run.model || 'unknown'}
                        </span>
                        {run.method && (
                          <Badge variant="secondary" className="text-xs">{run.method}</Badge>
                        )}
                        {run.converged && (
                          <Badge className="bg-green-100 text-green-800 text-xs">converged</Badge>
                        )}
                        {qualityBadge(run.quality_score)}
                        {(run.tags || []).map(tag => (
                          <Badge key={tag} className="bg-purple-100 text-purple-800 text-xs">
                            <Tag className="h-2.5 w-2.5 mr-0.5" />{tag}
                          </Badge>
                        ))}
                      </div>
                      <div className="flex items-center gap-4 text-xs text-muted-foreground mt-1">
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {formatDate(run.timestamp)}
                        </span>
                        {run.training_time_s && <span>{formatDuration(run.training_time_s)}</span>}
                        {run.epochs && <span>{run.epochs} epochs</span>}
                        {run.final_loss !== undefined && run.final_loss > 0 && <span>loss: {run.final_loss.toFixed(4)}</span>}
                        {run.perplexity !== undefined && run.perplexity > 0 && <span>ppl: {run.perplexity.toFixed(2)}</span>}
                        {run.notes && <span className="italic truncate max-w-[200px]">&quot;{run.notes}&quot;</span>}
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <Button
                        size="sm"
                        variant="ghost"
                        className={`h-7 px-2 ${run.bookmarked ? 'text-yellow-500' : 'text-muted-foreground'}`}
                        onClick={(e) => { e.stopPropagation(); handleBookmark(run.run_id) }}
                      >
                        <Star className={`h-4 w-4 ${run.bookmarked ? 'fill-current' : ''}`} />
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-7 px-2 text-muted-foreground"
                        onClick={(e) => { e.stopPropagation(); handleDuplicate(run.run_id) }}
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="text-destructive hover:text-destructive"
                        onClick={(e) => { e.stopPropagation(); handleDelete(run.run_id) }}
                        disabled={deleting === run.run_id}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </>
          )}
        </div>

        {selectedRun && (
          <Card className="w-80 flex-shrink-0">
            <CardContent className="pt-4 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-sm">Run Details</h3>
                <div className="flex items-center gap-1">
                  <Button
                    size="sm"
                    variant="ghost"
                    className={`h-6 px-2 ${selectedRun.bookmarked ? 'text-yellow-500' : 'text-muted-foreground'}`}
                    onClick={() => handleBookmark(selectedRun.run_id)}
                  >
                    <Star className={`h-3 w-3 ${selectedRun.bookmarked ? 'fill-current' : ''}`} />
                  </Button>
                  <Button size="sm" variant="ghost" className="h-6 px-2" onClick={() => handleExportSingle(selectedRun.run_id)}>
                    <Download className="h-3 w-3 mr-1" /> Export
                  </Button>
                  <button onClick={() => setSelectedRun(null)}>
                    <X className="h-4 w-4 text-muted-foreground" />
                  </button>
                </div>
              </div>
              <div className="space-y-2 text-sm">
                <div><span className="text-muted-foreground">ID:</span> <span className="font-mono text-xs">{selectedRun.run_id}</span></div>
                <div><span className="text-muted-foreground">Model:</span> {selectedRun.model}</div>
                <div><span className="text-muted-foreground">Method:</span> {selectedRun.method}</div>
                <div><span className="text-muted-foreground">Dataset:</span> {selectedRun.dataset}</div>
                <div><span className="text-muted-foreground">Epochs:</span> {selectedRun.epochs}</div>
                <div><span className="text-muted-foreground">Final Loss:</span> {selectedRun.final_loss?.toFixed(4)}</div>
                <div><span className="text-muted-foreground">Best Loss:</span> {selectedRun.best_loss?.toFixed(4)}</div>
                <div><span className="text-muted-foreground">Perplexity:</span> {selectedRun.perplexity?.toFixed(2)}</div>
                <div><span className="text-muted-foreground">Quality:</span> {selectedRun.quality_score ? `${Math.round(selectedRun.quality_score * 100)}%` : '-'}</div>
                <div><span className="text-muted-foreground">Duration:</span> {formatDuration(selectedRun.training_time_s)}</div>
                <div><span className="text-muted-foreground">Converged:</span> {selectedRun.converged ? 'Yes' : 'No'}</div>
                <div><span className="text-muted-foreground">Early Stopped:</span> {selectedRun.early_stopped ? 'Yes' : 'No'}</div>
              </div>

              <div className="border-t pt-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Tags</span>
                </div>
                <div className="flex flex-wrap gap-1 mb-2">
                  {(selectedRun.tags || []).map(tag => (
                    <Badge key={tag} className="bg-purple-100 text-purple-800 text-xs flex items-center gap-1">
                      <Tag className="h-2.5 w-2.5" />{tag}
                      <button onClick={() => handleRemoveTag(tag)} className="ml-0.5 hover:text-purple-600">
                        <X className="h-3 w-3" />
                      </button>
                    </Badge>
                  ))}
                </div>
                <div className="flex gap-1">
                  <Input
                    placeholder="Add tag..."
                    value={newTag}
                    onChange={(e) => setNewTag(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAddTag()}
                    className="text-xs h-7"
                  />
                  <Button size="sm" variant="outline" className="h-7 px-2" onClick={handleAddTag} disabled={!newTag.trim()}>
                    <Plus className="h-3 w-3" />
                  </Button>
                </div>
              </div>

              <div className="border-t pt-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Notes</span>
                  {!editingNotes && (
                    <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={() => { setEditingNotes(true); setNotesValue(selectedRun.notes || '') }}>
                      Edit
                    </Button>
                  )}
                </div>
                {editingNotes ? (
                  <div className="space-y-2">
                    <textarea
                      value={notesValue}
                      onChange={(e) => setNotesValue(e.target.value)}
                      className="w-full text-xs border rounded p-2 min-h-[80px]"
                      placeholder="Add notes about this training run..."
                    />
                    <div className="flex gap-1">
                      <Button size="sm" className="h-6 text-xs" onClick={handleSaveNotes}>Save</Button>
                      <Button size="sm" variant="outline" className="h-6 text-xs" onClick={() => setEditingNotes(false)}>Cancel</Button>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-muted-foreground">
                    {selectedRun.notes || 'No notes yet.'}
                  </p>
                )}
              </div>

              <div className="border-t pt-3">
                <div className="flex items-center gap-2 mb-2">
                  <GitCompare className="h-3.5 w-3.5 text-muted-foreground" />
                  <span className="text-sm font-medium">Compare</span>
                </div>
                <div className="flex gap-1">
                  <select
                    value={compareRunId}
                    onChange={(e) => { setCompareRunId(e.target.value); setCompareResult(null) }}
                    className="text-xs border rounded px-2 py-1 flex-1"
                  >
                    <option value="">Select run to compare...</option>
                    {runs.filter(r => r.run_id !== selectedRun.run_id).map(r => (
                      <option key={r.run_id} value={r.run_id}>
                        {r.model} - {r.run_id.slice(0, 12)}
                      </option>
                    ))}
                  </select>
                  <Button size="sm" variant="outline" className="h-7 text-xs" onClick={handleCompare} disabled={!compareRunId}>
                    Compare
                  </Button>
                </div>
                {compareResult && (
                  <div className="mt-2 text-xs space-y-1 bg-muted/50 rounded p-2">
                    <div className="font-medium">Differences:</div>
                    {Object.entries(compareResult.differences as Record<string, { run_a: unknown; run_b: unknown }> || {}).map(([field, diff]) => (
                      <div key={field} className="flex justify-between">
                        <span className="text-muted-foreground">{field}:</span>
                        <span>{String(diff.run_a)} → {String(diff.run_b)}</span>
                      </div>
                    ))}
                    {compareResult.a_wins !== undefined && (
                      <div className="pt-1 text-muted-foreground">
                        Score: {String(compareResult.a_wins)} vs {String(compareResult.b_wins)}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </PageContainer>
  )
}
