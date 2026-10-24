'use client'

import { useState } from 'react'
import { Card, CardContent, Button, Badge, Input, Select, SelectTrigger, SelectValue, SelectContent, SelectItem, IconDownload, IconSearch } from '@sloughgpt/strui'
import { IconClose } from '@/components/icons/NavIcons'
import { settingsController } from '@/lib/settings-controller'
import { downloadBlob } from '@/lib/download-utils'
import { formatDuration } from '@/lib/formatDuration'
import { useTrainingRuns } from '@/hooks/useTrainingRuns'
import { QualityBadge, formatRunDate } from './training-shared'
import { Clock, BarChart3, Trash2, Tag, Plus, Star, Copy, CheckSquare, Square } from 'lucide-react'

export function RunsSection() {
  const runs = useTrainingRuns()
  const [searchQuery, setSearchQuery] = useState('')
  const [format, setFormat] = useState<'json' | 'csv'>('json')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [newTag, setNewTag] = useState('')
  const [bulkTag, setBulkTag] = useState('')
  const [editingNotes, setEditingNotes] = useState(false)
  const [notesValue, setNotesValue] = useState('')
  const [confirmId, setConfirmId] = useState<string | null>(null)

  const selectedRun = runs.runs.find(r => r.run_id === selectedId) ?? null
  const filtered = runs.runs.filter(run => {
    if (!searchQuery) return true
    const q = searchQuery.toLowerCase()
    return (
      (run.model || '').toLowerCase().includes(q) ||
      (run.method || '').toLowerCase().includes(q) ||
      (run.dataset || '').toLowerCase().includes(q) ||
      run.run_id.toLowerCase().includes(q) ||
      (run.tags || []).some(t => t.toLowerCase().includes(q))
    )
  })
  const models = [...new Set(runs.runs.map(r => r.model).filter(Boolean))] as string[]
  const methods = [...new Set(runs.runs.map(r => r.method).filter(Boolean))] as string[]
  const summary = {
    total: runs.runs.length,
    converged: runs.runs.filter(r => r.converged).length,
    avgQuality: runs.runs.length > 0 ? runs.runs.reduce((s, r) => s + (r.quality_score || 0), 0) / runs.runs.length : 0,
  }

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const handleExport = async () => {
    const resp = await settingsController.exportTrainingHistory(format, 500)
    const content = format === 'csv' ? resp.content : JSON.stringify(resp, null, 2)
    downloadBlob(content ?? '', `training-history.${format}`, format === 'csv' ? 'text/csv' : 'application/json')
  }

  const openRun = (id: string, notes?: string) => {
    setSelectedId(id)
    setEditingNotes(false)
    setNotesValue(notes || '')
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Outcome history</h3>
        <div className="flex items-center gap-2">
          <Select value={format} onValueChange={v => setFormat(v as 'json' | 'csv')}>
            <SelectTrigger className="h-8 w-24 text-xs" aria-label="Export format">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="json">JSON</SelectItem>
              <SelectItem value="csv">CSV</SelectItem>
            </SelectContent>
          </Select>
          <Button size="sm" variant="outline" onClick={handleExport} leftIcon={<IconDownload className="h-4 w-4" />}>
            Export
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <Card><CardContent className="pt-4 text-center"><div className="text-2xl font-bold">{summary.total}</div><div className="text-xs text-muted-foreground">Total Runs</div></CardContent></Card>
        <Card><CardContent className="pt-4 text-center"><div className="text-2xl font-bold text-green-600">{summary.converged}</div><div className="text-xs text-muted-foreground">Converged</div></CardContent></Card>
        <Card><CardContent className="pt-4 text-center"><div className="text-2xl font-bold">{Math.round(summary.avgQuality * 100)}%</div><div className="text-xs text-muted-foreground">Avg Quality</div></CardContent></Card>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="relative flex-1 min-w-[200px]">
          <IconSearch className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input placeholder="Search runs, tags, notes..." value={searchQuery} onChange={e => setSearchQuery(e.target.value)} className="pl-8" aria-label="Search runs" />
          {searchQuery && <button onClick={() => setSearchQuery('')} className="absolute right-2 top-2.5" aria-label="Clear search"><IconClose className="h-4 w-4 text-muted-foreground" /></button>}
        </div>
        <select value={runs.filterModel} onChange={e => runs.setFilterModel(e.target.value)} className="text-sm border rounded px-2 py-1.5" aria-label="Filter by model">
          <option value="">All Models</option>
          {models.map(m => <option key={m} value={m}>{m}</option>)}
        </select>
        <select value={runs.filterMethod} onChange={e => runs.setFilterMethod(e.target.value)} className="text-sm border rounded px-2 py-1.5" aria-label="Filter by method">
          <option value="">All Methods</option>
          {methods.map(m => <option key={m} value={m}>{m}</option>)}
        </select>
      </div>

      {selectedIds.size > 0 && (
        <div className="flex items-center gap-2 p-3 bg-muted rounded-lg">
          <span className="text-sm font-medium">{selectedIds.size} selected</span>
          <Input placeholder="Tag name..." value={bulkTag} onChange={e => setBulkTag(e.target.value)} className="w-32 text-xs h-7" aria-label="Bulk tag name" />
          <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => { void runs.bulkTag(Array.from(selectedIds), bulkTag.trim()).then(() => { setBulkTag(''); setSelectedIds(new Set()) }) }} disabled={!bulkTag.trim()}>
            <Tag className="h-3 w-3 mr-1" /> Tag
          </Button>
          <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => { void runs.bulkBookmark(Array.from(selectedIds), true).then(() => setSelectedIds(new Set())) }}>Bookmark</Button>
          <Button size="sm" variant="destructive" className="h-7 text-xs" onClick={() => { void runs.bulkDelete(Array.from(selectedIds)).then(() => { setSelectedIds(new Set()); setSelectedId(null) }) }}>
            <Trash2 className="h-3 w-3 mr-1" /> Delete
          </Button>
          <Button size="sm" variant="ghost" className="h-7 text-xs ml-auto" onClick={() => setSelectedIds(new Set())}>Clear</Button>
        </div>
      )}

      <div className="flex gap-4">
        <div className="flex-1 space-y-2">
          {runs.loading ? (
            <Card><CardContent className="py-8 text-center text-sm text-muted-foreground">Loading runs…</CardContent></Card>
          ) : filtered.length === 0 ? (
            <Card><CardContent className="py-12 text-center text-muted-foreground"><BarChart3 className="h-12 w-12 mx-auto mb-3 opacity-30" /><p>No training runs found.</p></CardContent></Card>
          ) : filtered.map(run => (
            <Card key={run.run_id} className={`cursor-pointer ${selectedId === run.run_id ? 'ring-2 ring-primary' : ''}`} onClick={() => openRun(run.run_id, run.notes)}>
              <CardContent className="py-3 flex items-center gap-4">
                <button onClick={e => { e.stopPropagation(); toggleSelect(run.run_id) }} className="text-muted-foreground" aria-label={`Select ${run.run_id}`}>
                  {selectedIds.has(run.run_id) ? <CheckSquare className="h-4 w-4" /> : <Square className="h-4 w-4" />}
                </button>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-medium text-sm truncate">{run.model || 'unknown'}</span>
                    {run.method && <Badge variant="secondary" className="text-xs">{run.method}</Badge>}
                    {run.converged && <Badge className="bg-green-100 text-green-800 text-xs">converged</Badge>}
                    <QualityBadge score={run.quality_score} />
                  </div>
                  <div className="flex items-center gap-4 text-xs text-muted-foreground mt-1">
                    <span className="flex items-center gap-1"><Clock className="h-3 w-3" />{formatRunDate(run.timestamp)}</span>
                    {run.training_time_s ? <span>{formatDuration(run.training_time_s)}</span> : null}
                    {run.final_loss !== undefined && run.final_loss > 0 && <span>loss: {run.final_loss.toFixed(4)}</span>}
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  <Button size="sm" variant="ghost" className="h-7 px-2" onClick={e => { e.stopPropagation(); void runs.toggleBookmark(run.run_id) }} aria-label="Bookmark run">
                    <Star className={`h-4 w-4 ${run.bookmarked ? 'fill-current text-yellow-500' : ''}`} />
                  </Button>
                  <Button size="sm" variant="ghost" className="h-7 px-2" onClick={e => { e.stopPropagation(); void runs.duplicateRun(run.run_id) }} aria-label="Duplicate run">
                    <Copy className="h-4 w-4" />
                  </Button>
                  {confirmId === run.run_id ? (
                    <>
                      <Button size="sm" variant="destructive" className="h-7 text-xs" onClick={e => { e.stopPropagation(); void runs.deleteRun(run.run_id).then(() => { if (selectedId === run.run_id) setSelectedId(null); setConfirmId(null) }) }}>Confirm</Button>
                      <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={e => { e.stopPropagation(); setConfirmId(null) }}>Keep</Button>
                    </>
                  ) : (
                    <Button size="sm" variant="ghost" className="h-7 px-2 text-destructive" onClick={e => { e.stopPropagation(); setConfirmId(run.run_id) }} aria-label="Delete run">
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {selectedRun && (
          <Card className="w-80 flex-shrink-0">
            <CardContent className="pt-4 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-sm">Run Details</h3>
                <button onClick={() => setSelectedId(null)} aria-label="Close details"><IconClose className="h-4 w-4 text-muted-foreground" /></button>
              </div>
              <div className="space-y-2 text-sm">
                <div><span className="text-muted-foreground">Model:</span> {selectedRun.model}</div>
                <div><span className="text-muted-foreground">Method:</span> {selectedRun.method}</div>
                <div><span className="text-muted-foreground">Quality:</span> {selectedRun.quality_score ? `${Math.round(selectedRun.quality_score * 100)}%` : '-'}</div>
                <div><span className="text-muted-foreground">Duration:</span> {formatDuration(selectedRun.training_time_s ?? null)}</div>
              </div>
              <div className="border-t pt-3">
                <div className="flex items-center justify-between mb-2"><span className="text-sm font-medium">Tags</span></div>
                <div className="flex flex-wrap gap-1 mb-2">
                  {(selectedRun.tags || []).map(tag => (
                    <Badge key={tag} className="bg-purple-100 text-purple-800 text-xs flex items-center gap-1">
                      <Tag className="h-2.5 w-2.5" />{tag}
                      <button onClick={() => { void runs.removeTag(selectedRun.run_id, tag) }} aria-label={`Remove tag ${tag}`}><IconClose className="h-3 w-3" /></button>
                    </Badge>
                  ))}
                </div>
                <div className="flex gap-1">
                  <Input placeholder="Add tag..." value={newTag} onChange={e => setNewTag(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') void runs.addTag(selectedRun.run_id, newTag.trim()).then(() => setNewTag('')) }} className="text-xs h-7" aria-label="New tag" />
                  <Button size="sm" variant="outline" className="h-7 px-2" onClick={() => { void runs.addTag(selectedRun.run_id, newTag.trim()).then(() => setNewTag('')) }} disabled={!newTag.trim()}><Plus className="h-3 w-3" /></Button>
                </div>
              </div>
              <div className="border-t pt-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Notes</span>
                  {!editingNotes && <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={() => { setEditingNotes(true); setNotesValue(selectedRun.notes || '') }}>Edit</Button>}
                </div>
                {editingNotes ? (
                  <div className="space-y-2">
                    <textarea value={notesValue} onChange={e => setNotesValue(e.target.value)} className="w-full text-xs border rounded p-2 min-h-[80px]" aria-label="Run notes" />
                    <div className="flex gap-1">
                      <Button size="sm" className="h-6 text-xs" onClick={() => { void runs.saveNotes(selectedRun.run_id, notesValue).then(() => setEditingNotes(false)) }}>Save</Button>
                      <Button size="sm" variant="outline" className="h-6 text-xs" onClick={() => setEditingNotes(false)}>Cancel</Button>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-muted-foreground">{selectedRun.notes || 'No notes yet.'}</p>
                )}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
