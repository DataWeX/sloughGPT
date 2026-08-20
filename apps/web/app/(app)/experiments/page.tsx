'use client'

import { useState, useEffect, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid } from '@sloughgpt/strui'
import { IconRefresh, IconTrash } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { experimentsController } from '@/lib/experiments-controller'
import { ExperimentDetailsCard } from '@/components/experiments/ExperimentDetailsCard'
import { useToastStore } from '@/lib/toast-store'

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Awaited<ReturnType<typeof experimentsController.list>>>([])
  const [loading, setLoading] = useState(true)
  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [metricName, setMetricName] = useState('')
  const [metricValue, setMetricValue] = useState('')
  const [paramName, setParamName] = useState('')
  const [paramValue, setParamValue] = useState('')
  const [logMsg, setLogMsg] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [batchDeleting, setBatchDeleting] = useState(false)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const addToast = useToastStore(s => s.addToast)

  const fetchExperiments = async () => {
    setLoading(true)
    try {
      setExperiments(await experimentsController.list())
    } catch {
      addToast('Failed to load experiments', 'error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchExperiments() }, [])

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(fetchExperiments, 15000)
      const onVis = () => { if (!document.hidden && intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = setInterval(fetchExperiments, 15000) } }
      document.addEventListener('visibilitychange', onVis)
      return () => { clearInterval(intervalRef.current!); document.removeEventListener('visibilitychange', onVis) }
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }
  }, [autoRefresh])

  const handleCreate = async () => {
    if (!newName.trim()) return
    setCreating(true)
    try {
      await experimentsController.create(newName)
      setNewName('')
      await fetchExperiments()
    } catch {
      addToast('Failed to create experiment', 'error')
    } finally {
      setCreating(false)
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await experimentsController.delete(id)
      await fetchExperiments()
    } catch {
      addToast('Failed to delete experiment', 'error')
    }
  }

  const handleLogMetric = async () => {
    if (!selectedId || !metricName.trim() || !metricValue) return
    setLogMsg(null)
    try {
      const res = await experimentsController.logMetric(selectedId, metricName, Number(metricValue))
      setLogMsg(res.status === 'logged' ? `Logged ${metricName}=${metricValue}` : 'Failed')
      setMetricName('')
      setMetricValue('')
    } catch { setLogMsg('Failed') }
  }

  const handleLogParam = async () => {
    if (!selectedId || !paramName.trim() || !paramValue) return
    setLogMsg(null)
    try {
      const res = await experimentsController.logParam(selectedId, paramName, paramValue)
      setLogMsg(res.status === 'logged' ? `Logged ${paramName}=${paramValue}` : 'Failed')
      setParamName('')
      setParamValue('')
    } catch { setLogMsg('Failed') }
  }

  const handleComplete = async (id: string) => {
    try {
      await experimentsController.complete(id)
      setLogMsg(`Experiment ${id} marked complete`)
    } catch { setLogMsg('Failed') }
  }

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const toggleSelectAll = () => {
    const filtered = experiments.filter(exp => !search || exp.id.toLowerCase().includes(search.toLowerCase()))
    if (selectedIds.size === filtered.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(filtered.map(exp => exp.id)))
    }
  }

  const handleBatchDelete = async () => {
    if (selectedIds.size === 0) return
    setBatchDeleting(true)
    try {
      await Promise.all(Array.from(selectedIds).map(id => experimentsController.delete(id)))
      setSelectedIds(new Set())
      await fetchExperiments()
      addToast(`Deleted ${selectedIds.size} experiments`, 'success')
    } catch {
      addToast('Batch delete failed', 'error')
    } finally {
      setBatchDeleting(false)
    }
  }

  if (loading) {
    return (
      <PageContainer
        title="Experiments"
        subtitle="ML experiment tracking"
        loading
      >
        <KpiGrid>
          <StatCard label="Total" value="..." />
          <StatCard label="Selected" value="..." />
          <StatCard label="Auto-refresh" value="..." />
          <StatCard label="Last Created" value="..." />
        </KpiGrid>
        <Card><CardContent><div className="h-32 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
      </PageContainer>
    )
  }

  return (
    <PageContainer
      title="Experiments"
      subtitle={`${experiments.length} experiments`}
    >
        <KpiGrid>
          <StatCard label="Total Experiments" value={String(experiments.length)} />
          <StatCard label="Selected" value={String(selectedIds.size)} />
          <StatCard label="Auto-refresh" value={autoRefresh ? 'ON' : 'OFF'} />
          <StatCard label="Last Created" value={experiments.length > 0 ? experiments[0].id.slice(0, 20) : 'None'} />
        </KpiGrid>

        {logMsg && (
          <div className="rounded-md bg-primary/10 border border-primary/20 px-4 py-3 text-sm text-primary">
            {logMsg}
            <button className="ml-2 underline" onClick={() => setLogMsg(null)}>Dismiss</button>
          </div>
        )}

        <Card>
          <CardHeader>
            <CardTitle className="text-base">New Experiment</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-2">
              <Input
                value={newName}
                onChange={e => setNewName(e.target.value)}
                placeholder="Experiment name"
                onKeyDown={e => e.key === 'Enter' && handleCreate()}
              />
              <Button size="sm" onClick={handleCreate} disabled={creating || !newName.trim()}>
                {creating ? 'Creating...' : 'Create'}
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Experiments</CardTitle>
            <div className="flex items-center gap-2">
              <Input
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search..."
                className="h-9 w-32 text-sm"
              />
              <Button size="sm" variant={autoRefresh ? 'default' : 'ghost'} onClick={() => setAutoRefresh(!autoRefresh)}>
                {autoRefresh ? 'Auto' : 'Refresh'}
              </Button>
              <Button size="sm" variant="ghost" onClick={fetchExperiments}>
                <IconRefresh className="h-4 w-4" />
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {experiments.length === 0 ? (
              <div className="text-center py-6 space-y-2">
                <p className="text-sm text-muted-foreground">No experiments yet.</p>
                <Button size="sm" variant="outline" className="h-8 text-xs" onClick={() => setCreateDialogOpen(true)}>
                  New Experiment
                </Button>
              </div>
            ) : (
              <>
                {selectedIds.size > 0 && (
                  <div className="flex items-center gap-2 rounded-md bg-destructive/5 border border-destructive/20 px-3 py-2 mb-2">
                    <span className="text-sm text-destructive font-medium">{selectedIds.size} selected</span>
                    <Button size="sm" variant="ghost" className="text-destructive h-8 text-xs ml-auto" onClick={handleBatchDelete} disabled={batchDeleting}>
                      {batchDeleting ? 'Deleting...' : 'Delete Selected'}
                    </Button>
                    <Button size="sm" variant="ghost" className="h-8 text-xs" onClick={() => setSelectedIds(new Set())}>
                      Clear
                    </Button>
                  </div>
                )}
                <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer mb-2">
                  <input
                    type="checkbox"
                    checked={selectedIds.size === experiments.filter(exp => !search || exp.id.toLowerCase().includes(search.toLowerCase())).length && experiments.length > 0}
                    onChange={toggleSelectAll}
                    className="rounded border-border"
                  />
                  Select all
                </label>
                <div className="space-y-2">
                  {experiments
                    .filter(exp => !search || exp.id.toLowerCase().includes(search.toLowerCase()))
                    .map(exp => (
                    <div
                      key={exp.id}
                      className={`flex items-center justify-between rounded-md border px-3 py-2 text-sm transition-colors cursor-pointer ${
                        selectedId === exp.id
                          ? 'border-primary/40 bg-primary/5'
                          : selectedIds.has(exp.id)
                            ? 'border-primary/40 bg-primary/5'
                            : 'border-border/60 hover:bg-muted/50'
                      }`}
                      onClick={() => setSelectedId(selectedId === exp.id ? null : exp.id)}
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <input
                          type="checkbox"
                          checked={selectedIds.has(exp.id)}
                          onChange={() => toggleSelect(exp.id)}
                          onClick={e => e.stopPropagation()}
                          className="rounded border-border shrink-0"
                        />
                        <div className="font-medium truncate">{exp.id}</div>
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        <Button size="sm" variant="ghost" onClick={e => { e.stopPropagation(); handleComplete(exp.id) }}>
                          Done
                        </Button>
                        <Button size="sm" variant="ghost" className="text-destructive" onClick={e => { e.stopPropagation(); handleDelete(exp.id) }}>
                          <IconTrash className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </CardContent>
        </Card>

        {selectedId && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Log to: {selectedId}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex gap-2">
                <Input value={metricName} onChange={e => setMetricName(e.target.value)} placeholder="Metric name" className="flex-1" />
                <Input value={metricValue} onChange={e => setMetricValue(e.target.value)} placeholder="Value" type="number" className="w-24" />
                <Button size="sm" onClick={handleLogMetric} disabled={!metricName.trim() || !metricValue}>Log Metric</Button>
              </div>
              <div className="flex gap-2">
                <Input value={paramName} onChange={e => setParamName(e.target.value)} placeholder="Param name" className="flex-1" />
                <Input value={paramValue} onChange={e => setParamValue(e.target.value)} placeholder="Value" className="w-32" />
                <Button size="sm" variant="outline" onClick={handleLogParam} disabled={!paramName.trim() || !paramValue}>Log Param</Button>
              </div>
            </CardContent>
          </Card>
        )}

        {selectedId && (
          <ExperimentDetailsCard experimentId={selectedId} />
        )}
      </PageContainer>
  )
}
