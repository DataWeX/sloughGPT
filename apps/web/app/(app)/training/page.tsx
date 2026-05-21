'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Button } from '@/components/ui/button'
import { Select } from '@/components/ui/select'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { StatCard, KpiGrid } from '@/components/strui'
import { useToastStore } from '@/lib/toast-store'
import { trainingJobsController } from '@/lib/controllers'
import { datasetController } from '@/lib/controllers'
import type { Dataset } from '@/lib/dataset-controller'
import { DatasetImportModal } from '@/components/DatasetImportModal'
import { cn } from '@/lib/cn'
import type { Checkpoint } from '@/lib/souls-controller'
import { PUBLIC_API_URL } from '@/lib/config'

function formatSize(bytes: number): string {
  if (bytes >= 1073741824) return `${(bytes / 1073741824).toFixed(2)} GB`
  if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${bytes} B`
}

function datasetLabel(ds: Dataset): string {
  const size = formatSize(ds.size)
  const suffix = ds.samples && ds.samples > 0 ? ` (${ds.samples.toLocaleString()} samples, ${size})` : ` (${size})`
  return `${ds.name}${suffix}`
}

export default function TrainingPage() {
  const addToast = useToastStore(s => s.addToast)
  const initialLoadDone = useRef(false)

  // ===== Auto-train state =====
  const [trainingPhase, setTrainingPhase] = useState('idle')
  const [trainingLoss, setTrainingLoss] = useState<number | null>(null)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const esRef = useRef<EventSource | null>(null)
  const trainingRunning = trainingPhase !== 'idle' && trainingPhase !== 'complete' && trainingPhase !== 'error'

  // ===== Datasets =====
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selectedDataset, setSelectedDataset] = useState('')
  const [loadingDatasets, setLoadingDatasets] = useState(false)
  const [importModalOpen, setImportModalOpen] = useState(false)

  // ===== Checkpoints =====
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [activeCheckpoint, setActiveCheckpoint] = useState<string | null>(null)

  // ===== Jobs =====
  const [jobs, setJobs] = useState<any[]>([])
  const [loadingJobs, setLoadingJobs] = useState(true)

  const fetchDatasets = useCallback(async () => {
    setLoadingDatasets(true)
    try {
      const list = await datasetController.list()
      setDatasets(list)
      if (!selectedDataset && list.length > 0 && !initialLoadDone.current) {
        initialLoadDone.current = true
        setSelectedDataset(list[0].id)
      }
    }
    catch { addToast('Failed to fetch datasets', 'error') }
    finally { setLoadingDatasets(false) }
  }, [selectedDataset, addToast])

  const fetchCheckpoints = useCallback(async () => {
    try {
      const data = await trainingJobsController.listCheckpoints()
      setCheckpoints(data)
    } catch { /* ignore */ }
  }, [])

  const fetchJobs = useCallback(async () => {
    try { setJobs(await trainingJobsController.list()) }
    catch { /* ignore */ }
    finally { setLoadingJobs(false) }
  }, [])

  const startTraining = async () => {
    try {
      const res = await fetch(`${PUBLIC_API_URL}/auto-train/start`, { method: 'POST' })
      if (!res.ok) { addToast('Failed to start training', 'error'); return }
      setTrainingPhase('GENERATE_DATA'); setTrainingProgress(0); setTrainingLoss(null)
      const es = new EventSource(`${PUBLIC_API_URL}/auto-train/stream`)
      esRef.current = es
      es.onmessage = (e) => {
        try {
          const env = JSON.parse(e.data)
          if (env.stream !== 'auto-train') return
          setTrainingPhase(env.phase || trainingPhase)
          if (env.data?.loss != null) setTrainingLoss(env.data.loss)
          if (env.data?.progress != null) setTrainingProgress(env.data.progress)
          if (env.status === 'complete') { es.close(); esRef.current = null; setTrainingPhase('complete'); addToast('Training complete', 'success'); fetchCheckpoints() }
          if (env.status === 'error') { es.close(); esRef.current = null; setTrainingPhase('error'); addToast('Training failed', 'error') }
        } catch { /* ignore */ }
      }
      es.onerror = () => { es.close(); esRef.current = null; setTrainingPhase('error') }
    } catch { addToast('Failed to start training', 'error') }
  }

  const stopTraining = () => {
    esRef.current?.close(); esRef.current = null
    fetch(`${PUBLIC_API_URL}/auto-train/stop`, { method: 'POST' }).catch(() => {})
    setTrainingPhase('idle'); setTrainingProgress(0); setTrainingLoss(null)
  }

  const handleLoadCheckpoint = async (name: string) => {
    try {
      await trainingJobsController.loadCheckpoint?.(name)
      setActiveCheckpoint(name)
      addToast(`Loaded checkpoint: ${name}`, 'success')
    } catch { addToast('Failed to load checkpoint', 'error') }
  }

  const handleDeleteCheckpoint = async (name: string) => {
    if (!confirm(`Delete checkpoint "${name}"?`)) return
    try {
      await trainingJobsController.deleteCheckpoint?.(name)
      setCheckpoints(prev => prev.filter(c => c.name !== name))
      if (activeCheckpoint === name) setActiveCheckpoint(null)
      addToast(`Deleted ${name}`, 'success')
    } catch { addToast('Failed to delete checkpoint', 'error') }
  }

  useEffect(() => { return () => esRef.current?.close() }, [])
  useEffect(() => { void fetchDatasets(); void fetchCheckpoints(); void fetchJobs() }, [fetchDatasets, fetchCheckpoints, fetchJobs])
  useEffect(() => {
    const id = setInterval(() => void fetchCheckpoints(), 10000)
    return () => clearInterval(id)
  }, [fetchCheckpoints])

  const runningJob = jobs.find(j => j.status === 'running')
  const completedCount = jobs.filter(j => j.status === 'completed').length

  return (
    <div className="sl-page mx-auto max-w-6xl">
      <AppRouteHeader
        className="items-start"
        left={<AppRouteHeaderLead title="Training" subtitle="Fine-tune models on your datasets" />}
        right={
          <div className="flex items-center gap-2">
            <Button size="sm" variant="ghost" onClick={() => { void fetchJobs(); void fetchCheckpoints() }}>Refresh</Button>
          </div>
        }
      />

      <div className="space-y-4">
        {/* Stats */}
        <KpiGrid columns={4}>
          <StatCard label="Total Jobs" value={jobs.length} />
          <StatCard label="Running" value={runningJob ? 1 : 0} />
          <StatCard label="Completed" value={completedCount} />
          <StatCard label="Checkpoints" value={checkpoints.length} />
        </KpiGrid>

        {/* Auto-train pipeline */}
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Auto-train pipeline</CardTitle>
              <div className="flex items-center gap-2">
                {trainingRunning ? (
                  <span className="relative flex h-2 w-2">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
                    <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
                  </span>
                ) : trainingPhase === 'complete' ? (
                  <span className="text-xs text-success">Complete</span>
                ) : trainingPhase === 'error' ? (
                  <span className="text-xs text-destructive">Failed</span>
                ) : null}
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                {trainingRunning ? (
                  <>
                    <span>{trainingPhase}</span>
                    {trainingLoss != null && (
                      <span className="font-mono text-xs">Loss: {trainingLoss.toFixed(4)}</span>
                    )}
                  </>
                ) : trainingPhase === 'complete' ? (
                  <span>Training finished. Checkpoints ready.</span>
                ) : (
                  <span>GPT2 teacher distills into a compact LSTM student model</span>
                )}
              </div>
              <div className="flex gap-2 shrink-0">
                {trainingRunning ? (
                  <Button size="sm" variant="outline" onClick={stopTraining}>Stop</Button>
                ) : (
                  <Button size="sm" onClick={startTraining}>Start training</Button>
                )}
              </div>
            </div>
            {trainingRunning && (
              <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                <div className="h-full bg-primary transition-all duration-500 rounded-full" style={{ width: `${Math.min(trainingProgress, 100)}%` }} />
              </div>
            )}
          </CardContent>
        </Card>

        {/* Dataset */}
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Training data</CardTitle>
              <div className="flex gap-1">
                <Button size="sm" variant="ghost" onClick={() => setImportModalOpen(true)}>Import</Button>
                <Button size="sm" variant="ghost" onClick={() => void fetchDatasets()} disabled={loadingDatasets}>
                  {loadingDatasets ? '...' : 'Refresh'}
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {datasets.length > 0 ? (
              <Select
                value={selectedDataset}
                onValueChange={setSelectedDataset}
                options={datasets.map(ds => ({
                  value: ds.id,
                  label: datasetLabel(ds),
                }))}
              />
            ) : (
              <div className="text-sm text-muted-foreground py-2">No datasets found. Import one to get started.</div>
            )}
            {(() => {
              const ds = datasets.find(d => d.id === selectedDataset)
              if (!ds) return null
              return (
                <div className="flex gap-3 text-[11px] text-muted-foreground">
                  {ds.samples != null && ds.samples > 0 && <span>{ds.samples.toLocaleString()} samples</span>}
                  {ds.type && <span>{ds.type}</span>}
                </div>
              )
            })()}
          </CardContent>
        </Card>

        {/* Checkpoints */}
        {checkpoints.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Checkpoints</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-2 sm:grid-cols-2">
                {checkpoints.slice().reverse().map((cp: any) => (
                  <div key={cp.name} className={cn("flex items-center justify-between rounded-lg border p-3 text-sm", activeCheckpoint === cp.name ? "border-primary/30 bg-primary/5" : "border-border/50")}>
                    <div className="min-w-0 flex-1">
                      <p className="truncate font-mono text-xs font-medium">{cp.name}</p>
                      {cp.loss != null && <p className="text-xs text-muted-foreground mt-0.5">Loss: {cp.loss.toFixed(4)}</p>}
                      {cp.traits && Object.keys(cp.traits).length > 0 && (
                        <p className="text-xs text-muted-foreground mt-0.5">Traits: {Object.entries(cp.traits).map(([k, v]) => `${k}: ${v}`).join(', ')}</p>
                      )}
                    </div>
                    <div className="flex items-center gap-1 shrink-0 ml-2">
                      {activeCheckpoint === cp.name ? (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary font-medium">Active</span>
                      ) : (
                        <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={() => handleLoadCheckpoint(cp.name)}>Load</Button>
                      )}
                      <Button size="sm" variant="ghost" className="h-6 text-xs text-destructive hover:text-destructive" onClick={() => handleDeleteCheckpoint(cp.name)}>Del</Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Jobs */}
        {jobs.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Job history</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="divide-y divide-border/50">
                {jobs.slice().reverse().map((job) => (
                  <div key={job.id} className="flex items-center justify-between px-4 py-3 text-sm">
                    <div className="min-w-0 flex-1">
                      <p className="truncate font-medium">{job.name}</p>
                      <p className="text-xs text-muted-foreground mt-0.5">{job.status} &middot; {job.created_at ? new Date(job.created_at).toLocaleDateString() : ''}</p>
                    </div>
                    {job.status === 'running' && <span className="relative flex h-2 w-2 shrink-0"><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-success/60" /><span className="relative inline-flex h-2 w-2 rounded-full bg-success" /></span>}
                    {job.status === 'completed' && <span className="text-xs text-success shrink-0">Done</span>}
                    {job.status === 'failed' && <span className="text-xs text-destructive shrink-0">Failed</span>}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {!trainingRunning && checkpoints.length === 0 && jobs.length === 0 && (
          <Card className="border-dashed py-8">
            <CardContent className="text-center text-sm text-muted-foreground">
              No training activity yet. Import a dataset and start training to create checkpoints.
            </CardContent>
          </Card>
        )}
      </div>

      <DatasetImportModal
        open={importModalOpen}
        onOpenChange={setImportModalOpen}
        onImportComplete={(datasetId: string) => {
          void fetchDatasets().then(() => setSelectedDataset(datasetId))
        }}
      />
    </div>
  )
}
