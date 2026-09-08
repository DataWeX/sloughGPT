'use client'

import { useRouter } from 'next/navigation'
import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, IconDownload, cn, Spinner } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { StatusBanner } from '@/components/composed/StatusBanner'
import { modelController } from '@/lib/model-controller'
import { trainingJobsController } from '@/lib/training-controller'
import { ExportHistoryCard, recordExport } from '@/components/export/ExportHistoryCard'
import { downloadJson, downloadBlob } from '@/lib/download-utils'
import { apiGet } from '@/lib/http-client'

interface ExportFormat {
  key: string
  label: string
  description: string
}

interface Checkpoint {
  name: string
  path: string
  size_bytes?: number
  created_at?: string
  loss?: number
}

export default function ExportPage() {
  const router = useRouter()
  const [formats, setFormats] = useState<ExportFormat[]>([])
  const [selectedFormat, setSelectedFormat] = useState('sou')
  const [exporting, setExporting] = useState(false)
  const [exportResult, setExportResult] = useState<string | null>(null)
  const [exportError, setExportError] = useState<string | null>(null)
  const [formatLoadError, setFormatLoadError] = useState<string | null>(null)

  const [exportingPairs, setExportingPairs] = useState(false)

  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [loadingCheckpoints, setLoadingCheckpoints] = useState(false)

  useEffect(() => {
    modelController.getExportFormats?.()
      .then((res: Record<string, string>) => {
        const list = Object.entries(res).map(([key, description]) => ({
          key,
          label: key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
          description,
        }))
        setFormats(list)
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : String(err)
        setFormatLoadError(`Could not load export formats: ${msg}`)
      })
  }, [])

  const fetchCheckpoints = async () => {
    setLoadingCheckpoints(true)
    try {
      const data = await apiGet<{ checkpoints: Checkpoint[] }>('/training/checkpoints')
      setCheckpoints(data?.checkpoints ?? [])
    } catch {
      setCheckpoints([])
    } finally {
      setLoadingCheckpoints(false)
    }
  }

  useEffect(() => { fetchCheckpoints() }, [])

  const handleExportModel = async () => {
    setExporting(true)
    setExportResult(null)
    setExportError(null)
    try {
      const { apiPost } = await import('@/lib/http-client')
      const res = await apiPost<{ format: string; files: Record<string, string> }>(
        '/models/export',
        { format: selectedFormat, output_path: 'models/exported', include_tokenizer: true },
      )
      const fileCount = Object.keys(res.files ?? {}).length
      setExportResult(`Exported ${fileCount} file(s) in ${res.format} format`)
      recordExport(res.format, fileCount).catch(() => {})
    } catch (err) {
      setExportError(err instanceof Error ? err.message : 'Could not export')
    } finally {
      setExporting(false)
    }
  }

  const handleExportTrainingData = async () => {
    setExportingPairs(true)
    try {
      const blob = await trainingJobsController.exportTrainingPairs()
      downloadBlob(blob, `training-pairs-${Date.now()}.jsonl`)
    } catch (err) {
      setExportError(err instanceof Error ? err.message : 'Could not export training data')
    } finally {
      setExportingPairs(false)
    }
  }

  const handleDownloadCheckpoint = async (name: string) => {
    try {
      const blob = await trainingJobsController.downloadCheckpoint(name)
      downloadBlob(blob, `${name}.soul`)
    } catch (err) {
      setExportError(err instanceof Error ? err.message : 'Could not checkpoint download')
    }
  }

  const fmtBytes = (bytes?: number) => {
    if (!bytes) return ''
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return (
    <PageContainer title="Export" subtitle="Export models, training data, and checkpoints">
      {exportResult && (
        <StatusBanner variant="success" message={exportResult} dismissible={false} />
      )}
      {exportError && (
        <StatusBanner variant="error" message={exportError} />
      )}
      {formatLoadError && (
        <StatusBanner variant="error" message={formatLoadError} dismissible={false} />
      )}

      <ExportHistoryCard />

      <Card>
        <CardHeader className="pb-2 pt-2.5 px-2.5">
          <CardTitle className="text-[11px] font-medium">Model Export</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 px-2.5 pb-2.5">
          <p className="text-[10px] text-muted-foreground/60">
            Export the currently loaded model to a file format.
          </p>
          <div className="flex flex-wrap gap-1">
            {formats.map(f => (
              <button
                key={f.key}
                type="button"
                onClick={() => setSelectedFormat(f.key)}
                className={cn('rounded-full px-2.5 py-1 text-[10px] font-medium transition-colors', selectedFormat === f.key ? 'bg-primary/15 text-primary' : 'bg-muted/50 text-muted-foreground hover:bg-muted/80')}
                title={f.description}
              >
                {f.label}
              </button>
            ))}
          </div>
          {selectedFormat && (
            <p className="text-[10px] text-muted-foreground/60">
              {formats.find(f => f.key === selectedFormat)?.description}
            </p>
          )}
          <Button
            size="sm"
            className="h-7 text-[11px]"
            onClick={handleExportModel}
            disabled={exporting}
          >
            {exporting ? (
              <span className="inline-flex items-center gap-1.5">
                <Spinner size="xs" />
                Exporting...
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5">
                <IconDownload className="h-3 w-3" />
                Export Model
              </span>
            )}
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2 pt-2.5 px-2.5">
          <CardTitle className="text-[11px] font-medium">Training Data Export</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 px-2.5 pb-2.5">
          <p className="text-[10px] text-muted-foreground/60">
            Download your training pairs as JSON for use in other tools.
          </p>
          <div className="flex flex-wrap gap-1">
            <Button
              size="sm"
              variant="outline"
              className="h-7 text-[11px]"
              onClick={handleExportTrainingData}
              disabled={exportingPairs}
            >
              {exportingPairs ? (
                <span className="inline-flex items-center gap-1.5">
                  <Spinner size="xs" />
                  Exporting...
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5">
                  <IconDownload className="h-3 w-3" />
                  Download Training Pairs (JSONL)
                </span>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between pb-2 pt-2.5 px-2.5">
          <CardTitle className="text-[11px] font-medium">Checkpoints</CardTitle>
          <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={fetchCheckpoints} disabled={loadingCheckpoints} aria-label="Refresh checkpoints">
            <Spinner className="h-3 w-3" />
          </Button>
        </CardHeader>
        <CardContent className="px-2.5 pb-2.5">
          {checkpoints.length === 0 ? (
            <div className="text-center py-4 space-y-1.5">
              <p className="text-[10px] text-muted-foreground/60">No checkpoints found. Train a model to create checkpoints.</p>
              <Button size="sm" variant="outline" className="h-6 text-[10px]" onClick={() => router.push('/training')}>
                Go to Training
              </Button>
            </div>
          ) : (
            <div className="space-y-1">
              {checkpoints.map(cp => (
                <div
                  key={cp.name}
                  className="flex items-center justify-between rounded-lg border border-border/40 px-2.5 py-2 text-[11px] hover:bg-muted/20 transition-colors"
                >
                  <div className="min-w-0">
                    <div className="font-medium truncate">{cp.name}</div>
                    <div className="text-[10px] text-muted-foreground/60">
                      {fmtBytes(cp.size_bytes)}
                      {cp.loss != null && <> · Loss: {cp.loss.toFixed(3)}</>}
                      {cp.created_at && <> · {new Date(cp.created_at).toLocaleDateString()}</>}
                    </div>
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 text-[10px]"
                    onClick={() => handleDownloadCheckpoint(cp.name)}
                  >
                    <IconDownload className="h-3 w-3" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </PageContainer>
  )
}
