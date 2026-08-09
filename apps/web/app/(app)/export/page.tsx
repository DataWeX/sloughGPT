'use client'

import { useState, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Badge } from '@sloughgpt/strui'
import { IconDownload } from '@/components/icons/NavIcons'
import { IconRefresh } from '@sloughgpt/strui'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
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
  const [formats, setFormats] = useState<ExportFormat[]>([])
  const [selectedFormat, setSelectedFormat] = useState('sou')
  const [exporting, setExporting] = useState(false)
  const [exportResult, setExportResult] = useState<string | null>(null)
  const [exportError, setExportError] = useState<string | null>(null)

  const [trainingPairs, setTrainingPairs] = useState<Record<string, unknown>[] | null>(null)
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
      .catch(() => {
        setFormats([
          { key: 'sou', label: 'SOU', description: 'SloughGPT self-contained + personality' },
          { key: 'safetensors', label: 'SafeTensors', description: 'Safe, fast (recommended)' },
          { key: 'onnx', label: 'ONNX', description: 'Cross-platform (server, web, TF.js)' },
          { key: 'gguf_q4_k_m', label: 'GGUF Q4_K_M', description: 'Mobile (llama.rn)' },
          { key: 'torch', label: 'PyTorch', description: 'Training checkpoint' },
        ])
      })
  }, [])

  const fetchCheckpoints = async () => {
    setLoadingCheckpoints(true)
    try {
      const data = await apiGet<{ checkpoints: Checkpoint[] }>('/auto-train/checkpoints')
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
      recordExport(res.format, fileCount)
    } catch (err) {
      setExportError(err instanceof Error ? err.message : 'Export failed')
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
      setExportError(err instanceof Error ? err.message : 'Training data export failed')
    } finally {
      setExportingPairs(false)
    }
  }

  const handleDownloadCheckpoint = async (name: string) => {
    try {
      const blob = await trainingJobsController.downloadCheckpoint(name)
      downloadBlob(blob, `${name}.soul`)
    } catch (err) {
      setExportError(err instanceof Error ? err.message : 'Checkpoint download failed')
    }
  }

  const handleDownloadFeedbackExport = async () => {
    try {
      const blob = await trainingJobsController.downloadTrainingJob('latest')
      downloadBlob(blob, `feedback-export-${Date.now()}.json`)
    } catch (err) {
      setExportError(err instanceof Error ? err.message : 'Feedback export not available')
    }
  }

  const fmtBytes = (bytes?: number) => {
    if (!bytes) return ''
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Export" subtitle="Export models, training data, and checkpoints" />}
      />
      <div className="space-y-4">
        {exportResult && (
          <div className="rounded-md bg-success/10 border border-success/20 px-4 py-3 text-sm text-success">
            {exportResult}
          </div>
        )}
        {exportError && (
          <div className="rounded-md bg-destructive/10 border border-destructive/20 px-4 py-3 text-sm text-destructive">
            {exportError}
            <button className="ml-2 underline" onClick={() => setExportError(null)}>Dismiss</button>
          </div>
        )}

        <ExportHistoryCard />

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Model Export</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground">
              Export the currently loaded model to a file format.
            </p>
            <div className="flex flex-wrap gap-2">
              {formats.map(f => (
                <button
                  key={f.key}
                  onClick={() => setSelectedFormat(f.key)}
                  className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                    selectedFormat === f.key
                      ? 'bg-primary/15 text-primary'
                      : 'bg-muted text-muted-foreground hover:bg-muted/80'
                  }`}
                  title={f.description}
                >
                  {f.label}
                </button>
              ))}
            </div>
            {selectedFormat && (
              <p className="text-xs text-muted-foreground">
                {formats.find(f => f.key === selectedFormat)?.description}
              </p>
            )}
            <Button
              size="sm"
              onClick={handleExportModel}
              disabled={exporting}
            >
              {exporting ? (
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  Exporting...
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5">
                  <IconDownload className="h-3.5 w-3.5" />
                  Export Model
                </span>
              )}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Training Data Export</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground">
              Download your training pairs as JSON for use in other tools.
            </p>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={handleExportTrainingData}
                disabled={exportingPairs}
              >
                {exportingPairs ? (
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />
                    Exporting...
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5">
                    <IconDownload className="h-3.5 w-3.5" />
                    Download Training Pairs (JSONL)
                  </span>
                )}
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={handleDownloadFeedbackExport}
              >
                <span className="inline-flex items-center gap-1.5">
                  <IconDownload className="h-3.5 w-3.5" />
                  Download Feedback Export
                </span>
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Checkpoints</CardTitle>
            <Button size="sm" variant="ghost" onClick={fetchCheckpoints} disabled={loadingCheckpoints}>
              <IconRefresh className={`h-3.5 w-3.5 ${loadingCheckpoints ? 'animate-spin' : ''}`} />
            </Button>
          </CardHeader>
          <CardContent>
            {checkpoints.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No checkpoints found. Train a model to create checkpoints.
              </p>
            ) : (
              <div className="space-y-2">
                {checkpoints.map(cp => (
                  <div
                    key={cp.name}
                    className="flex items-center justify-between rounded-md border border-border/60 px-3 py-2 text-sm hover:bg-muted/50 transition-colors"
                  >
                    <div className="min-w-0">
                      <div className="font-medium truncate">{cp.name}</div>
                      <div className="text-xs text-muted-foreground">
                        {fmtBytes(cp.size_bytes)}
                        {cp.loss != null && <> · Loss: {cp.loss.toFixed(3)}</>}
                        {cp.created_at && <> · {new Date(cp.created_at).toLocaleDateString()}</>}
                      </div>
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleDownloadCheckpoint(cp.name)}
                    >
                      <IconDownload className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
