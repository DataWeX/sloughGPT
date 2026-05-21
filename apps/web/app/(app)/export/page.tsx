'use client'

import { useState, useEffect, useCallback } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Spinner } from '@/components/ui'
import { modelController } from '@/lib/model-controller'
import { apiGet, apiPost } from '@/lib/http-client'

type ExportFormat = 'sou' | 'pytorch' | 'onnx' | 'gguf'

export default function ExportPage() {
  const [formats, setFormats] = useState<string[]>([])
  const [selectedFormat, setSelectedFormat] = useState<string>('sou')
  const [outputPath, setOutputPath] = useState('models/exported')
  const [includeTokenizer, setIncludeTokenizer] = useState(true)
  const [exporting, setExporting] = useState(false)
  const [exportResult, setExportResult] = useState<string | null>(null)
  const [exportError, setExportError] = useState<string | null>(null)
  const [health, setHealth] = useState<any>(null)
  const [pageLoaded, setPageLoaded] = useState(false)
  const [textExportCount, setTextExportCount] = useState(100)
  const [textExportResult, setTextExportResult] = useState<string | null>(null)
  const [textExportBusy, setTextExportBusy] = useState(false)

  const fetchFormats = useCallback(async () => {
    try {
      const res = await apiGet<{ formats: Record<string, string> }>('/models/export/formats')
      setFormats(Object.keys(res.formats))
    } catch { setFormats(['sou']) }
  }, [])

  const fetchHealth = useCallback(async () => {
    try {
      const h = await modelController.getHealth()
      setHealth(h)
    } catch {
      setHealth(null)
    }
  }, [])

  useEffect(() => {
    Promise.all([fetchFormats(), fetchHealth()])
      .then(() => setPageLoaded(true))
      .catch(() => setPageLoaded(true))
  }, [fetchFormats, fetchHealth])

  const handleExport = async () => {
    setExporting(true)
    setExportResult(null)
    setExportError(null)
    try {
      const res = await apiPost<{ status: string; format: string; files?: string[]; error?: string }>('/models/export', {
        output_path: outputPath,
        format: selectedFormat,
        include_tokenizer: includeTokenizer,
      })
      if (res.error) {
        setExportError(res.error)
      } else {
        setExportResult(`Exported as ${res.format}: ${res.files?.join(', ') || 'done'}`)
      }
    } catch (e: any) {
      setExportError(e.message || 'Export failed')
    }
    setExporting(false)
  }

  const handleTextExport = async () => {
    setTextExportBusy(true)
    setTextExportResult(null)
    try {
      const res = await apiPost<{ status: string; pairs_count: number; filepath: string }>('/training/export-text', {
        min_quality: 0,
        target_count: textExportCount,
      })
      setTextExportResult(`${res.pairs_count} pairs exported to ${res.filepath}`)
    } catch (e: any) {
      setTextExportResult(`Error: ${e.message}`)
    }
    setTextExportBusy(false)
  }

  if (!pageLoaded) {
    return (
      <div className="sl-page mx-auto max-w-4xl">
        <AppRouteHeader left={<AppRouteHeaderLead title="Export" />} />
        <div className="space-y-4 animate-pulse">
          <div className="h-48 rounded-lg bg-muted" />
          <div className="h-32 rounded-lg bg-muted" />
        </div>
      </div>
    )
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Export" />} />
      <div className="space-y-4">
        {/* Model Export */}
        <Card>
          <CardHeader><CardTitle className="text-base">Model Export</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <p className="text-xs text-muted-foreground">
              {health?.model_loaded
                ? `Exporting ${health.model_type || 'loaded model'}`
                : 'No model loaded — model export requires a loaded model'}
            </p>
            <div className="flex flex-wrap gap-2">
              {formats.map(f => (
                <Button
                  key={f}
                  variant={selectedFormat === f ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setSelectedFormat(f)}
                >
                  {f.toUpperCase()}
                </Button>
              ))}
            </div>
            <div className="flex items-center gap-2">
              <Input
                value={outputPath}
                onChange={e => setOutputPath(e.target.value)}
                placeholder="Output path"
                className="text-sm flex-1"
              />
              <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeTokenizer}
                  onChange={e => setIncludeTokenizer(e.target.checked)}
                />
                Include tokenizer
              </label>
            </div>
            <Button onClick={handleExport} disabled={exporting || !health?.model_loaded}>
              {exporting ? <><Spinner className="w-3 h-3 mr-1" /> Exporting...</> : 'Export'}
            </Button>
            {exportResult && <p className="text-xs text-success">{exportResult}</p>}
            {exportError && <p className="text-xs text-destructive">{exportError}</p>}
          </CardContent>
        </Card>

        {/* Training Data Export */}
        <Card>
          <CardHeader><CardTitle className="text-base">Training Data Export</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <p className="text-xs text-muted-foreground">
              Export feedback pairs as training data for fine-tuning.
            </p>
            <div className="flex items-center gap-2">
              <Input
                type="number"
                value={textExportCount}
                onChange={e => setTextExportCount(Math.max(1, parseInt(e.target.value) || 100))}
                className="text-sm w-24"
                min={1}
              />
              <span className="text-xs text-muted-foreground">pairs</span>
              <Button size="sm" onClick={handleTextExport} disabled={textExportBusy}>
                {textExportBusy ? 'Exporting...' : 'Export'}
              </Button>
            </div>
            {textExportResult && <p className="text-xs text-muted-foreground">{textExportResult}</p>}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
