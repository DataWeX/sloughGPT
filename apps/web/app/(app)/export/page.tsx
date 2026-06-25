'use client'

import { useCallback, useEffect, useState } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { StatCard, KpiGrid } from '@/components/ui'
import { IconDownload, IconCheck } from '@/components/ui'
import { exportController } from '@/lib/controllers'
import { modelController, type ModelInfo } from '@/lib/model-controller'
import { useToastStore } from '@/lib/toast-store'

export default function ExportPage() {
  const addToast = useToastStore(s => s.addToast)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loadedModel, setLoadedModel] = useState<ModelInfo | null>(null)
  const [formats, setFormats] = useState<string[]>(['sou', 'onnx', 'gguf'])
  const [selectedFormat, setSelectedFormat] = useState('sou')
  const [outputPath, setOutputPath] = useState('models/exported')
  const [includeTokenizer, setIncludeTokenizer] = useState(true)
  const [exporting, setExporting] = useState(false)
  const [result, setResult] = useState<{ files?: string[] } | null>(null)
  const [loading, setLoading] = useState(true)

  const fetchData = useCallback(async () => {
    try {
      const [list, health, fmts] = await Promise.all([
        modelController.list(),
        modelController.getHealth(),
        exportController.getFormats().catch(() => []),
      ])
      setModels(list)
      setFormats(fmts.length > 0 ? fmts : formats)
      const loaded = list.find(m => m.loaded || health?.model_type?.includes(m.id || m.name)) || null
      setLoadedModel(loaded)
    } catch {} finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchData() }, [fetchData])

  const handleExport = async () => {
    if (!loadedModel) { addToast('No model loaded', 'error'); return }
    setExporting(true)
    setResult(null)
    try {
      const res = await exportController.exportModel({
        format: selectedFormat,
        output_path: outputPath,
        include_tokenizer: includeTokenizer,
      })
      if (res.error) {
        addToast(`Export failed: ${res.error}`, 'error')
      } else {
        setResult(res)
        addToast(`Exported as ${selectedFormat.toUpperCase()}`, 'success')
      }
    } catch (e: any) {
      addToast(`Export error: ${e.message}`, 'error')
    }
    setExporting(false)
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Export" />} />

      <div className="space-y-4">
        {/* Stats */}
        <KpiGrid columns={3}>
          <StatCard label="Available Models" value={models.length} />
          <StatCard label="Loaded" value={loadedModel?.name || 'None'} />
          <StatCard label="Formats" value={formats.length} />
        </KpiGrid>

        {/* Model card */}
        <Card>
          <CardHeader><CardTitle className="text-base">Model</CardTitle></CardHeader>
          <CardContent>
            {loading ? (
              <div className="h-8 animate-pulse rounded bg-muted" />
            ) : loadedModel ? (
              <div className="flex items-center gap-3">
                <IconCheck className="h-5 w-5 text-success" />
                <div>
                  <p className="text-sm font-medium">{loadedModel.name}</p>
                  <p className="text-xs text-muted-foreground">{loadedModel.id}</p>
                </div>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">No model loaded. Load one from the Models page first.</p>
            )}
          </CardContent>
        </Card>

        {/* Export config */}
        <Card>
          <CardHeader><CardTitle className="text-base">Export Settings</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-1.5">Format</p>
              <div className="flex flex-wrap gap-2">
                {formats.map(f => (
                  <button
                    key={f}
                    onClick={() => setSelectedFormat(f)}
                    className={`rounded-lg border px-4 py-2 text-sm font-medium transition-colors ${
                      selectedFormat === f
                        ? 'bg-primary text-primary-foreground border-primary'
                        : 'bg-background text-muted-foreground border-border hover:border-primary/50'
                    }`}
                  >
                    {f.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <p className="text-xs font-medium text-muted-foreground mb-1">Output path (relative to repo root)</p>
              <Input value={outputPath} onChange={e => setOutputPath(e.target.value)} />
            </div>

            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={includeTokenizer}
                onChange={e => setIncludeTokenizer(e.target.checked)}
                className="rounded border-border"
              />
              <span className="text-sm">Include tokenizer</span>
            </label>

            <Button onClick={handleExport} disabled={exporting || !loadedModel}>
              <IconDownload className="h-4 w-4 mr-1" />
              {exporting ? 'Exporting...' : `Export as ${selectedFormat.toUpperCase()}`}
            </Button>
          </CardContent>
        </Card>

        {/* Result */}
        {result && result.files && result.files.length > 0 && (
          <Card>
            <CardHeader><CardTitle className="text-base">Exported Files</CardTitle></CardHeader>
            <CardContent>
              <ul className="space-y-1">
                {result.files.map((f, i) => (
                  <li key={i} className="text-sm text-muted-foreground font-mono">{f}</li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
