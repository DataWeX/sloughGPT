'use client'

import { useEffect, useState } from 'react'
import { datasetController, type DatasetPreview as DatasetPreviewType } from '@/lib/dataset-controller'
import { Badge } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@sloughgpt/strui'
import { extractErrorMessage } from '@/lib/error-utils'

interface DatasetValidation {
  dataset_id: string
  valid: boolean
  issues: string[]
  warnings: string[]
  stats: Record<string, unknown>
}

interface DatasetPreviewProps {
  datasetId: string
  onUseForTraining?: () => void
}

export function DatasetPreview({ datasetId, onUseForTraining }: DatasetPreviewProps) {
  const [preview, setPreview] = useState<DatasetPreviewType | null>(null)
  const [validation, setValidation] = useState<DatasetValidation | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!datasetId) return

    let active = true
    const fetchData = async () => {
      setLoading(true)
      setError(null)
      try {
        const [previewData] = await Promise.all([
          datasetController.preview(datasetId),
        ])
        if (active) {
          setPreview(previewData)
          setValidation(null)
        }
      } catch (err) {
        if (active) setError(extractErrorMessage(err, 'Could not load preview'))
      } finally {
        if (active) setLoading(false)
      }
    }

    void fetchData()
    return () => { active = false }
  }, [datasetId])

  if (loading) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground" role="status" aria-live="polite">
          Loading preview...
        </CardContent>
      </Card>
    )
  }

  if (error || !preview) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-destructive" role="alert" aria-live="assertive">
          {error || 'Could not load preview'}
        </CardContent>
      </Card>
    )
  }

  const languageEntries = Object.entries(preview.languages || {}).sort((a, b) => b[1] - a[1])
  const totalFiles = Object.values(preview.languages || {}).reduce((sum, count) => sum + count, 0)

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xs">Preview: {datasetId}</CardTitle>
          <div className="flex items-center gap-1.5">
            {validation && (
              <Badge variant={validation.valid ? 'default' : 'destructive'}>
                {validation.valid ? 'Valid' : 'Invalid'}
              </Badge>
            )}
             <Badge variant="outline" className="text-[9px]">{totalFiles} files</Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <div className="rounded-lg bg-muted/20 px-2 py-1.5">
            <div className="text-[11px] font-mono tabular-nums">{preview.total_samples}</div>
            <div className="text-[9px] text-muted-foreground/60">Samples</div>
          </div>
          <div className="rounded-lg bg-muted/20 px-2 py-1.5">
            <div className="text-[11px] font-mono tabular-nums">
              {(preview.total_chars / 1024).toFixed(1)}K
            </div>
            <div className="text-[9px] text-muted-foreground/60">Characters</div>
          </div>
          <div className="rounded-lg bg-muted/20 px-2 py-1.5">
            <div className="text-[11px] font-mono tabular-nums">{languageEntries.length}</div>
            <div className="text-[9px] text-muted-foreground/60">Languages</div>
          </div>
          <div className="rounded-lg bg-muted/20 px-2 py-1.5">
            <div className="text-[11px] font-mono tabular-nums truncate">
              {languageEntries[0]?.[0] || '—'}
            </div>
            <div className="text-[9px] text-muted-foreground/60">Top Language</div>
          </div>
        </div>

        {validation && validation.warnings.length > 0 && (
          <div className="rounded-md bg-warning/10 p-2">
            <div className="text-[11px] font-medium text-warning">Warnings</div>
            <ul className="mt-0.5 space-y-px text-[10px] text-warning/80">
              {validation.warnings.map((warning, i) => (
                <li key={i}>• {warning}</li>
              ))}
            </ul>
          </div>
        )}

        {languageEntries.length > 1 && (
          <figure className="space-y-1">
            <figcaption className="text-[10px] font-medium text-muted-foreground/60">Language Distribution</figcaption>
            <div
              className="flex h-1.5 overflow-hidden rounded-full bg-muted"
              role="img"
              aria-label={`Language distribution: ${languageEntries.slice(0, 6).map(([lang, count]) => `${lang} ${Math.round((count / totalFiles) * 100)}%`).join(', ')}`}
            >
              {languageEntries.slice(0, 6).map(([lang, count]) => (
                <div
                  key={lang}
                  className="bg-primary"
                  style={{ width: `${(count / totalFiles) * 100}%` }}
                  aria-hidden="true"
                />
              ))}
            </div>
            <div className="flex flex-wrap gap-1">
              {languageEntries.slice(0, 6).map(([lang, count]) => (
                <Badge key={lang} variant="outline" className="text-[9px]">
                  {lang}: {count}
                </Badge>
              ))}
            </div>
          </figure>
        )}

        <Tabs defaultValue="samples" className="w-full">
          <TabsList className="h-7 w-full">
            <TabsTrigger value="samples" className="flex-1 text-[10px]">Samples</TabsTrigger>
            <TabsTrigger value="content" className="flex-1 text-[10px]">Content</TabsTrigger>
          </TabsList>

          <TabsContent value="samples" className="mt-1.5" tabIndex={0}>
            <ul className="max-h-56 space-y-1 overflow-y-auto">
              {preview.samples.map((sample, i) => (
                <li
                  key={i}
                  className="rounded-md border border-border/40 bg-muted/20 p-2"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-[9px] text-muted-foreground/60">
                      {sample.path || `sample_${i}`}
                    </span>
                    <Badge variant="outline" className="text-[8px]">
                      {sample.language}
                    </Badge>
                  </div>
                  <pre className="mt-0.5 whitespace-pre-wrap font-mono text-[10px]" aria-label={`Sample ${i + 1} content preview`}>
                    {sample.content.slice(0, 200)}
                    {sample.content.length > 200 && '...'}
                  </pre>
                </li>
              ))}
            </ul>
          </TabsContent>

          <TabsContent value="content" className="mt-1.5" tabIndex={0}>
            <label htmlFor="dataset-content" className="sr-only">Dataset content</label>
            <textarea
              id="dataset-content"
              readOnly
              value={preview.samples
                .map((s) => `// ${s.path}\n${s.content}`)
                .join('\n\n')
                .slice(0, 2000)}
              className="h-56 font-mono text-[10px]"
              aria-label="Full dataset content preview"
            />
          </TabsContent>
        </Tabs>

        {onUseForTraining && (
          <div className="flex justify-end">
            <Button
              onClick={onUseForTraining}
              disabled={validation ? !validation.valid : false}
              aria-disabled={validation ? !validation.valid : false}
              aria-describedby={validation && !validation.valid ? 'validation-warning' : undefined}
              className="h-7 text-[11px]"
            >
              Use for Training
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
