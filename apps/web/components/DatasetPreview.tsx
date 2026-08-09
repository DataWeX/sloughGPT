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

    const fetchData = async () => {
      setLoading(true)
      setError(null)
      try {
        const [previewData] = await Promise.all([
          datasetController.preview(datasetId),
        ])
        setPreview(previewData)
        setValidation(null)
      } catch (err) {
        setError(extractErrorMessage(err, 'Failed to load preview'))
      } finally {
        setLoading(false)
      }
    }

    fetchData()
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
          {error || 'Failed to load preview'}
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
          <CardTitle className="text-base">Preview: {datasetId}</CardTitle>
          <div className="flex items-center gap-2">
            {validation && (
              <Badge variant={validation.valid ? 'default' : 'destructive'}>
                {validation.valid ? 'Valid' : 'Invalid'}
              </Badge>
            )}
            <Badge variant="secondary">{totalFiles} files</Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div>
            <div className="text-base font-semibold">{preview.total_samples}</div>
            <div className="text-xs text-muted-foreground">Samples</div>
          </div>
          <div>
            <div className="text-base font-semibold">
              {(preview.total_chars / 1024).toFixed(1)}K
            </div>
            <div className="text-xs text-muted-foreground">Characters</div>
          </div>
          <div>
            <div className="text-base font-semibold">{languageEntries.length}</div>
            <div className="text-xs text-muted-foreground">Languages</div>
          </div>
          <div>
            <div className="text-base font-semibold">
              {languageEntries[0]?.[0] || '—'}
            </div>
            <div className="text-xs text-muted-foreground">Top Language</div>
          </div>
        </div>

        {validation && validation.warnings.length > 0 && (
          <div className="rounded-md bg-warning/10 p-3">
            <div className="text-sm font-medium text-warning">Warnings</div>
            <ul className="mt-1 space-y-1 text-xs text-warning/80">
              {validation.warnings.map((warning, i) => (
                <li key={i}>• {warning}</li>
              ))}
            </ul>
          </div>
        )}

        {languageEntries.length > 1 && (
          <figure className="space-y-1">
            <figcaption className="text-xs font-medium text-muted-foreground">Language Distribution</figcaption>
            <div
              className="flex h-2 overflow-hidden rounded-full bg-muted"
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
                <Badge key={lang} variant="secondary" className="text-xs">
                  {lang}: {count}
                </Badge>
              ))}
            </div>
          </figure>
        )}

        <Tabs defaultValue="samples" className="w-full">
          <TabsList className="w-full">
            <TabsTrigger value="samples" className="flex-1">Samples</TabsTrigger>
            <TabsTrigger value="content" className="flex-1">Content</TabsTrigger>
          </TabsList>

          <TabsContent value="samples" className="mt-2" tabIndex={0}>
            <ul className="max-h-64 space-y-2 overflow-y-auto">
              {preview.samples.map((sample, i) => (
                <li
                  key={i}
                  className="rounded-md border bg-muted/30 p-2"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs text-muted-foreground">
                      {sample.path || `sample_${i}`}
                    </span>
                    <Badge variant="outline" className="text-xs">
                      {sample.language}
                    </Badge>
                  </div>
                  <pre className="mt-1 whitespace-pre-wrap font-mono text-xs" aria-label={`Sample ${i + 1} content preview`}>
                    {sample.content.slice(0, 200)}
                    {sample.content.length > 200 && '...'}
                  </pre>
                </li>
              ))}
            </ul>
          </TabsContent>

          <TabsContent value="content" className="mt-2" tabIndex={0}>
            <label htmlFor="dataset-content" className="sr-only">Dataset content</label>
            <textarea
              id="dataset-content"
              readOnly
              value={preview.samples
                .map((s) => `// ${s.path}\n${s.content}`)
                .join('\n\n')
                .slice(0, 2000)}
              className="h-64 font-mono text-xs"
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
            >
              Use for Training
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
