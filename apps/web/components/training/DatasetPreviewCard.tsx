'use client'

import { useCallback, useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import { datasetController } from '@/lib/controllers'
import type { DatasetPreview } from '@/lib/dataset-controller'

interface DatasetPreviewCardProps {
  datasetId: string | null
}

export function DatasetPreviewCard({ datasetId }: DatasetPreviewCardProps) {
  const [preview, setPreview] = useState<DatasetPreview | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchPreview = useCallback(async () => {
    if (!datasetId) { setPreview(null); return }
    setLoading(true)
    setError(null)
    try {
      const p = await datasetController.preview(datasetId, 5)
      setPreview(p)
    } catch {
      setError('Could not load preview')
    } finally {
      setLoading(false)
    }
  }, [datasetId])

  useEffect(() => { void fetchPreview() }, [fetchPreview])

  if (!datasetId) return null

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Dataset preview</CardTitle>
        </CardHeader>
        <CardContent><Skeleton className="h-24 w-full" /></CardContent>
      </Card>
    )
  }

  if (error || !preview) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Dataset preview</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground/50">{error || 'No preview available'}</p>
        </CardContent>
      </Card>
    )
  }

  const avgLineLen = preview.total_samples > 0
    ? Math.round(preview.total_chars / preview.total_samples)
    : 0
  const languages = Object.entries(preview.languages || {})
  const uniqueChars = new Set(preview.samples.flatMap(s => s.content)).size

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Dataset preview</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground/70">
          <span>{preview.total_samples.toLocaleString()} samples</span>
          <span>{(preview.total_chars / 1000).toFixed(1)}K chars</span>
          <span>~{avgLineLen} chars/sample</span>
          <span>{uniqueChars} unique chars</span>
          {languages.length > 0 && (
            <span>{languages.map(([lang, count]) => `${lang}(${count})`).join(', ')}</span>
          )}
        </div>

        {preview.samples.length > 0 && (
          <div className="space-y-1.5">
            <p className="text-[10px] text-muted-foreground/50 uppercase tracking-wider">Samples</p>
            <div className="space-y-1 max-h-40 overflow-y-auto">
              {preview.samples.slice(0, 3).map((sample, i) => (
                <div key={i} className="rounded border border-border/30 p-2 text-[10px] font-mono text-muted-foreground/70 whitespace-pre-wrap break-all line-clamp-3">
                  {sample.content.slice(0, 200)}
                  {sample.content.length > 200 && '...'}
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
