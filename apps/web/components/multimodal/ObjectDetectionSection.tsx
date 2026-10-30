'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button } from '@sloughgpt/strui'

export interface DetectedObject {
  label: string
  bbox: number[]
  confidence: number
}

interface ObjectDetectionSectionProps {
  detecting: boolean
  onDetect: (file: File) => void
  objects?: DetectedObject[] | null
}

export default function ObjectDetectionSection({
  detecting,
  onDetect,
  objects,
}: ObjectDetectionSectionProps) {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)

  useEffect(() => {
    if (!file || typeof URL.createObjectURL !== 'function') {
      setPreviewUrl(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const handleDetect = () => {
    if (file) onDetect(file)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Object Detection</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center gap-2">
          <input
            type="file"
            accept="image/*"
            aria-label="Object detection image"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="text-xs"
          />
          <Button
            size="sm"
            className="h-8 text-xs shrink-0"
            onClick={handleDetect}
            disabled={detecting || !file}
          >
            {detecting ? 'Detecting…' : 'Detect'}
          </Button>
        </div>

        {previewUrl && (objects?.length ?? 0) > 0 && (
          <div
            className="relative rounded-lg border border-border/50 overflow-hidden"
            data-testid="detection-preview"
          >
            <img
              src={previewUrl}
              alt="Detection target"
              className="w-full h-auto max-h-64 object-contain bg-muted/20"
            />
            {objects!.map((obj, i) => {
              const [x1, y1, x2, y2] = obj.bbox
              return (
                <div
                  key={`${obj.label}-${i}`}
                  data-testid="detection-box"
                  className="absolute border-2 border-violet-400/90 bg-violet-400/10"
                  style={{
                    left: `${x1 * 100}%`,
                    top: `${y1 * 100}%`,
                    width: `${(x2 - x1) * 100}%`,
                    height: `${(y2 - y1) * 100}%`,
                  }}
                >
                  <span className="absolute -top-4 left-0 text-[9px] leading-none px-1 rounded bg-violet-500 text-white whitespace-nowrap">
                    {obj.label} {(obj.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              )
            })}
          </div>
        )}

        {(objects?.length ?? 0) > 0 && (
          <ul className="space-y-1" data-testid="detection-list">
            {objects!.map((obj, i) => (
              <li
                key={`${obj.label}-${i}`}
                className="flex items-center justify-between text-xs rounded border border-border/50 px-2 py-1"
              >
                <span>{obj.label}</span>
                <span className="text-muted-foreground">
                  {(obj.confidence * 100).toFixed(0)}% · [
                  {obj.bbox.map((n) => n.toFixed(2)).join(', ')}]
                </span>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  )
}
