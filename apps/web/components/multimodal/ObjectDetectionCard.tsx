'use client'

import { useState, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'
import { IconScan } from '@sloughgpt/strui'
import { apiPost } from '@/lib/http-client'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'

interface DetectedObject {
  label: string
  bbox: number[]
  confidence: number
}

export function ObjectDetectionCard() {
  const [detecting, setDetecting] = useState(false)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [objects, setObjects] = useState<DetectedObject[]>([])
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const addToast = useToastStore(s => s.addToast)

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setDetecting(true)
    setError(null)
    setObjects([])
    const url = URL.createObjectURL(file)
    setImageUrl(url)
    try {
      const form = new FormData()
      form.append('file', file)
      const data = await apiPost<{ objects: DetectedObject[] }>('/multimodal/detect', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setObjects(data.objects ?? [])
      addToast(`Detected ${data.objects?.length ?? 0} objects`, 'success')
    } catch (err) {
      const msg = extractErrorMessage(err, 'Detection failed')
      setError(msg)
      addToast(msg, 'error')
    } finally {
      setDetecting(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Object Detection</CardTitle>
        <Button size="sm" variant="ghost" onClick={() => inputRef.current?.click()} disabled={detecting}>
          <IconScan className="h-4 w-4 mr-1" /> Scan
        </Button>
        <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
      </CardHeader>
      <CardContent className="space-y-3">
        {imageUrl && (
          <div className="relative rounded-md border border-border/60 overflow-hidden">
            <img src={imageUrl} alt="Scan target" className="w-full max-h-64 object-contain bg-muted/20" />
            {objects.length > 0 && (
              <div className="absolute inset-0 pointer-events-none">
                {objects.map((obj, i) => {
                  const [x1, y1, x2, y2] = obj.bbox
                  return (
                    <div
                      key={i}
                      className="absolute border-2 border-primary bg-primary/10"
                      style={{ left: `${x1}%`, top: `${y1}%`, width: `${x2 - x1}%`, height: `${y2 - y1}%` }}
                    >
                      <span className="absolute -top-5 left-0 text-[10px] bg-primary text-primary-foreground px-1 rounded">
                        {obj.label} {(obj.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}
        {error && <div className="text-xs text-destructive">{error}</div>}
        {!imageUrl && (
          <p className="text-sm text-muted-foreground">Upload an image to detect objects.</p>
        )}
      </CardContent>
    </Card>
  )
}
