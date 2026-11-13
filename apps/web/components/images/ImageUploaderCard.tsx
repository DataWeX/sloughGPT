'use client'

import { useState, useRef, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

interface UploadedImage {
  id: string
  name: string
  size: number
  type: string
  dataUrl: string
  timestamp: number
}

const STORAGE_KEY = 'sloughgpt-image-uploads'

function loadUploads(): UploadedImage[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch { return [] }
}

function saveUploads(uploads: UploadedImage[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(uploads))
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)}KB`
  return `${(bytes / 1048576).toFixed(1)}MB`
}

interface ImageUploaderCardProps {
  onUpload?: (image: UploadedImage) => void
}

export function ImageUploaderCard({ onUpload }: ImageUploaderCardProps) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)
  const [uploads, setUploads] = useState<UploadedImage[]>(() => loadUploads())

  const processFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) return
    const reader = new FileReader()
    reader.onload = () => {
      const img: UploadedImage = {
        id: `img-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        name: file.name,
        size: file.size,
        type: file.type,
        dataUrl: reader.result as string,
        timestamp: Date.now(),
      }
      const updated = [img, ...uploads].slice(0, 50)
      setUploads(updated)
      saveUploads(updated)
      onUpload?.(img)
    }
    reader.readAsDataURL(file)
  }, [uploads, onUpload])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    for (const file of Array.from(e.dataTransfer.files)) {
      processFile(file)
    }
  }, [processFile])

  const handleFiles = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    for (const file of Array.from(e.target.files ?? [])) {
      processFile(file)
    }
  }, [processFile])

  const handleDelete = (id: string) => {
    const updated = uploads.filter(u => u.id !== id)
    setUploads(updated)
    saveUploads(updated)
  }

  return (
    <Card data-testid="image-uploader">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Image Upload</CardTitle>
          <Button size="sm" onClick={() => fileRef.current?.click()}>
            + Upload
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={handleFiles}
        />
        <div
          className={cn(
            'border-2 border-dashed rounded-lg p-6 text-center transition-colors',
            dragOver ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'
          )}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <div className="text-sm text-muted-foreground">
            Drag & drop images or <button className="text-primary underline" onClick={() => fileRef.current?.click()}>browse</button>
          </div>
          <div className="text-[10px] text-muted-foreground mt-1">PNG, JPG, WebP up to 10MB</div>
        </div>
        {uploads.length > 0 && (
          <div className="mt-4 space-y-2">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">
              {uploads.length} upload{uploads.length !== 1 ? 's' : ''}
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {uploads.slice(0, 12).map((img) => (
                <div key={img.id} className="group relative">
                  <img
                    src={img.dataUrl}
                    alt={img.name}
                    className="w-full aspect-square object-cover rounded border border-border"
                  />
                  <button
                    className="absolute top-0.5 right-0.5 bg-background/80 text-foreground text-[9px] px-1 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={() => handleDelete(img.id)}
                  >
                    ✕
                  </button>
                  <div className="text-[9px] text-muted-foreground truncate mt-0.5">{img.name}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
