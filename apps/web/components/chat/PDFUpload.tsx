'use client'

import { useRef, useState, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { IconDocument } from '@/components/ui/icons'
import { PUBLIC_API_URL } from '@/lib/config'

interface PDFUploadProps {
  onAnalysis: (analysis: string, filename: string) => void
  onError: (error: string) => void
  disabled?: boolean
}

export function PDFUpload({ onAnalysis, onError, disabled }: PDFUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [filename, setFilename] = useState<string | null>(null)

  const handleFile = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      onError('Only PDF files are accepted')
      return
    }

    setUploading(true)
    setFilename(file.name)

    try {
      const form = new FormData()
      form.append('file', file)
      form.append('question', 'Analyze this document and summarize its contents.')
      form.append('per_page', 'false')
      form.append('max_new_tokens', '512')

      const res = await fetch(`${PUBLIC_API_URL}/vlm/pdf/upload`, {
        method: 'POST',
        body: form,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Upload failed' }))
        throw new Error(err.detail || `HTTP ${res.status}`)
      }
      const data = await res.json()
      onAnalysis(data.analysis || JSON.stringify(data), file.name)
    } catch (err: any) {
      onError(err.message || 'Failed to analyze PDF')
    } finally {
      setUploading(false)
      setFilename(null)
      e.target.value = ''
    }
  }, [onAnalysis, onError])

  return (
    <div className="relative">
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        onChange={handleFile}
        className="absolute inset-0 opacity-0 w-full cursor-pointer"
        disabled={disabled || uploading}
      />
      <Button
        variant="ghost"
        size="icon"
        onClick={() => inputRef.current?.click()}
        disabled={disabled || uploading}
        title="Upload PDF for analysis"
        className="relative"
      >
        {uploading ? (
          <span className="h-5 w-5 animate-spin rounded-full border-2 border-muted-foreground border-t-primary" />
        ) : (
          <IconDocument className="h-5 w-5" />
        )}
      </Button>
      {filename && !uploading && (
        <span className="text-xs text-muted-foreground ml-1">{filename}</span>
      )}
    </div>
  )
}
