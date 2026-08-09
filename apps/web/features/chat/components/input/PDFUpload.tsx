'use client'

import { useRef, useState, useCallback } from 'react'
import { Button } from '@sloughgpt/strui'
import { IconDocument } from '@sloughgpt/strui'
import { multimodalController } from '@/lib/multimodal-controller'
import { extractErrorMessage } from '@/lib/error-utils'
import { PDF_ANALYSIS_MAX_TOKENS } from '@/lib/format-bytes'

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
      const data = await multimodalController.uploadPDF(file, 'Analyze this document and summarize its contents.', {
        perPage: false,
        maxNewTokens: PDF_ANALYSIS_MAX_TOKENS,
      })
      onAnalysis(data.analysis || JSON.stringify(data), file.name)
    } catch (err: unknown) {
      onError(extractErrorMessage(err, 'Failed to analyze PDF'))
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
        aria-label="Upload PDF for analysis"
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
