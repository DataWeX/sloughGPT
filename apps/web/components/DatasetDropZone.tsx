'use client'

import { useState, useCallback, useRef } from 'react'
import { cn, Button } from '@sloughgpt/strui'
import { IconUpload } from '@sloughgpt/strui'
import { filesController } from '@/lib/files-controller'
import { useToastStore } from '@/lib/toast-store'

interface DatasetDropZoneProps {
  onUploadComplete: () => void
  className?: string
}

export function DatasetDropZone({ onUploadComplete, className }: DatasetDropZoneProps) {
  const addToast = useToastStore(s => s.addToast)
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFiles = useCallback(async (files: FileList | File[]) => {
    const fileArray = Array.from(files)
    if (fileArray.length === 0) return

    setUploading(true)
    setProgress(`Uploading ${fileArray.length} file(s)...`)

    try {
      for (const file of fileArray) {
        const formData = new FormData()
        formData.append('file', file)
        setProgress(`Uploading ${file.name}...`)
        await filesController.upload(formData)
      }

      addToast(`Uploaded ${fileArray.length} file(s)`, 'success')
      onUploadComplete()
    } catch {
      addToast('Upload failed', 'error')
    } finally {
      setUploading(false)
      setProgress(null)
    }
  }, [addToast, onUploadComplete])

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.currentTarget === e.target) {
      setIsDragging(false)
    }
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFiles(files)
    }
  }, [handleFiles])

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(e.target.files)
      e.target.value = ''
    }
  }

  return (
    <div
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onClick={handleClick}
      className={cn(
        'relative border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors',
        isDragging
          ? 'border-primary bg-primary/5'
          : 'border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/50',
        uploading && 'pointer-events-none opacity-60',
        className
      )}
    >
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".jsonl,.json,.csv,.txt,.md,.pdf,.parquet"
        onChange={handleFileChange}
        className="sr-only"
        aria-label="Upload dataset files"
      />

      <div className="flex flex-col items-center gap-2">
        <IconUpload className={cn('h-6 w-6', isDragging ? 'text-primary' : 'text-muted-foreground')} />
        {uploading ? (
          <div className="space-y-1">
            <p className="text-sm font-medium">{progress}</p>
          </div>
        ) : (
          <div className="space-y-1">
            <p className="text-sm font-medium">
              {isDragging ? 'Drop files here' : 'Drag & drop files here'}
            </p>
            <p className="text-xs text-muted-foreground">
              or click to browse · .jsonl, .json, .csv, .txt, .md, .pdf
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
