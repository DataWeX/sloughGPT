'use client'

import { useState, useCallback, useEffect, type ReactNode } from 'react'
import { cn, IconUpload } from '@sloughgpt/strui'

const TEXT_EXTENSIONS = new Set([
  'txt', 'md', 'json', 'js', 'jsx', 'ts', 'tsx', 'py', 'rb', 'go', 'rs',
  'java', 'c', 'cpp', 'h', 'hpp', 'css', 'scss', 'html', 'xml', 'yaml',
  'yml', 'toml', 'ini', 'cfg', 'conf', 'sh', 'bash', 'zsh', 'fish',
  'sql', 'csv', 'log', 'env', 'gitignore', 'dockerfile', 'makefile',
  'csv', 'tsv', 'rtf', 'tex', 'bib', 'r', 'R', 'swift', 'kt', 'scala',
  'php', 'pl', 'lua', 'vim', 'el', 'clj', 'hs', 'ml', 'fs', 'ex', 'exs',
])

function isTextFile(file: File): boolean {
  if (file.type.startsWith('text/')) return true
  if (file.type === 'application/json' || file.type === 'application/xml') return true
  const ext = file.name.split('.').pop()?.toLowerCase()
  return ext ? TEXT_EXTENSIONS.has(ext) : false
}

function isPDF(file: File): boolean {
  return file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
}

interface FileDropZoneProps {
  onImageDropped: (file: File) => void
  onTextDropped?: (content: string, filename: string) => void
  onPDFDropped?: (file: File) => void
  children: ReactNode
}

export function ImageDropZone({ onImageDropped, onTextDropped, onPDFDropped, children }: FileDropZoneProps) {
  const [dragOver, setDragOver] = useState(false)

  useEffect(() => {
    if (!dragOver) return
    const timer = setTimeout(() => setDragOver(false), 3000)
    return () => clearTimeout(timer)
  }, [dragOver])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.dataTransfer.types.includes('Files')) {
      setDragOver(true)
    }
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.currentTarget === e.target || !e.currentTarget.contains(e.relatedTarget as Node)) {
      setDragOver(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)

    const files = Array.from(e.dataTransfer.files)
    if (files.length === 0) return

    for (const file of files) {
      if (file.type.startsWith('image/')) {
        onImageDropped(file)
      } else if (isPDF(file) && onPDFDropped) {
        onPDFDropped(file)
      } else if (isTextFile(file) && onTextDropped) {
        const reader = new FileReader()
        reader.onload = () => {
          const content = reader.result as string
          const MAX_TEXT_LENGTH = 50000
          if (content.length > MAX_TEXT_LENGTH) {
            onTextDropped(content.slice(0, MAX_TEXT_LENGTH), file.name)
          } else {
            onTextDropped(content, file.name)
          }
        }
        reader.readAsText(file)
      } else {
        onImageDropped(file)
      }
    }
  }, [onImageDropped, onTextDropped, onPDFDropped])

  return (
    <div
      className="relative flex-1 flex flex-col min-h-0"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {children}

      {dragOver && (
        <div className="absolute inset-0 z-50 flex items-center justify-center pointer-events-none">
          <div className={cn(
            'absolute inset-0 bg-primary/5 backdrop-blur-[1px]',
            'transition-all duration-200',
          )} />
          <div className={cn(
            'relative flex flex-col items-center gap-2 px-6 py-4 rounded-xl',
            'bg-background/90 border-2 border-dashed border-primary/50 shadow-lg',
            'animate-in fade-in zoom-in-95 duration-200',
          )}>
            <IconUpload className="h-8 w-8 text-primary/70" />
            <span className="text-sm font-medium">Drop files to attach</span>
            <span className="text-xs text-muted-foreground">
              Images, PDFs, text, and code files supported
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
