'use client'

import { useState, useCallback, useEffect, type ReactNode } from 'react'
import { cn } from '@/lib/cn'

interface ImageDropZoneProps {
  onImageDropped: (file: File) => void
  children: ReactNode
}

export function ImageDropZone({ onImageDropped, children }: ImageDropZoneProps) {
  const [dragOver, setDragOver] = useState(false)
  const [imageCount, setImageCount] = useState(0)

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

    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'))
    if (files.length === 0) return
    setImageCount(files.length)
    files.forEach(f => onImageDropped(f))
  }, [onImageDropped])

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
            <svg className="h-8 w-8 text-primary/70" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <span className="text-sm font-medium">Drop image to attach</span>
            <span className="text-xs text-muted-foreground">Release to add image to your message</span>
          </div>
        </div>
      )}
    </div>
  )
}
