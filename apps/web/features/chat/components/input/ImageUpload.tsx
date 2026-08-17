'use client'

import { useCallback, useRef } from 'react'

import { IconX } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'

interface ImageUploadProps {
  onImage: (dataUrl: string) => void
  disabled?: boolean
}

function ImageIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
  )
}

export interface ImageAttachment {
  id: string
  dataUrl: string
  name: string
}

interface ImagePreviewProps {
  image: ImageAttachment
  onRemove: (id: string) => void
}

export function ImagePreview({ image, onRemove }: ImagePreviewProps) {
  return (
    <div className="relative group">
      <img
        src={image.dataUrl}
        alt={image.name}
        className="h-16 w-16 rounded-lg object-cover border border-border"
      />
      <Button
        variant="ghost"
        size="icon"
        onClick={() => onRemove(image.id)}
        className="absolute -top-1.5 -right-1.5 h-7 w-7 rounded-full bg-destructive/80 text-destructive-foreground opacity-100 transition-opacity hover:bg-destructive sm:opacity-0 sm:group-hover:opacity-100 focus-within:opacity-100"
        aria-label={`Remove ${image.name}`}
      >
            <IconX className="h-3 w-3" />
      </Button>
    </div>
  )
}

export function resizeImage(file: File, maxDim: number): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      let { width, height } = img
      if (width <= maxDim && height <= maxDim) {
        const reader = new FileReader()
        reader.onload = () => resolve(reader.result as string)
        reader.onerror = reject
        reader.readAsDataURL(file)
        return
      }
      const ratio = Math.min(maxDim / width, maxDim / height)
      width = Math.round(width * ratio)
      height = Math.round(height * ratio)
      const canvas = document.createElement('canvas')
      canvas.width = width
      canvas.height = height
      const ctx = canvas.getContext('2d')
      if (!ctx) { reject(new Error('Canvas context not available')); return }
      ctx.drawImage(img, 0, 0, width, height)
      resolve(canvas.toDataURL('image/jpeg', 0.85))
    }
    img.onerror = reject
    img.src = URL.createObjectURL(file)
  })
}

export function ImageUpload({ onImage, disabled }: ImageUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.type.startsWith('image/')) return

    try {
      const dataUrl = await resizeImage(file, 512)
      onImage(dataUrl)
    } catch {
      // fallback: send original
      const reader = new FileReader()
      reader.onload = (event) => {
        const dataUrl = event.target?.result as string
        onImage(dataUrl)
      }
      reader.readAsDataURL(file)
    }

    e.target.value = ''
  }, [onImage])

  return (
    <div className="relative">
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        disabled={disabled}
        tabIndex={-1}
        className="absolute inset-0 z-10 h-full w-full cursor-pointer opacity-0"
        aria-label="Upload image"
      />
      <Button
        variant="ghost"
        size="icon"
        disabled={disabled}
        className="h-10 w-10"
        aria-label="Upload image"
        title="Upload image"
      >
        <ImageIcon className="h-5 w-5" />
      </Button>
    </div>
  )
}
