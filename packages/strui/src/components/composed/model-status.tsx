'use client'

import { useState } from 'react'
import { cn } from '../../lib/cn'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '../ui/dialog'

export type ModelStatus = 'loading' | 'loaded' | 'offline' | 'no-model'

export interface ModelStatusProps {
  status: ModelStatus
  modelName?: string
  vocabSize?: number
  blockSize?: number
  numParameters?: number
  className?: string
  onClick?: () => void
  size?: 'sm' | 'md' | 'lg'
}

const STATUS_CONFIG: Record<ModelStatus, { label: string }> = {
  loading: { label: 'Loading' },
  loaded: { label: 'Ready' },
  offline: { label: 'Offline' },
  'no-model': { label: 'No Model' },
}

const STATUS_STYLES: Record<ModelStatus, string> = {
  loading: 'text-primary',
  loaded: 'text-green-500',
  offline: 'text-red-500',
  'no-model': 'text-yellow-500',
}

const DOT_STYLES: Record<ModelStatus, string> = {
  loading: 'bg-primary',
  loaded: 'bg-green-500',
  offline: 'bg-red-500',
  'no-model': 'bg-yellow-500',
}

export function ModelStatusPill({
  status,
  modelName,
  vocabSize,
  blockSize,
  numParameters,
  className,
  onClick,
  size = 'md',
}: ModelStatusProps) {
  const config = STATUS_CONFIG[status]
  const textColor = STATUS_STYLES[status]
  const dotColor = DOT_STYLES[status]
  const isLoaded = status === 'loaded'

  const sizeStyles = {
    sm: 'text-[10px] gap-1.5 px-2 py-0.5',
    md: 'text-xs gap-2 px-2.5 py-1',
    lg: 'text-sm gap-2.5 px-3 py-1.5',
  }

  const button = (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'inline-flex items-center rounded-full font-medium transition-all',
        sizeStyles[size],
        textColor,
        onClick && 'cursor-pointer hover:opacity-80'
      )}
    >
      <span className={cn('rounded-full', dotColor, size === 'sm' ? 'w-1.5 h-1.5' : size === 'md' ? 'w-2 h-2' : 'w-2.5 h-2.5', (status === 'loaded' || status === 'loading') && 'animate-pulse')} />
      <span className="font-sans">{modelName || config.label}</span>
    </button>
  )

  if (!isLoaded || !(vocabSize || blockSize || numParameters)) {
    return button
  }

  return (
    <Dialog>
      <DialogTrigger asChild>
        {button}
      </DialogTrigger>
      <DialogContent className="max-w-xs">
        <DialogHeader>
          <DialogTitle className="text-sm">Model Info</DialogTitle>
        </DialogHeader>
        <div className="space-y-2 text-xs">
          {modelName && (
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Model</span>
              <span className="font-mono truncate max-w-[160px]">{modelName}</span>
            </div>
          )}
          {vocabSize && (
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Vocab</span>
              <span className="font-mono">{vocabSize.toLocaleString()}</span>
            </div>
          )}
          {blockSize && (
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Block</span>
              <span className="font-mono">{blockSize}</span>
            </div>
          )}
          {numParameters && (
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Params</span>
              <span className="font-mono">{(numParameters / 1_000_000).toFixed(1)}M</span>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
