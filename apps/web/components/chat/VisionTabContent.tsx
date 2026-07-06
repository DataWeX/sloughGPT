'use client'

import { useState, useEffect, useCallback } from 'react'
import { Button } from '@sloughgpt/strui'
import { cn } from '@/lib/cn'
import { VisionStudioDialog } from './VisionStudioDialog'

interface VisionTabContentProps {
  visionImagesLearned?: number
  visionTrained?: boolean
  visionStatus?: string
  visionCaptionHistory?: string[]
  visionVocabSize?: number
  sessionId: string | null
  onGeneratedImage: (dataUrl: string, prompt: string) => void
  meanAccuracy?: number
  lastAccuracy?: number
  onSendText: (text: string) => void
}

export function VisionTabContent({
  visionImagesLearned,
  visionTrained,
  visionStatus,
  visionVocabSize,
  sessionId,
  onGeneratedImage,
  meanAccuracy,
}: VisionTabContentProps) {
  const [studioOpen, setStudioOpen] = useState(false)
  const [statusLabel, setStatusLabel] = useState('')

  useEffect(() => {
    if (visionTrained) setStatusLabel('Trained')
    else if ((visionImagesLearned ?? 0) > 0) setStatusLabel('Learning')
    else setStatusLabel('Ready')
  }, [visionTrained, visionImagesLearned])

  const handleSendText = useCallback((text: string) => {
    window.dispatchEvent(new CustomEvent('send-text', { detail: { text } }))
  }, [])

  return (
    <>
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium">Vision Model</span>
          <span
            className={cn(
              'inline-block h-1.5 w-1.5 rounded-full',
              visionTrained ? 'bg-success' : (visionImagesLearned ?? 0) > 0 ? 'bg-warning' : 'bg-muted-foreground/30',
            )}
          />
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div className="p-2 rounded bg-muted/30 border border-border/40">
            <div className="text-[10px] text-muted-foreground">Images learned</div>
            <div className="text-sm font-medium">{visionImagesLearned ?? 0}</div>
          </div>
          <div className="p-2 rounded bg-muted/30 border border-border/40">
            <div className="text-[10px] text-muted-foreground">Status</div>
            <div className="text-sm font-medium capitalize">{visionStatus || statusLabel || 'ready'}</div>
          </div>
        </div>

        {meanAccuracy !== undefined && meanAccuracy > 0 && (
          <div className="p-2 rounded bg-muted/30 border border-border/40 flex justify-between items-center">
            <span className="text-[10px] text-muted-foreground">Mean accuracy</span>
            <span className={cn(
              'text-sm font-medium',
              meanAccuracy >= 80 ? 'text-success' : meanAccuracy >= 40 ? 'text-warning' : 'text-muted-foreground',
            )}>
              {meanAccuracy.toFixed(1)}%
            </span>
          </div>
        )}

        {visionVocabSize !== undefined && visionVocabSize > 0 && (
          <div className="p-2 rounded bg-muted/30 border border-border/40 flex justify-between items-center">
            <span className="text-[10px] text-muted-foreground">Vocabulary</span>
            <span className="text-sm font-medium">{visionVocabSize} words</span>
          </div>
        )}

        <Button
          size="sm"
          className="w-full text-xs"
          onClick={() => setStudioOpen(true)}
        >
          Open Vision Studio
        </Button>
      </div>

      <VisionStudioDialog
        open={studioOpen}
        onOpenChange={setStudioOpen}
        sessionId={sessionId}
        onGeneratedImage={onGeneratedImage}
        onSendText={handleSendText}
        initialCaps={{
          images_learned: visionImagesLearned,
          trained: visionTrained,
          status: visionStatus,
          vocab_size: visionVocabSize,
          mean_accuracy: meanAccuracy,
        }}
      />
    </>
  )
}
