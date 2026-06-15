'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { IconUpload } from '@/components/ui'
import { cn } from '@/lib/cn'

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
}

export function VisionTabContent({
  visionImagesLearned,
  visionTrained,
  visionStatus,
  visionCaptionHistory,
  visionVocabSize,
  sessionId,
  onGeneratedImage,
  meanAccuracy,
}: VisionTabContentProps) {
  const [genPrompt, setGenPrompt] = useState('')
  const [genLoading, setGenLoading] = useState(false)
  const [trainLoading, setTrainLoading] = useState(false)
  const [audioTranscript, setAudioTranscript] = useState('')
  const [audioLoading, setAudioLoading] = useState(false)
  const [genResult, setGenResult] = useState<string | null>(null)
  const [lastGenPrompt, setLastGenPrompt] = useState('')
  const [trainLabel, setTrainLabel] = useState('')
  const [trainImagePreview, setTrainImagePreview] = useState<string | null>(null)
  const [trainResult, setTrainResult] = useState<{ accuracy: number; caption: string } | null>(null)

  const handleGenerate = async () => {
    if (!genPrompt.trim() || genLoading) return
    setGenLoading(true)
    setGenResult(null)
    setLastGenPrompt(genPrompt.trim())
    try {
      const { multimodalController } = await import('@/lib/multimodal-controller')
      const result = await multimodalController.generateImage(genPrompt.trim())
      if (result?.image) {
        setGenResult(result.image)
        onGeneratedImage(result.image, genPrompt.trim())
      }
    } catch {
      // silent
    } finally {
      setGenLoading(false)
    }
  }

  const handleSendAudioTranscript = () => {
    if (!audioTranscript.trim()) return
    window.dispatchEvent(new CustomEvent('send-text', { detail: { text: audioTranscript } }))
    setAudioTranscript('')
  }

  const handleSendGeneratedImage = () => {
    if (!genResult || !lastGenPrompt) return
    onGeneratedImage(genResult, lastGenPrompt)
  }

  const handleAudioTranscribe = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setAudioLoading(true)
    try {
      const { multimodalController } = await import('@/lib/multimodal-controller')
      const result = await multimodalController.transcribeAudio(file)
      if (result?.text) setAudioTranscript(result.text)
    } catch {
      setAudioTranscript('Transcription failed')
    } finally {
      setAudioLoading(false)
    }
  }

  const handleTrainFromSession = async () => {
    if (!sessionId || trainLoading) return
    setTrainLoading(true)
    try {
      const { multimodalController } = await import('@/lib/multimodal-controller')
      await multimodalController.trainImage('', sessionId)
    } catch {
      // silent
    } finally {
      setTrainLoading(false)
    }
  }

  const handleTrainImageWithLabel = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setTrainLoading(true)
    setTrainResult(null)
    try {
      const reader = new FileReader()
      reader.onload = async () => {
        const dataUrl = reader.result as string
        setTrainImagePreview(dataUrl)
        const { multimodalController } = await import('@/lib/multimodal-controller')
        const result = await multimodalController.trainImage(dataUrl, file.name, trainLabel.trim() || undefined)
        setTrainResult({ accuracy: result.accuracy, caption: result.caption })
        window.dispatchEvent(new CustomEvent('refresh-vision'))
      }
      reader.readAsDataURL(file)
    } catch {
      setTrainResult({ accuracy: 0, caption: 'Training failed' })
    } finally {
      setTrainLoading(false)
      if (e.target) e.target.value = ''
    }
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium">Vision Model</span>
        <span className={cn(
          'inline-block h-1.5 w-1.5 rounded-full',
          visionTrained ? 'bg-success' : (visionImagesLearned || 0) > 0 ? 'bg-warning' : 'bg-muted-foreground/30',
        )} />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div className="p-2 rounded bg-muted/30 border border-border/40">
          <div className="text-[10px] text-muted-foreground">Images learned</div>
          <div className="text-sm font-medium">{visionImagesLearned ?? 0}</div>
        </div>
        <div className="p-2 rounded bg-muted/30 border border-border/40">
          <div className="text-[10px] text-muted-foreground">Status</div>
          <div className="text-sm font-medium capitalize">{visionStatus || 'ready'}</div>
        </div>
        {visionVocabSize !== undefined && (
          <div className="p-2 rounded bg-muted/30 border border-border/40">
            <div className="text-[10px] text-muted-foreground">Vocabulary</div>
            <div className="text-sm font-medium">{visionVocabSize} words</div>
          </div>
        )}
        {meanAccuracy !== undefined && meanAccuracy > 0 && (
          <div className="p-2 rounded bg-muted/30 border border-border/40">
            <div className="text-[10px] text-muted-foreground">Mean accuracy</div>
            <div className="text-sm font-medium">{meanAccuracy.toFixed(1)}%</div>
          </div>
        )}
      </div>

      {visionCaptionHistory && visionCaptionHistory.length > 0 && (
        <div className="space-y-1">
          <div className="text-[10px] text-muted-foreground font-medium">Recent captions</div>
          <ul className="space-y-1 max-h-24 overflow-y-auto">
            {visionCaptionHistory.slice(-8).map((cap, i) => (
              <li key={i} className="p-1.5 rounded bg-muted/20 text-[10px] leading-relaxed border border-border/20">{cap}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="pt-1 border-t border-border/30 space-y-1">
        <div className="text-[10px] text-muted-foreground font-medium">Train with label (supervised)</div>
        <input
          className="w-full px-2 py-1 text-xs border border-input rounded bg-background focus:outline-none focus:ring-1 focus:ring-primary/40"
          placeholder="Label (e.g., 'a red car')"
          value={trainLabel}
          onChange={e => setTrainLabel(e.target.value)}
        />
        <label className="flex items-center gap-1 px-2 py-1.5 rounded border border-border/40 bg-muted/10 cursor-pointer hover:bg-muted/20 text-[10px]">
          <IconUpload className="h-3 w-3" />
          {trainLoading ? 'Training...' : 'Upload image to train'}
          <input type="file" accept="image/*" className="hidden" onChange={handleTrainImageWithLabel} disabled={trainLoading} />
        </label>
        {trainImagePreview && (
          <img src={trainImagePreview} alt="Training preview" className="w-full rounded border border-border/40 object-cover max-h-20" />
        )}
        {trainResult && (
          <div className="p-1.5 rounded bg-muted/20 text-[10px] border border-border/20 space-y-0.5">
            <div className="flex justify-between">
              <span>Accuracy:</span>
              <span className={cn('font-medium', trainResult.accuracy >= 80 ? 'text-success' : trainResult.accuracy >= 50 ? 'text-warning' : 'text-destructive')}>
                {trainResult.accuracy.toFixed(1)}%
              </span>
            </div>
            <div className="text-muted-foreground">Caption: {trainResult.caption}</div>
          </div>
        )}
      </div>

      <Button
        size="sm"
        variant="outline"
        className="w-full text-[10px] h-7"
        disabled={trainLoading || !sessionId}
        onClick={async () => {
          await handleTrainFromSession()
          window.dispatchEvent(new CustomEvent('refresh-vision'))
        }}
      >
        {trainLoading ? 'Training...' : 'Train from session images'}
      </Button>

      <div className="pt-1 border-t border-border/30">
        <div className="text-[10px] text-muted-foreground font-medium mb-1">Generate Image</div>
        <div className="flex gap-1">
          <input
            className="flex-1 px-2 py-1 text-xs border border-input rounded bg-background focus:outline-none focus:ring-1 focus:ring-primary/40"
            placeholder="A cat in space..."
            value={genPrompt}
            onChange={e => setGenPrompt(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') handleGenerate() }}
          />
          <Button size="sm" className="h-7 text-[10px] shrink-0" disabled={genLoading || !genPrompt.trim()} onClick={handleGenerate}>
            {genLoading ? '...' : 'Go'}
          </Button>
        </div>
        {genResult && (
          <div className="mt-1 space-y-1">
            <img src={genResult} alt="Generated" className="w-full rounded border border-border/40 object-cover max-h-32" />
            <div className="flex gap-1">
              <Button size="sm" variant="outline" className="h-6 text-[10px] flex-1" onClick={handleSendGeneratedImage}>
                Send to chat
              </Button>
              <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => setGenResult(null)}>
                Dismiss
              </Button>
            </div>
          </div>
        )}
      </div>

      <div className="pt-1 border-t border-border/30">
        <div className="text-[10px] text-muted-foreground font-medium mb-1">Transcribe Audio</div>
        <label className="flex items-center gap-1 px-2 py-1.5 rounded border border-border/40 bg-muted/10 cursor-pointer hover:bg-muted/20 text-[10px]">
          <IconUpload className="h-3 w-3" />
          {audioLoading ? 'Transcribing...' : 'Upload audio file'}
          <input type="file" accept="audio/*" className="hidden" onChange={handleAudioTranscribe} disabled={audioLoading} />
        </label>
        {audioTranscript && (
          <div className="mt-1 space-y-1">
            <div className="p-1.5 rounded bg-muted/20 text-[10px] leading-relaxed border border-border/20">{audioTranscript}</div>
            <div className="flex gap-1">
              <Button size="sm" variant="outline" className="h-6 text-[10px] flex-1" onClick={handleSendAudioTranscript}>
                Send to chat
              </Button>
              <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={() => setAudioTranscript('')}>
                Dismiss
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
