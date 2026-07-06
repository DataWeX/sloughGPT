'use client'

import { useRef, useState } from 'react'
import { VoiceInput } from './VoiceInput'
import { ImageUpload } from './ImageUpload'
import { PDFUpload } from './PDFUpload'
import { Button } from '@sloughgpt/strui'
import { IconUpload } from '@sloughgpt/strui'
import { multimodalController } from '@/lib/multimodal-controller'

interface ChatInputAccessoriesProps {
  onImage: (dataUrl: string) => void
  onTranscript: (text: string) => void
  disabled: boolean
  onAudioTranscript?: (text: string) => void
  onGeneratedImage?: (dataUrl: string, prompt: string) => void
  onPDFAnalysis?: (analysis: string, filename: string) => void
  onPDFError?: (error: string) => void
}

export function ChatInputAccessories({
  onImage,
  onTranscript,
  disabled,
  onAudioTranscript,
  onGeneratedImage,
  onPDFAnalysis,
  onPDFError,
}: ChatInputAccessoriesProps) {
  const audioInputRef = useRef<HTMLInputElement>(null)
  const [audioLoading, setAudioLoading] = useState(false)

  const handleAudioUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setAudioLoading(true)
    try {
      const result = await multimodalController.transcribeAudio(file)
      if (result.text && onAudioTranscript) onAudioTranscript(result.text)
    } catch {
      /* silent */
    } finally {
      setAudioLoading(false)
      if (audioInputRef.current) audioInputRef.current.value = ''
    }
  }

  return (
    <div className="flex items-center">
      <ImageUpload onImage={onImage} disabled={disabled} />
      <VoiceInput onTranscript={onTranscript} disabled={disabled} />
      {onPDFAnalysis && onPDFError && (
        <PDFUpload onAnalysis={onPDFAnalysis} onError={onPDFError} disabled={disabled} />
      )}
      <Button
        variant="ghost"
        size="icon"
        className="h-10 w-10 text-muted-foreground"
        disabled={disabled || audioLoading}
        onClick={() => audioInputRef.current?.click()}
        aria-label="Upload audio"
        title="Upload audio file"
      >
        <IconUpload className="h-4 w-4" />
      </Button>
      <input
        ref={audioInputRef}
        type="file"
        accept="audio/*"
        className="hidden"
        onChange={handleAudioUpload}
      />
    </div>
  )
}
