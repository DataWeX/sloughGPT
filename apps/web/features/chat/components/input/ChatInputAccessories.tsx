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
  onTranscript?: (text: string) => void
  onAudioRecorded?: (blob: Blob) => void
  disabled: boolean
  onAudioTranscript?: (text: string) => void
  onGeneratedImage?: (dataUrl: string, prompt: string) => void
  onPDFAnalysis?: (analysis: string, filename: string) => void
  onPDFError?: (error: string) => void
  onCodeBlock?: () => void
  textareaRef?: React.RefObject<HTMLTextAreaElement>
  value?: string
  onChange?: (value: string) => void
}

export function ChatInputAccessories({
  onImage,
  onTranscript,
  onAudioRecorded,
  disabled,
  onAudioTranscript,
  onGeneratedImage,
  onPDFAnalysis,
  onPDFError,
  onCodeBlock,
  textareaRef,
  value,
  onChange,
}: ChatInputAccessoriesProps) {
  const audioInputRef = useRef<HTMLInputElement>(null)
  const [audioLoading, setAudioLoading] = useState(false)

  const handleCodeBlock = () => {
    if (onCodeBlock) {
      onCodeBlock()
      return
    }
    if (!textareaRef?.current || !onChange || !value) return
    const ta = textareaRef.current
    const start = ta.selectionStart
    const end = ta.selectionEnd
    const selected = value.slice(start, end)
    if (selected) {
      const wrapped = `\`\`\`\n${selected}\n\`\`\``
      const newVal = value.slice(0, start) + wrapped + value.slice(end)
      onChange(newVal)
      setTimeout(() => { ta.selectionStart = start + 4; ta.selectionEnd = start + 4 + selected.length; ta.focus() }, 0)
    } else {
      const insert = '```\n\n```'
      const newVal = value.slice(0, start) + insert + value.slice(end)
      onChange(newVal)
      setTimeout(() => { ta.selectionStart = start + 4; ta.selectionEnd = start + 4; ta.focus() }, 0)
    }
  }

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
    <div className="flex items-center shrink-0">
      <ImageUpload onImage={onImage} disabled={disabled} />
      {onTranscript && <VoiceInput onTranscript={onTranscript} onAudioRecorded={onAudioRecorded} disabled={disabled} />}
      {onPDFAnalysis && onPDFError && (
        <PDFUpload onAnalysis={onPDFAnalysis} onError={onPDFError} disabled={disabled} />
      )}
      <Button
        variant="ghost"
        size="icon"
        className="h-10 w-10 text-muted-foreground"
        disabled={disabled}
        onClick={handleCodeBlock}
        aria-label="Insert code block"
        title="Insert code block"
      >
        <span className="font-mono text-xs font-bold">{'</>'}</span>
      </Button>
      <Button
        variant="ghost"
        size="icon"
        className="h-10 w-10 text-muted-foreground"
        disabled={disabled || audioLoading}
        onClick={() => audioInputRef.current?.click()}
        aria-label="Upload audio"
        title="Upload audio file"
      >
        <IconUpload className="h-4 w-4" aria-hidden="true" />
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
