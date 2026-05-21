'use client'

import { VoiceInput } from './VoiceInput'
import { ImageUpload } from './ImageUpload'

interface ChatInputAccessoriesProps {
  onImage: (dataUrl: string) => void
  onTranscript: (text: string) => void
  disabled: boolean
}

export function ChatInputAccessories({ onImage, onTranscript, disabled }: ChatInputAccessoriesProps) {
  return (
    <div className="flex items-center">
      <ImageUpload onImage={onImage} disabled={disabled} />
      <VoiceInput onTranscript={onTranscript} disabled={disabled} />
    </div>
  )
}
