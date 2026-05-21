'use client'

import type { RefObject } from 'react'
import { ChatInputAccessories } from './ChatInputAccessories'
import { ChatInputField } from './ChatInputField'
import { ChatSendButton } from './ChatSendButton'

interface ChatInputRowProps {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onStop?: () => void
  loading: boolean
  disabled: boolean
  placeholder: string
  textareaRef: RefObject<HTMLTextAreaElement>
  onImage: (dataUrl: string) => void
  onTranscript: (text: string) => void
  hasContent: boolean
}

export function ChatInputRow({
  value, onChange, onSend, onStop,
  loading, disabled, placeholder,
  textareaRef, onImage, onTranscript, hasContent,
}: ChatInputRowProps) {
  return (
    <div className="flex items-center justify-center gap-3 mx-auto">
      <ChatInputAccessories
        onImage={onImage}
        onTranscript={onTranscript}
        disabled={disabled}
      />
      <ChatInputField
        value={value}
        onChange={onChange}
        onSend={onSend}
        placeholder={placeholder}
        disabled={disabled}
        textareaRef={textareaRef}
      />
      <ChatSendButton
        loading={loading}
        hasContent={hasContent}
        onSend={onSend}
        onStop={onStop}
        disabled={disabled}
      />
    </div>
  )
}
