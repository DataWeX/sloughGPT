'use client'

import { useCallback, useEffect, useState, type KeyboardEvent, type ChangeEvent } from 'react'
import { Textarea } from '@sloughgpt/strui'

interface ChatInputFieldProps {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  placeholder: string
  disabled: boolean
  textareaRef: React.RefObject<HTMLTextAreaElement>
  suppressEnter?: boolean
}

const PLACEHOLDERS = [
  'Ask anything...',
  'Type a message...',
  'What\'s on your mind?',
  'Ask me something...',
]

function autoResize(textarea: HTMLTextAreaElement | null) {
  if (textarea) {
    textarea.style.height = 'auto'
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`
  }
}

export function ChatInputField({ value, onChange, onSend, placeholder, disabled, textareaRef, suppressEnter }: ChatInputFieldProps) {
  const [phIndex, setPhIndex] = useState(0)

  useEffect(() => {
    if (value) return
    const interval = setInterval(() => {
      setPhIndex(i => (i + 1) % PLACEHOLDERS.length)
    }, 5000)
    return () => clearInterval(interval)
  }, [value])

  const activePlaceholder = value ? placeholder : PLACEHOLDERS[phIndex]

  const handleChange = useCallback((e: ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value)
    autoResize(e.target.value ? textareaRef.current : null)
  }, [onChange, textareaRef])

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (!suppressEnter) onSend()
    }
  }, [onSend, suppressEnter])

  return (
    <>
      <Textarea
        ref={textareaRef}
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder={activePlaceholder}
        disabled={disabled}
        className="flex-1 min-w-0 resize-none min-h-[40px] max-h-[160px] bg-transparent border-0 shadow-none focus:shadow-none focus-visible:ring-0 focus-visible:ring-offset-0 text-sm placeholder:text-muted-foreground/50 py-2"
        rows={1}
        aria-label="Message input"
        aria-describedby="chat-input-hint"
      />
      <p id="chat-input-hint" className="sr-only">Press Enter to send, Shift+Enter for new line</p>
    </>
  )
}
