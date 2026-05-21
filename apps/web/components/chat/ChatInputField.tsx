'use client'

import { useCallback, type KeyboardEvent, type ChangeEvent } from 'react'
import { Textarea } from '@/components/ui/textarea'

interface ChatInputFieldProps {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  placeholder: string
  disabled: boolean
  textareaRef: React.RefObject<HTMLTextAreaElement>
}

function autoResize(textarea: HTMLTextAreaElement | null) {
  if (textarea) {
    textarea.style.height = 'auto'
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`
  }
}

export function ChatInputField({ value, onChange, onSend, placeholder, disabled, textareaRef }: ChatInputFieldProps) {
  const handleChange = useCallback((e: ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value)
    autoResize(e.target.value ? textareaRef.current : null)
  }, [onChange, textareaRef])

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      onSend()
    }
  }, [onSend])

  return (
    <Textarea
      ref={textareaRef}
      value={value}
      onChange={handleChange}
      onKeyDown={handleKeyDown}
      placeholder={placeholder}
      disabled={disabled}
      className="flex-1 min-w-0 max-w-xl resize-none min-h-[40px] max-h-[160px] bg-background/60 shadow-sm focus:shadow-md transition-shadow"
      rows={1}
      aria-label="Message input"
    />
  )
}
