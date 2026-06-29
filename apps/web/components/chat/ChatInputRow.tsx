'use client'

import { useState, useCallback, useRef, type RefObject } from 'react'
import { ChatInputAccessories } from './ChatInputAccessories'
import { ChatInputField } from './ChatInputField'
import { ChatSendButton } from './ChatSendButton'
import { SlashCommandMenu } from './SlashCommandMenu'
import { cn } from '@/lib/cn'
import type { ChatCommand } from '@/lib/chat-commands'

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
  onAudioTranscript?: (text: string) => void
  onGeneratedImage?: (dataUrl: string, prompt: string) => void
  onPDFAnalysis?: (analysis: string, filename: string) => void
  onPDFError?: (error: string) => void
  hasContent: boolean
  onExecuteCommand?: (cmd: ChatCommand, args: string[]) => void
}

export function ChatInputRow({
  value, onChange, onSend, onStop,
  loading, disabled, placeholder,
  textareaRef, onImage, onTranscript, onAudioTranscript, onGeneratedImage,
  onPDFAnalysis, onPDFError, hasContent, onExecuteCommand,
}: ChatInputRowProps) {
  const [showSlashMenu, setShowSlashMenu] = useState(false)
  const [slashMenuDismissed, setSlashMenuDismissed] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const showSlash = showSlashMenu && value.startsWith('/')

  const handleExecuteCommand = useCallback((cmd: ChatCommand, args: string[]) => {
    if (onExecuteCommand) {
      onExecuteCommand(cmd, args)
    } else {
      onChange(cmd.command + ' ')
    }
    setShowSlashMenu(false)
    setSlashMenuDismissed(true)
  }, [onChange, onExecuteCommand])

  const handleCloseSlash = useCallback(() => {
    setShowSlashMenu(false)
    setSlashMenuDismissed(true)
  }, [])

  const handleChange = useCallback((newVal: string) => {
    onChange(newVal)
    if (newVal.startsWith('/') && !slashMenuDismissed) {
      setShowSlashMenu(true)
    } else {
      setShowSlashMenu(false)
    }
    if (!newVal.startsWith('/')) {
      setSlashMenuDismissed(false)
    }
  }, [onChange, slashMenuDismissed])

  return (
    <div className="flex flex-col w-full" ref={containerRef}>
      <div className="relative">
        {showSlash && (
          <SlashCommandMenu
            value={value}
            onInsert={onChange}
            onClose={handleCloseSlash}
            onExecute={handleExecuteCommand}
          />
        )}
      </div>
      <div className="flex items-end gap-1.5 w-full rounded-xl border border-border/40 bg-muted/10 px-2.5 py-1.5 focus-within:border-primary/30 focus-within:bg-muted/15 transition-all duration-200" role="group" aria-label="Message composition">
      <ChatInputAccessories
        onImage={onImage}
        onTranscript={onTranscript}
        disabled={disabled}
        onAudioTranscript={onAudioTranscript}
        onGeneratedImage={onGeneratedImage}
        onPDFAnalysis={onPDFAnalysis}
        onPDFError={onPDFError}
      />
      <ChatInputField
        value={value}
        onChange={handleChange}
        onSend={onSend}
        placeholder={placeholder}
        disabled={disabled}
        textareaRef={textareaRef}
        suppressEnter={showSlash}
      />
      {value.length > 0 && (
        <span
          className={cn(
            "text-[10px] tabular-nums self-end mb-1.5 mr-1",
            value.length > 4000 ? 'text-destructive' : value.length > 2000 ? 'text-warning' : 'text-muted-foreground/50'
          )}
          aria-live="polite"
          aria-atomic="true"
          aria-label={`Characters typed: ${value.length}`}
        >
          {value.length} chars
        </span>
      )}
      <ChatSendButton
        loading={loading}
        hasContent={hasContent}
        onSend={onSend}
        onStop={onStop}
        disabled={disabled}
      />
      </div>
    </div>
  )
}
