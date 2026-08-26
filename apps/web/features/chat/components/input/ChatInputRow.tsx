'use client'

import { useState, useCallback, useRef, memo, type RefObject } from 'react'
import { ChatInputAccessories } from './ChatInputAccessories'
import { ChatInputField } from './ChatInputField'
import { ChatSendButton } from './ChatSendButton'
import { SlashCommandMenu } from './SlashCommandMenu'
import { MentionMenu } from './MentionMenu'
import { cn } from '@sloughgpt/strui'
import { estimateTokens } from '@/lib/format-bytes'
import type { ChatCommand } from '@/lib/chat-commands'

interface ChatInputRowProps {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onStop?: () => void
  onCancel?: () => void
  loading: boolean
  disabled: boolean
  placeholder: string
  textareaRef: RefObject<HTMLTextAreaElement>
  onImage: (dataUrl: string) => void
  onTranscript: (text: string) => void
  onAudioRecorded?: (blob: Blob) => void
  onAudioTranscript?: (text: string) => void
  onGeneratedImage?: (dataUrl: string, prompt: string) => void
  onPDFAnalysis?: (analysis: string, filename: string) => void
  onPDFError?: (error: string) => void
  hasContent: boolean
  onExecuteCommand?: (cmd: ChatCommand, args: string[]) => void
}

export const ChatInputRow = memo(function ChatInputRow({
  value, onChange, onSend, onStop, onCancel,
  loading, disabled, placeholder,
  textareaRef, onImage, onTranscript, onAudioRecorded, onAudioTranscript, onGeneratedImage,
  onPDFAnalysis, onPDFError, hasContent, onExecuteCommand,
}: ChatInputRowProps) {
  const [showSlashMenu, setShowSlashMenu] = useState(false)
  const [slashMenuDismissed, setSlashMenuDismissed] = useState(false)
  const [showMentionMenu, setShowMentionMenu] = useState(false)
  const [mentionMenuDismissed, setMentionMenuDismissed] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const showSlash = showSlashMenu && value.startsWith('/')
  const showMention = showMentionMenu && value.startsWith('@')

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

  const handleCloseMention = useCallback(() => {
    setShowMentionMenu(false)
    setMentionMenuDismissed(true)
  }, [])

  const handleChange = useCallback((newVal: string) => {
    onChange(newVal)
    if (newVal.startsWith('/') && !slashMenuDismissed) {
      setShowSlashMenu(true)
      setShowMentionMenu(false)
    } else if (newVal.startsWith('@') && !mentionMenuDismissed) {
      setShowMentionMenu(true)
      setShowSlashMenu(false)
    } else {
      setShowSlashMenu(false)
      setShowMentionMenu(false)
    }
    if (!newVal.startsWith('/')) {
      setSlashMenuDismissed(false)
    }
    if (!newVal.startsWith('@')) {
      setMentionMenuDismissed(false)
    }
  }, [onChange, slashMenuDismissed, mentionMenuDismissed])

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
        {showMention && (
          <MentionMenu
            value={value}
            onInsert={onChange}
            onClose={handleCloseMention}
          />
        )}
      </div>
      <div className="flex items-end gap-2 w-full flex-wrap rounded-2xl border border-border/50 bg-card px-3 py-2 shadow-sm focus-within:border-primary/40 focus-within:shadow-md focus-within:ring-1 focus-within:ring-primary/10 transition-all duration-200" role="group" aria-label="Message composition">
      <ChatInputAccessories
        onImage={onImage}
        onTranscript={onTranscript}
        onAudioRecorded={onAudioRecorded}
        disabled={disabled}
        onAudioTranscript={onAudioTranscript}
        onGeneratedImage={onGeneratedImage}
        onPDFAnalysis={onPDFAnalysis}
        onPDFError={onPDFError}
        textareaRef={textareaRef}
        value={value}
        onChange={onChange}
      />
      <ChatInputField
        value={value}
        onChange={handleChange}
        onSend={onSend}
        placeholder={placeholder}
        disabled={disabled}
        textareaRef={textareaRef}
        suppressEnter={showSlash || showMention}
      />
      {value.length > 0 && (
        <span
          className={cn(
            "text-[10px] tabular-nums self-end mb-2 mr-1",
            value.length > 4000 ? 'text-destructive' : value.length > 2000 ? 'text-warning' : 'text-muted-foreground/40'
          )}
          aria-live="polite"
          aria-atomic="true"
          aria-label={`Estimated ${estimateTokens(value)} tokens`}
        >
          ~{estimateTokens(value)}
        </span>
      )}
      {loading && onCancel && (
        <button
          type="button"
          className="h-10 w-10 shrink-0 flex items-center justify-center rounded-xl text-destructive hover:bg-destructive/10 transition-colors"
          onClick={onCancel}
          aria-label="Cancel generation"
        >
          <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        </button>
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
})
