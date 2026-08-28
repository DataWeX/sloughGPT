'use client'

import { useState, memo } from 'react'

import { cn, Button } from '@sloughgpt/strui'
import { IconCopy, IconCheck, IconRefresh, IconEdit, IconStar, IconTrash, IconThumbUp, IconThumbDown, IconSpeaker, IconRewrite, IconExplain, IconTranslate, IconPlay, IconMapPin, IconCheckCircle, IconMessage, IconExport, IconExternalLink, IconChevronDown } from '@sloughgpt/strui'
import { ConfirmDialog } from '@/components/ConfirmDialog'
import { useMessageActions } from './useMessageActions'

interface MessageActionsProps {
  content: string
  messageId: string
  role?: 'user' | 'assistant'
  onCopy?: (text: string) => void
  onRegenerate?: (messageId: string) => void
  onRegenerateWithOptions?: (messageId: string, options: { temperature?: number; maxTokens?: number }) => void
  onThumbsUp?: (messageId: string) => void
  onThumbsDown?: (messageId: string) => void
  onEdit?: (messageId: string) => void
  onSuggestionClick?: (text: string) => void
  isBookmarked?: boolean
  onBookmark?: (messageId: string) => void
  onDelete?: (messageId: string) => void
  onSaveToKnowledge?: (messageId: string, content: string) => void
  reactions?: Record<string, number>
  onReact?: (messageId: string, emoji: string) => void
  temperature?: number
  onAddNote?: (messageId: string) => void
  hasNote?: boolean
  onQuickReply?: (messageId: string) => void
  onForward?: (content: string) => void
  onExportMessageAsMarkdown?: (messageId: string, content: string, role: string, timestamp: string | number) => void
}

const QUICK_REACTIONS = ['👍', '👎', '❤️', '🔥', '🎉', '🤔', '👀', '💯']
const TRANSLATE_LANGUAGES = [
  { code: 'es', label: 'Spanish' },
  { code: 'fr', label: 'French' },
  { code: 'de', label: 'German' },
  { code: 'it', label: 'Italian' },
  { code: 'pt', label: 'Portuguese' },
  { code: 'ja', label: 'Japanese' },
  { code: 'ko', label: 'Korean' },
  { code: 'zh', label: 'Chinese' },
  { code: 'ar', label: 'Arabic' },
  { code: 'hi', label: 'Hindi' },
  { code: 'ru', label: 'Russian' },
]

function ThumbsUpIcon({ className, animated }: { className?: string; animated?: boolean }) {
  return (
    <span className={cn(animated && 'animate-bounce')}>
      <IconThumbUp className={className} />
    </span>
  )
}

function ThumbsDownIcon({ className, animated }: { className?: string; animated?: boolean }) {
  return (
    <span className={cn(animated && 'animate-bounce')}>
      <IconThumbDown className={className} />
    </span>
  )
}

const sparkles = ['✦', '✧', '⋆', '⁂', '✶', '✴']

function ConfettiBurst({ active }: { active: boolean }) {
  const [particles, setParticles] = useState<Array<{ x: number; y: number; char: string; delay: number }>>([])

  if (active && particles.length === 0) {
    const burst = Array.from({ length: 8 }, (_, i) => ({
      x: (Math.random() - 0.5) * 60,
      y: -Math.random() * 40 - 10,
      char: sparkles[i % sparkles.length],
      delay: i * 30,
    }))
    setParticles(burst)
  }

  if (!active && particles.length > 0) {
    setParticles([])
  }

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden" aria-hidden="true">
      {particles.map((p, i) => (
        <span
          key={i}
          className="absolute text-xs animate-ping"
          style={{ left: `calc(50% + ${p.x}px)`, top: `calc(50% + ${p.y}px)`, animationDelay: `${p.delay}ms` }}
        >
          {p.char}
        </span>
      ))}
    </div>
  )
}

export const MessageActions = memo(function MessageActions({ content, messageId, role, onCopy, onRegenerate, onRegenerateWithOptions, onThumbsUp, onThumbsDown, onEdit, onSuggestionClick, isBookmarked, onBookmark, onDelete, onSaveToKnowledge, temperature = 0.7, onAddNote, hasNote, onQuickReply, onForward, onExportMessageAsMarkdown }: MessageActionsProps) {
  const ma = useMessageActions(content, messageId, role, onCopy, onThumbsUp, onThumbsDown, onSaveToKnowledge, temperature)

  return (
    <>
    <div
      className="flex items-center gap-0 mt-0 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity relative"
      role="group"
      aria-label="Message actions"
    >
      <ConfettiBurst active={ma.celebrate} />

      {onCopy && (
        <Button variant="ghost" size="icon-sm" onClick={ma.handleCopy} className="p-2" aria-label={ma.copied ? "Copied" : "Copy message"} aria-pressed={ma.copied}>
          {ma.copied ? <IconCheck className="h-3.5 w-3.5" aria-hidden="true" /> : <IconCopy className="h-3.5 w-3.5" aria-hidden="true" />}
        </Button>
      )}

      {onQuickReply && role === 'assistant' && (
        <Button variant="ghost" size="icon-sm" onClick={() => onQuickReply(messageId)} className="p-2" aria-label="Reply to this message">
          <IconMessage className="h-3.5 w-3.5" aria-hidden="true" />
        </Button>
      )}

      {onForward && (
        <Button variant="ghost" size="icon-sm" onClick={() => onForward(content)} className="p-2" aria-label="Forward message">
          <IconExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
        </Button>
      )}

      {onExportMessageAsMarkdown && (
        <Button variant="ghost" size="icon-sm" onClick={() => onExportMessageAsMarkdown(messageId, content, role || 'user', '')} className="p-2" aria-label="Export as Markdown">
          <IconExport className="h-3.5 w-3.5" aria-hidden="true" />
        </Button>
      )}

      {onRegenerate && (
        <div className="relative" ref={ma.retryOptionsRef}>
          <div className="flex">
            <Button variant="ghost" size="icon-sm" onClick={() => onRegenerate(messageId)} className="p-2 rounded-r-none" aria-label="Regenerate response">
              <IconRefresh className="h-3.5 w-3.5" aria-hidden="true" />
            </Button>
            {onRegenerateWithOptions && (
              <Button variant="ghost" size="icon-sm" onClick={() => { ma.setRetryTemperature(temperature); ma.setShowRetryOptions(!ma.showRetryOptions) }} className="p-1 rounded-l-none border-l border-border/30" aria-label="Regenerate with options" aria-expanded={ma.showRetryOptions}>
                <IconChevronDown className="h-2.5 w-2.5" aria-hidden="true" />
              </Button>
            )}
          </div>
          {ma.showRetryOptions && onRegenerateWithOptions && (
            <div className="absolute bottom-full left-0 mb-1 bg-popover/95 backdrop-blur-sm border border-border/40 rounded-lg p-3 shadow-xl z-50 w-52">
              <div className="space-y-2">
                <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Temperature: {ma.retryTemperature.toFixed(2)}</label>
                <input type="range" min={0} max={2} step={0.05} value={ma.retryTemperature} onChange={e => ma.setRetryTemperature(parseFloat(e.target.value))} className="w-full h-1.5 bg-muted rounded-full appearance-none cursor-pointer accent-primary" aria-label="Retry temperature" />
                <div className="flex justify-between text-[9px] text-muted-foreground/50"><span>Precise</span><span>Creative</span></div>
                <Button size="sm" className="w-full h-7 text-xs" onClick={() => { onRegenerateWithOptions(messageId, { temperature: ma.retryTemperature }); ma.setShowRetryOptions(false) }}>Retry with options</Button>
              </div>
            </div>
          )}
        </div>
      )}

      {onThumbsUp && (
        <Button variant="ghost" size="icon-sm" onClick={ma.handleThumbsUp} className="p-2" aria-label="Mark as helpful" aria-pressed={ma.thumbsUp}>
          <ThumbsUpIcon className="h-3.5 w-3.5" animated={ma.thumbsUp} />
        </Button>
      )}

      {onThumbsDown && (
        <Button variant="ghost" size="icon-sm" onClick={ma.handleThumbsDown} className="p-2" aria-label="Mark as unhelpful" aria-pressed={ma.thumbsDown}>
          <ThumbsDownIcon className="h-3.5 w-3.5" animated={ma.thumbsDown} />
        </Button>
      )}

      {onBookmark && (
        <Button variant="ghost" size="icon-sm" onClick={() => onBookmark(messageId)} className="p-2" aria-label={isBookmarked ? 'Remove bookmark' : 'Bookmark message'}>
          <IconStar className={cn('h-3.5 w-3.5', isBookmarked && 'fill-current')} aria-hidden="true" />
        </Button>
      )}

      {onAddNote && (
        <Button variant="ghost" size="icon-sm" onClick={() => onAddNote(messageId)} className={cn('p-2', hasNote && 'text-primary/70')} aria-label={hasNote ? 'Edit note' : 'Add note'}>
          <IconMessage className="h-3.5 w-3.5" aria-hidden="true" />
        </Button>
      )}

      {onSaveToKnowledge && (
        <Button variant="ghost" size="icon-sm" onClick={ma.handleSaveToKnowledge} className="p-2" aria-label={ma.savedToKnowledge ? 'Already saved to knowledge' : 'Save to knowledge'} disabled={ma.savedToKnowledge}>
          {ma.savedToKnowledge ? <IconCheckCircle className={cn('h-3.5 w-3.5 text-success')} aria-hidden="true" /> : <IconMapPin className="h-3.5 w-3.5" aria-hidden="true" />}
        </Button>
      )}

      {role === 'assistant' && 'speechSynthesis' in window && (
        <Button variant="ghost" size="icon-sm" onClick={ma.handleSpeak} className="p-2" aria-label={ma.speaking ? 'Stop reading aloud' : 'Read aloud'}>
          {ma.speaking ? <IconPlay className="h-3.5 w-3.5" /> : <IconSpeaker className="h-3.5 w-3.5" />}
        </Button>
      )}

      {onEdit && (
        <Button variant="ghost" size="icon-sm" onClick={() => onEdit(messageId)} aria-label="Edit and resend message">
          <IconEdit className="h-3.5 w-3.5" aria-hidden="true" />
        </Button>
      )}

      {onDelete && (
        <>
          <Button variant="ghost" size="icon-sm" onClick={() => ma.setShowDeleteConfirm(true)} aria-label="Delete message" className="hover:text-destructive text-muted-foreground p-2">
            <IconTrash className="h-3.5 w-3.5" aria-hidden="true" />
          </Button>
          <ConfirmDialog open={ma.showDeleteConfirm} onOpenChange={ma.setShowDeleteConfirm} title="Delete message" description="Are you sure you want to delete this message? This cannot be undone." confirmLabel="Delete" onConfirm={() => { onDelete(messageId); ma.setShowDeleteConfirm(false) }} />
        </>
      )}

      {onSuggestionClick && (
        <>
          <span className="w-px h-4 mx-0.5 bg-border/50" aria-hidden="true" />
          <Button variant="ghost" size="icon-sm" onClick={() => onSuggestionClick(`Rewrite this:\n\n${content}`)} aria-label="Rewrite this message" title="Rewrite">
            <IconRewrite className="h-3.5 w-3.5" aria-hidden="true" />
          </Button>
          <Button variant="ghost" size="icon-sm" onClick={() => onSuggestionClick(`Explain this simply:\n\n${content}`)} aria-label="Explain this message" title="Explain">
            <IconExplain className="h-3.5 w-3.5" aria-hidden="true" />
          </Button>
          <div className="relative" ref={ma.translateMenuRef}>
            <Button variant="ghost" size="icon-sm" onClick={() => ma.setShowTranslateMenu(!ma.showTranslateMenu)} aria-label="Translate message" aria-expanded={ma.showTranslateMenu} title="Translate">
              <IconTranslate className="h-3.5 w-3.5" aria-hidden="true" />
            </Button>
            {ma.showTranslateMenu && (
              <div className="absolute bottom-full left-0 mb-1 bg-popover/95 backdrop-blur-sm border border-border/40 rounded-lg p-1 shadow-xl z-50 min-w-[140px]">
                {TRANSLATE_LANGUAGES.map(lang => (
                  <button type="button" key={lang.code} onClick={() => { onSuggestionClick(`Translate this to ${lang.label}:\n\n${content}`); ma.setShowTranslateMenu(false) }} className="w-full text-left px-2 py-1 text-xs rounded hover:bg-accent/50 transition-colors text-foreground">
                    {lang.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        </>
      )}

      <div className="relative" ref={ma.emojiPickerRef}>
        <Button variant="ghost" size="icon-sm" onClick={() => ma.setShowEmojiPicker(!ma.showEmojiPicker)} className="p-2" aria-label="Add reaction" aria-expanded={ma.showEmojiPicker}>
          <span className="text-sm">😊</span>
        </Button>
        {ma.showEmojiPicker && (
          <div className="absolute bottom-full left-0 mb-1 flex gap-0.5 bg-popover/95 backdrop-blur-sm border border-border/40 rounded-lg p-1 shadow-xl z-50">
            {QUICK_REACTIONS.map(emoji => (
              <button type="button" key={emoji} onClick={() => { ma.handleToggleReaction(emoji); ma.setShowEmojiPicker(false) }} className="w-7 h-7 flex items-center justify-center hover:bg-accent/50 rounded transition-colors text-sm" aria-label={`React with ${emoji}`}>
                {emoji}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>

    {Object.keys(ma.localReactions).length > 0 && (
      <div className="flex flex-wrap gap-1 mt-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
        {Object.entries(ma.localReactions).map(([emoji, count]) => (
          <button type="button" key={emoji} onClick={() => ma.handleToggleReaction(emoji)} className={cn("inline-flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded-full border transition-colors", count > 0 ? "bg-primary/15 border-primary/30 text-primary" : "bg-muted/50 border-border/30 text-muted-foreground hover:bg-muted/80")}>
            <span>{emoji}</span>
            <span>{count}</span>
          </button>
        ))}
      </div>
    )}
    </>
  )
})
