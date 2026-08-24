'use client'

import { useState, useCallback, useEffect, useRef, memo } from 'react'

import { cn, Button } from '@sloughgpt/strui'
import { IconCopy, IconCheck, IconRefresh, IconEdit, IconStar, IconTrash, IconThumbUp, IconThumbDown, IconSpeaker, IconRewrite, IconExplain, IconTranslate, IconPlay, IconMapPin, IconCheckCircle } from '@sloughgpt/strui'
import { knowledgeController } from '@/lib/knowledge-controller'
import { useToastStore } from '@/lib/toast-store'
import { toggleReaction, getReactions } from '@/lib/reaction-store'

interface MessageActionsProps {
  content: string
  messageId: string
  role?: 'user' | 'assistant'
  onCopy?: (text: string) => void
  onRegenerate?: (messageId: string) => void
  onThumbsUp?: (messageId: string) => void
  onThumbsDown?: (messageId: string) => void
  onEdit?: (messageId: string) => void
  onSuggestionClick?: (text: string) => void
  isBookmarked?: boolean
  onBookmark?: (messageId: string) => void
  onDelete?: (messageId: string) => void
  onSaveToKnowledge?: (messageId: string, content: string) => void
  reactions?: Record<string, string[]>
  onReact?: (messageId: string, emoji: string) => void
}

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

  useEffect(() => {
    if (!active) {
      setParticles([])
      return
    }
    const burst = Array.from({ length: 8 }, (_, i) => ({
      x: Math.random() * 60 - 30,
      y: Math.random() * -40 - 10,
      char: sparkles[Math.floor(Math.random() * sparkles.length)],
      delay: i * 30,
    }))
    setParticles(burst)
    const timer = setTimeout(() => setParticles([]), 600)
    return () => clearTimeout(timer)
  }, [active])

  if (particles.length === 0) return null

  return (
    <div className="absolute inset-0 pointer-events-none overflow-visible" aria-hidden="true">
      {particles.map((p, i) => (
        <span
          key={i}
          className="absolute text-xs animate-ping"
          style={{
            left: `calc(50% + ${p.x}px)`,
            top: `calc(50% + ${p.y}px)`,
            animationDelay: `${p.delay}ms`,
            animationDuration: '400ms',
            opacity: 0.7,
            color: `hsl(${i * 45}, 70%, 55%)`,
          }}
        >
          {p.char}
        </span>
      ))}
    </div>
  )
}

const QUICK_REACTIONS = ['👍', '👎', '❤️', '🔥', '🎉', '🤔', '👀', '💯']

export const MessageActions = memo(function MessageActions({ content, messageId, role, onCopy, onRegenerate, onThumbsUp, onThumbsDown, onEdit, onSuggestionClick, isBookmarked, onBookmark, onDelete, onSaveToKnowledge }: MessageActionsProps) {
  const [copied, setCopied] = useState(false)
  const [thumbsUp, setThumbsUp] = useState(false)
  const [thumbsDown, setThumbsDown] = useState(false)
  const [celebrate, setCelebrate] = useState(false)
  const [speaking, setSpeaking] = useState(false)
  const [savedToKnowledge, setSavedToKnowledge] = useState(false)
  const [showEmojiPicker, setShowEmojiPicker] = useState(false)
  const [localReactions, setLocalReactions] = useState<Record<string, string[]>>(() => getReactions(messageId))
  const addToast = useToastStore(s => s.addToast)
  const copiedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const celebrateTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    return () => {
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current)
      if (celebrateTimerRef.current) clearTimeout(celebrateTimerRef.current)
    }
  }, [])

  const handleCopy = useCallback(async () => {
    if (!content || !onCopy) return
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      onCopy(content)
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current)
      copiedTimerRef.current = setTimeout(() => setCopied(false), 1500)
    } catch { /* clipboard API may be unavailable */ }
  }, [content, onCopy])

  const handleThumbsUp = useCallback(() => {
    const newVal = !thumbsUp
    setThumbsUp(newVal)
    if (newVal) {
      setThumbsDown(false)
      setCelebrate(true)
      if (celebrateTimerRef.current) clearTimeout(celebrateTimerRef.current)
      celebrateTimerRef.current = setTimeout(() => setCelebrate(false), 600)
    }
    onThumbsUp?.(messageId)
  }, [thumbsUp, messageId, onThumbsUp])

  const handleThumbsDown = useCallback(() => {
    const newVal = !thumbsDown
    setThumbsDown(newVal)
    if (newVal) {
      setThumbsUp(false)
    }
    onThumbsDown?.(messageId)
  }, [thumbsDown, messageId, onThumbsDown])

  const handleSpeak = useCallback(() => {
    if (!('speechSynthesis' in window)) return
    if (speaking) {
      window.speechSynthesis.cancel()
      setSpeaking(false)
      return
    }
    const utterance = new SpeechSynthesisUtterance(content)
    utterance.rate = 0.9
    utterance.pitch = 1.0
    utterance.onend = () => setSpeaking(false)
    utterance.onerror = () => setSpeaking(false)
    window.speechSynthesis.speak(utterance)
    setSpeaking(true)
  }, [content, speaking])

  const handleSaveToKnowledge = useCallback(async () => {
    if (!content || savedToKnowledge) return
    try {
      await knowledgeController.add(content.slice(0, 500), 'chat-saved', true)
      setSavedToKnowledge(true)
      addToast('Saved to knowledge', 'success')
    } catch {
      addToast('Could not save to knowledge', 'error')
    }
  }, [content, savedToKnowledge, addToast])

  const handleToggleReaction = useCallback((emoji: string) => {
    toggleReaction(messageId, emoji)
    setLocalReactions(getReactions(messageId))
  }, [messageId])

  return (
    <>
    <div
      className="flex items-center gap-0 mt-0 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity relative"
      role="group"
      aria-label="Message actions"
    >
      <ConfettiBurst active={celebrate} />

      {onCopy && (
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={handleCopy}
          className="p-2"
          aria-label={copied ? "Copied" : "Copy message"}
          aria-pressed={copied}
        >
          {copied ? <IconCheck className="h-3.5 w-3.5" aria-hidden="true" /> : <IconCopy className="h-3.5 w-3.5" aria-hidden="true" />}
        </Button>
      )}

      {onRegenerate && (
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => onRegenerate(messageId)}
          className="p-2"
          aria-label="Regenerate response"
        >
          <IconRefresh className="h-3.5 w-3.5" aria-hidden="true" />
        </Button>
      )}

      {onThumbsUp && (
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={handleThumbsUp}
          className="p-2"
          aria-label="Mark as helpful"
          aria-pressed={thumbsUp}
        >
          <ThumbsUpIcon className="h-3.5 w-3.5" animated={thumbsUp} />
        </Button>
      )}

      {onThumbsDown && (
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={handleThumbsDown}
          className="p-2"
          aria-label="Mark as unhelpful"
          aria-pressed={thumbsDown}
        >
          <ThumbsDownIcon className="h-3.5 w-3.5" animated={thumbsDown} />
        </Button>
      )}

      {onBookmark && (
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => onBookmark(messageId)}
          className="p-2"
          aria-label={isBookmarked ? 'Remove bookmark' : 'Bookmark message'}
        >
          <IconStar className={cn('h-3.5 w-3.5', isBookmarked && 'fill-current')} aria-hidden="true" />
        </Button>
      )}

      {onSaveToKnowledge && (
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={handleSaveToKnowledge}
          className="p-2"
          aria-label={savedToKnowledge ? 'Already saved to knowledge' : 'Save to knowledge'}
          disabled={savedToKnowledge}
        >
          {savedToKnowledge ? (
            <IconCheckCircle className={cn('h-3.5 w-3.5 text-success')} aria-hidden="true" />
          ) : (
            <IconMapPin className="h-3.5 w-3.5" aria-hidden="true" />
          )}
        </Button>
      )}

      {role === 'assistant' && 'speechSynthesis' in window && (
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={handleSpeak}
          className="p-2"
          aria-label={speaking ? 'Stop reading aloud' : 'Read aloud'}
        >
          {speaking ? (
            <IconPlay className="h-3.5 w-3.5" />
          ) : (
            <IconSpeaker className="h-3.5 w-3.5" />
          )}
        </Button>
      )}

      {onEdit && (
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => onEdit(messageId)}
          aria-label="Edit and resend message"
        >
          <IconEdit className="h-3.5 w-3.5" aria-hidden="true" />
        </Button>
      )}

      {onDelete && (
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => onDelete(messageId)}
          aria-label="Delete message"
          className="hover:text-destructive text-muted-foreground p-2"
        >
          <IconTrash className="h-3.5 w-3.5" aria-hidden="true" />
        </Button>
      )}

      {onSuggestionClick && (
        <>
          <span className="w-px h-4 mx-0.5 bg-border/50" aria-hidden="true" />
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => onSuggestionClick(`Rewrite this:\n\n${content}`)}
            aria-label="Rewrite this message"
            title="Rewrite"
          >
            <IconRewrite className="h-3.5 w-3.5" aria-hidden="true" />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => onSuggestionClick(`Explain this simply:\n\n${content}`)}
            aria-label="Explain this message"
            title="Explain"
          >
            <IconExplain className="h-3.5 w-3.5" aria-hidden="true" />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => onSuggestionClick(`Translate this to Spanish:\n\n${content}`)}
            aria-label="Translate to Spanish"
            title="Translate"
          >
            <IconTranslate className="h-3.5 w-3.5" aria-hidden="true" />
          </Button>
        </>
      )}

      <div className="relative">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setShowEmojiPicker(!showEmojiPicker)}
            className="p-2"
            aria-label="Add reaction"
            aria-expanded={showEmojiPicker}
          >
            <span className="text-sm">😊</span>
          </Button>
          {showEmojiPicker && (
            <div className="absolute bottom-full left-0 mb-1 flex gap-0.5 bg-popover/95 backdrop-blur-sm border border-border/40 rounded-lg p-1 shadow-xl z-50">
              {QUICK_REACTIONS.map(emoji => (
                <button
                  type="button"
                  key={emoji}
                  onClick={() => {
                    handleToggleReaction(emoji)
                    setShowEmojiPicker(false)
                  }}
                  className="w-7 h-7 flex items-center justify-center hover:bg-accent/50 rounded transition-colors text-sm"
                  aria-label={`React with ${emoji}`}
                >
                  {emoji}
                </button>
              ))}
            </div>
          )}
        </div>
    </div>

    {Object.keys(localReactions).length > 0 && (
      <div className="flex flex-wrap gap-1 mt-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
        {Object.entries(localReactions).map(([emoji, users]) => (
          <button
            type="button"
            key={emoji}
            onClick={() => handleToggleReaction(emoji)}
            className={cn(
              "inline-flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded-full border transition-colors",
              users.includes('user') ? "bg-primary/15 border-primary/30 text-primary" : "bg-muted/50 border-border/30 text-muted-foreground hover:bg-muted/80"
            )}
          >
            <span>{emoji}</span>
            <span>{users.length}</span>
          </button>
        ))}
      </div>
    )}
    </>
  )
})
