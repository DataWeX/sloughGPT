'use client'

import { useState, useCallback, useEffect, useRef, memo } from 'react'

import { cn, Button } from '@sloughgpt/strui'
import { IconCopy, IconCheck, IconRefresh, IconEdit, IconStar, IconTrash } from '@sloughgpt/strui'
import { knowledgeController } from '@/lib/knowledge-controller'
import { useToastStore } from '@/lib/toast-store'
import { toggleReaction, getReactions } from '@/lib/reaction-store'

interface MessageActionsProps {
  content: string
  messageId: string
  role?: 'user' | 'assistant'
  onCopy?: (text: string) => void
  onRegenerate?: () => void
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
    <svg
      className={cn(className, animated && 'animate-bounce')}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z" />
      <path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
    </svg>
  )
}

function ThumbsDownIcon({ className, animated }: { className?: string; animated?: boolean }) {
  return (
    <svg
      className={cn(className, animated && 'animate-bounce')}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z" />
      <path d="M17 2h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3" />
    </svg>
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
      addToast('Failed to save to knowledge', 'error')
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
          {copied ? <IconCheck className="h-3.5 w-3.5" /> : <IconCopy className="h-3.5 w-3.5" />}
        </Button>
      )}

      {onRegenerate && (
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onRegenerate}
          className="p-2"
          aria-label="Regenerate response"
        >
          <IconRefresh className="h-3.5 w-3.5" />
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
          <IconStar className={cn('h-3.5 w-3.5', isBookmarked && 'fill-current')} />
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
          <svg className={cn('h-3.5 w-3.5', savedToKnowledge && 'text-success')} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M12 2a7 7 0 0 0-7 7c0 5 7 13 7 13s7-8 7-13a7 7 0 0 0-7-7z" />
            {savedToKnowledge && <path d="M9 12l2 2 4-4" />}
          </svg>
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
            <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z"/>
            </svg>
          ) : (
            <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3A4.5 4.5 0 0014 8.5v7a4.49 4.49 0 002.5-3.5zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>
            </svg>
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
          <IconEdit className="h-3.5 w-3.5" />
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
          <IconTrash className="h-3.5 w-3.5" />
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
            <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
            </svg>
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => onSuggestionClick(`Explain this simply:\n\n${content}`)}
            aria-label="Explain this message"
            title="Explain"
          >
            <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => onSuggestionClick(`Translate this to Spanish:\n\n${content}`)}
            aria-label="Translate to Spanish"
            title="Translate"
          >
            <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
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
          >
            <span className="text-sm">😊</span>
          </Button>
          {showEmojiPicker && (
            <div className="absolute bottom-full left-0 mb-1 flex gap-0.5 bg-popover/95 backdrop-blur-sm border border-border/40 rounded-lg p-1 shadow-xl z-50">
              {QUICK_REACTIONS.map(emoji => (
                <button
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
