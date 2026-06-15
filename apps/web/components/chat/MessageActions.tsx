'use client'

import { useState, useCallback, useEffect } from 'react'

import { Button } from '@/components/ui/button'
import { IconCopy, IconCheck, IconRefresh, IconEdit } from '@/components/ui'
import { cn } from '@/lib/cn'

interface MessageActionsProps {
  content: string
  messageId: string
  onCopy?: (text: string) => void
  onRegenerate?: () => void
  onThumbsUp?: (messageId: string) => void
  onThumbsDown?: (messageId: string) => void
  onEdit?: (messageId: string) => void
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

export function MessageActions({ content, messageId, onCopy, onRegenerate, onThumbsUp, onThumbsDown, onEdit }: MessageActionsProps) {
  const [copied, setCopied] = useState(false)
  const [thumbsUp, setThumbsUp] = useState(false)
  const [thumbsDown, setThumbsDown] = useState(false)
  const [celebrate, setCelebrate] = useState(false)

  const handleCopy = useCallback(async () => {
    if (!content || !onCopy) return
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      onCopy(content)
      setTimeout(() => setCopied(false), 1500)
    } catch {}
  }, [content, onCopy])

  const handleThumbsUp = useCallback(() => {
    const newVal = !thumbsUp
    setThumbsUp(newVal)
    if (newVal) {
      setThumbsDown(false)
      setCelebrate(true)
      setTimeout(() => setCelebrate(false), 600)
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

  return (
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
          className={cn('transition-all duration-200', copied && 'text-success scale-110')}
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
          className={cn('transition-all duration-200', thumbsUp && 'text-success')}
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
          className={cn('transition-all duration-200', thumbsDown && 'text-destructive/80')}
          aria-label="Mark as unhelpful"
          aria-pressed={thumbsDown}
        >
          <ThumbsDownIcon className="h-3.5 w-3.5" animated={thumbsDown} />
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
    </div>
  )
}
