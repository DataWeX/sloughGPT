'use client'

import { useState, useCallback, useRef, useEffect, memo } from 'react'
import { Button } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

interface MessageFormattingToolbarProps {
  text: string
  onFormat: (formattedText: string) => void
  onCopy: (text: string) => void
  onClear?: () => void
  className?: string
}

const FORMAT_OPTIONS = [
  { label: 'Bold', prefix: '**', suffix: '**', shortcut: 'Ctrl+B' },
  { label: 'Italic', prefix: '_', suffix: '_', shortcut: 'Ctrl+I' },
  { label: 'Code', prefix: '`', suffix: '`', shortcut: 'Ctrl+E' },
  { label: 'Link', prefix: '[', suffix: '](url)', shortcut: 'Ctrl+K' },
  { label: 'List', prefix: '\n- ', suffix: '', shortcut: '' },
  { label: 'Numbered', prefix: '\n1. ', suffix: '', shortcut: '' },
  { label: 'Quote', prefix: '\n> ', suffix: '', shortcut: '' },
  { label: 'H1', prefix: '\n# ', suffix: '', shortcut: '' },
  { label: 'H2', prefix: '\n## ', suffix: '', shortcut: '' },
  { label: 'H3', prefix: '\n### ', suffix: '', shortcut: '' },
]

export const MessageFormattingToolbar = memo(function MessageFormattingToolbar({
  text,
  onFormat,
  onCopy,
  onClear,
  className,
}: MessageFormattingToolbarProps) {
  const [expanded, setExpanded] = useState(false)
  const [copied, setCopied] = useState(false)
  const [selectedFormat, setSelectedFormat] = useState<string | null>(null)
  const timerRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [])

  const handleFormat = useCallback((prefix: string, suffix: string) => {
    const formatted = `${prefix}${text || 'text'}${suffix}`
    onFormat(formatted)
    setSelectedFormat(prefix)
    setTimeout(() => setSelectedFormat(null), 200)
  }, [text, onFormat])

  const handleCopy = useCallback(() => {
    onCopy(text)
    setCopied(true)
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setCopied(false), 2000)
  }, [text, onCopy])

  return (
    <div className={cn('flex items-center gap-1', className)}>
      <Button
        variant="ghost"
        size="sm"
        className="text-[10px] h-6"
        onClick={() => setExpanded(!expanded)}
        title="Formatting options"
      >
        {expanded ? 'Less' : 'More'}
      </Button>

      {expanded && (
        <>
          {FORMAT_OPTIONS.map(option => (
            <Button
              key={option.label}
              variant="ghost"
              size="sm"
              className={cn(
                'text-[10px] h-6',
                selectedFormat === option.prefix && 'bg-primary/20 text-primary',
              )}
              onClick={() => handleFormat(option.prefix, option.suffix)}
              title={option.shortcut ? `${option.label} (${option.shortcut})` : option.label}
            >
              {option.label}
            </Button>
          ))}
        </>
      )}

      <Button
        variant="ghost"
        size="sm"
        className={cn(
          'text-[10px] h-6',
          copied && 'bg-success/20 text-success',
        )}
        onClick={handleCopy}
      >
        {copied ? 'Copied!' : 'Copy'}
      </Button>

      {onClear && text && (
        <Button
          variant="ghost"
          size="sm"
          className="text-[10px] h-6 text-destructive"
          onClick={onClear}
        >
          Clear
        </Button>
      )}
    </div>
  )
})