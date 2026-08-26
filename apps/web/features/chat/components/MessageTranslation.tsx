'use client'

import { useState, useCallback, memo } from 'react'
import { Button, IconX, IconCheck } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

interface MessageTranslationProps {
  content: string
  onTranslate: (content: string, targetLang: string) => Promise<string>
  className?: string
}

const LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'it', name: 'Italian' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'ja', name: 'Japanese' },
  { code: 'ko', name: 'Korean' },
  { code: 'zh', name: 'Chinese' },
  { code: 'ru', name: 'Russian' },
  { code: 'ar', name: 'Arabic' },
  { code: 'hi', name: 'Hindi' },
]

export const MessageTranslation = memo(function MessageTranslation({
  content,
  onTranslate,
  className,
}: MessageTranslationProps) {
  const [open, setOpen] = useState(false)
  const [targetLang, setTargetLang] = useState('es')
  const [translated, setTranslated] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleTranslate = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await onTranslate(content, targetLang)
      setTranslated(result)
    } catch (e) {
      setError('Translation failed')
    } finally {
      setLoading(false)
    }
  }, [content, targetLang, onTranslate])

  const handleOpen = useCallback(() => {
    setOpen(true)
    setTranslated(null)
    setError(null)
  }, [])

  const handleClose = useCallback(() => {
    setOpen(false)
    setTranslated(null)
    setError(null)
  }, [])

  if (!open) {
    return (
      <Button
        variant="ghost"
        size="sm"
        className={cn('text-[10px] h-5 text-muted-foreground/50 hover:text-muted-foreground', className)}
        onClick={handleOpen}
      >
        Translate
      </Button>
    )
  }

  return (
    <div className={cn('mt-1 rounded border bg-muted/30 p-2 space-y-2', className)}>
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-medium text-muted-foreground">Translate to</span>
        <Button variant="ghost" size="icon-sm" className="h-4 w-4" onClick={handleClose}>
          <IconX className="h-2.5 w-2.5" />
        </Button>
      </div>

      <div className="flex flex-wrap gap-1">
        {LANGUAGES.map(lang => (
          <button
            key={lang.code}
            type="button"
            onClick={() => setTargetLang(lang.code)}
            className={cn(
              'text-[10px] px-1.5 py-0.5 rounded transition-colors',
              targetLang === lang.code
                ? 'bg-primary/20 text-primary'
                : 'text-muted-foreground hover:bg-muted/50',
            )}
          >
            {lang.name}
          </button>
        ))}
      </div>

      <Button
        variant="ghost"
        size="sm"
        className="text-[10px] h-6"
        onClick={handleTranslate}
        disabled={loading}
      >
        {loading ? 'Translating...' : 'Translate'}
      </Button>

      {error && (
        <p className="text-[10px] text-destructive">{error}</p>
      )}

      {translated && (
        <div className="rounded bg-background p-2">
          <p className="text-xs whitespace-pre-wrap">{translated}</p>
        </div>
      )}
    </div>
  )
})