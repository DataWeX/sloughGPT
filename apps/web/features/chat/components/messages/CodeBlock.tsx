'use client'

/**
 * CodeBlock — syntax-highlighted code block with lazy Prism language loading.
 *
 * During streaming, shows plain text to avoid jank from Prism highlighting.
 * After streaming completes, lazily imports the language grammar and highlights.
 */

import { memo, useState, useEffect, useMemo } from 'react'
import { COPY_FEEDBACK_DURATION_MS } from '@/lib/constants'

const loadedLangs = new Set<string>()

async function ensureLanguage(lang: string) {
  if (!lang || loadedLangs.has(lang)) return
  try {
    await import(`prismjs/components/prism-${lang}`)
    loadedLangs.add(lang)
  } catch {
    // Language not available — fallback to plain text
  }
}

function escapeHtml(code: string): string {
  return code
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

function highlightCode(code: string, language: string): string {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const Prism = require('prismjs')
    const lang = language.toLowerCase()
    const grammar = Prism.languages[lang]
    if (grammar) {
      const highlighted = Prism.highlight(code, grammar, lang)
      // Strip unsafe HTML tags from Prism output
      return highlighted.replace(/<(?!\/?span\b|\/?code\b)[^>]*>/g, (tag: string) => {
        if (tag.startsWith('<script') || tag.startsWith('<iframe') || tag.startsWith('<object') || tag.startsWith('<embed')) return ''
        return tag.replace(/</g, '&lt;').replace(/>/g, '&gt;')
      })
    }
  } catch {
    // Prism not loaded yet
  }
  return escapeHtml(code)
}

interface CodeBlockProps {
  language: string
  code: string
  isStreaming?: boolean
}

export const CodeBlock = memo(function CodeBlock({ language, code, isStreaming }: CodeBlockProps) {
  const [copied, setCopied] = useState(false)
  const [highlighted, setHighlighted] = useState<string | null>(null)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), COPY_FEEDBACK_DURATION_MS)
  }

  // Lazy load language grammar and highlight after streaming stops
  useEffect(() => {
    if (isStreaming || !language) {
      setHighlighted(null)
      return
    }
    let cancelled = false
    ensureLanguage(language).then(() => {
      if (!cancelled) {
        setHighlighted(highlightCode(code, language))
      }
    })
    return () => { cancelled = true }
  }, [code, language, isStreaming])

  const displayHtml = useMemo(() => {
    if (highlighted) return highlighted
    return escapeHtml(code)
  }, [highlighted, code])

  return (
    <div className="relative my-3 rounded-xl border border-border/40 bg-[#1a1a2e]/60 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-white/[0.06] bg-white/[0.03]">
        <span className="text-[10px] font-medium text-muted-foreground/70 uppercase tracking-wider">{language || 'code'}</span>
        <button
          type="button"
          onClick={handleCopy}
          className="inline-flex items-center gap-1 text-[10px] font-medium text-muted-foreground/60 hover:text-foreground/80 transition-colors"
          aria-label="Copy code"
        >
          {copied ? (
            <span className="text-success">Copied</span>
          ) : (
            <>
              <svg className="h-3 w-3" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="5" y="5" width="8" height="8" rx="1.5" />
                <path d="M3 11V3.5A1.5 1.5 0 0 1 4.5 2H11" />
              </svg>
              Copy
            </>
          )}
        </button>
      </div>
      <pre className="overflow-x-auto px-4 py-3 text-[13px] leading-[1.6] font-mono">
        <code dangerouslySetInnerHTML={{ __html: displayHtml }} />
      </pre>
    </div>
  )
})
