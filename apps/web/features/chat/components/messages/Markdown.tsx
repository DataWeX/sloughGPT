'use client'

import { memo, useMemo, useRef, useState } from 'react'
import { cn } from '@sloughgpt/strui'
import Prism from 'prismjs'
import 'prismjs/components/prism-typescript'
import 'prismjs/components/prism-javascript'
import 'prismjs/components/prism-jsx'
import 'prismjs/components/prism-tsx'
import 'prismjs/components/prism-python'
import 'prismjs/components/prism-bash'
import 'prismjs/components/prism-json'
import 'prismjs/components/prism-markdown'
import 'prismjs/components/prism-yaml'
import 'prismjs/components/prism-rust'
import 'prismjs/components/prism-go'
import 'prismjs/components/prism-java'
import 'prismjs/components/prism-css'
import 'prismjs/components/prism-sql'
import 'prismjs/components/prism-c'
import 'prismjs/components/prism-cpp'

interface MarkdownProps {
  content: string
  className?: string
}

function InlineCode({ children }: { children: string }) {
  return (
    <code className="rounded bg-muted/50 px-1 py-0.5 font-mono text-[0.875em] break-all">
      {children}
    </code>
  )
}

/** Strip all HTML tags except safe span elements from Prism output. */
function sanitizeHtml(html: string): string {
  return html.replace(/<(?!\/?span\b|\/?code\b)[^>]*>/g, (tag) => {
    if (tag.startsWith('<script') || tag.startsWith('<iframe') || tag.startsWith('<object') || tag.startsWith('<embed')) return ''
    return tag.replace(/</g, '&lt;').replace(/>/g, '&gt;')
  })
}

function highlightCode(code: string, language: string): string {
  const lang = language.toLowerCase()
  const grammar = Prism.languages[lang]
  if (grammar) {
    return sanitizeHtml(Prism.highlight(code, grammar, lang))
  }
  return code
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

function CodeBlock({ language, code }: { language: string; code: string }) {
  const [copied, setCopied] = useState(false)
  const handleCopy = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }
  const highlighted = useMemo(() => highlightCode(code, language), [code, language])
  return (
    <div className="relative my-2 rounded-lg border border-border/50 bg-muted/30 overflow-hidden">
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-border/30 bg-muted/20">
        <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">{language || 'code'}</span>
        <button
          type="button"
          onClick={handleCopy}
          className="text-[10px] font-medium text-muted-foreground hover:text-foreground transition-colors"
          aria-label="Copy code"
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="overflow-x-auto p-3 text-xs leading-relaxed font-mono">
        {language ? (
          <code dangerouslySetInnerHTML={{ __html: highlighted }} />
        ) : (
          <code>{code}</code>
        )}
      </pre>
    </div>
  )
}

function parseMarkdown(text: string): React.ReactNode[] {
  const lines = text.split('\n')
  const nodes: React.ReactNode[] = []
  let i = 0
  let key = 0

  while (i < lines.length) {
    const line = lines[i]

    // Code block
    const codeMatch = line.match(/^```(\w*)/)
    if (codeMatch) {
      const lang = codeMatch[1]
      const codeLines: string[] = []
      i++
      while (i < lines.length && !lines[i].startsWith('```')) {
        codeLines.push(lines[i])
        i++
      }
      i++ // skip closing ```
      nodes.push(<CodeBlock key={key++} language={lang} code={codeLines.join('\n')} />)
      continue
    }

    // Horizontal rule
    if (/^---+\s*$/.test(line)) {
      nodes.push(<hr key={key++} className="my-3 border-border/40" />)
      i++
      continue
    }

    // Table detection: look for header row, separator row, then data rows
    if (line.includes('|') && i + 1 < lines.length && /^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/.test(lines[i + 1])) {
      const headerCells = line.split('|').map(c => c.trim()).filter(Boolean)
      i += 2 // skip header + separator
      const rows: string[][] = []
      while (i < lines.length && lines[i].includes('|') && lines[i].trim() !== '') {
        rows.push(lines[i].split('|').map(c => c.trim()).filter(Boolean))
        i++
      }
      nodes.push(
        <div key={key++} className="my-2 overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr>
                {headerCells.map((cell, ci) => (
                  <th key={ci} className="border border-border/40 bg-muted/30 px-2 py-1 text-left font-medium text-muted-foreground">
                    {parseInline(cell)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, ri) => (
                <tr key={ri}>
                  {row.map((cell, ci) => (
                    <td key={ci} className="border border-border/40 px-2 py-1">
                      {parseInline(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )
      continue
    }

    // Headings
    const headingMatch = line.match(/^(#{1,6})\s+(.+)/)
    if (headingMatch) {
      const level = headingMatch[1].length
      const headingText = headingMatch[2]
      const Tag = `h${level}` as keyof React.JSX.IntrinsicElements
      nodes.push(
        <Tag key={key++} className={cn(
          'font-semibold mt-3 mb-1',
          level === 1 && 'text-base',
          level === 2 && 'text-sm',
          level >= 3 && 'text-xs',
        )}>
          {parseInline(headingText)}
        </Tag>
      )
      i++
      continue
    }

    // Blockquote
    const bqMatch = line.match(/^>\s+(.*)/)
    if (bqMatch) {
      const bqLines: string[] = [bqMatch[1]]
      i++
      while (i < lines.length) {
        const m = lines[i].match(/^>\s?(.*)/)
        if (m) { bqLines.push(m[1]); i++ }
        else break
      }
      nodes.push(
        <blockquote key={key++} className="border-l-2 border-primary/30 pl-3 my-2 text-sm text-muted-foreground italic break-words">
          {bqLines.join('\n')}
        </blockquote>
      )
      continue
    }

    // Task list
    const taskMatch = line.match(/^[-*]\s+\[([ xX])\]\s+(.*)/)
    if (taskMatch) {
      const checked = taskMatch[1] !== ' '
      const items: React.ReactNode[] = []
      items.push(
        <li key={`li-0`} className="text-sm flex items-start gap-1.5">
          <span className={cn('mt-0.5 inline-flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded border', checked ? 'bg-primary border-primary text-primary-foreground' : 'border-muted-foreground/40')}>
            {checked && <span className="text-[10px]">✓</span>}
          </span>
          {parseInline(taskMatch[2])}
        </li>
      )
      i++
      let liKey = 1
      while (i < lines.length) {
        const m = lines[i].match(/^[-*]\s+\[([ xX])\]\s+(.*)/)
        if (m) {
          const c = m[1] !== ' '
          items.push(
            <li key={`li-${liKey++}`} className="text-sm flex items-start gap-1.5">
              <span className={cn('mt-0.5 inline-flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded border', c ? 'bg-primary border-primary text-primary-foreground' : 'border-muted-foreground/40')}>
                {c && <span className="text-[10px]">✓</span>}
              </span>
              {parseInline(m[2])}
            </li>
          )
          i++
        } else break
      }
      nodes.push(<ul key={key++} className="space-y-0.5 my-1">{items}</ul>)
      continue
    }

    // Unordered list
    const ulMatch = line.match(/^[-*]\s+(.*)/)
    if (ulMatch) {
      const items: React.ReactNode[] = []
      items.push(<li key={`li-0`} className="text-sm">{parseInline(ulMatch[1])}</li>)
      i++
      let liKey = 1
      while (i < lines.length) {
        const m = lines[i].match(/^[-*]\s+(.*)/)
        if (m) { items.push(<li key={`li-${liKey++}`} className="text-sm">{parseInline(m[1])}</li>); i++ }
        else break
      }
      nodes.push(<ul key={key++} className="list-disc list-inside space-y-0.5 my-1">{items}</ul>)
      continue
    }

    // Ordered list
    const olMatch = line.match(/^\d+\.\s+(.*)/)
    if (olMatch) {
      const items: React.ReactNode[] = []
      items.push(<li key={`li-0`} className="text-sm">{parseInline(olMatch[1])}</li>)
      i++
      let liKey = 1
      while (i < lines.length) {
        const m = lines[i].match(/^\d+\.\s+(.*)/)
        if (m) { items.push(<li key={`li-${liKey++}`} className="text-sm">{parseInline(m[1])}</li>); i++ }
        else break
      }
      nodes.push(<ol key={key++} className="list-decimal list-inside space-y-0.5 my-1">{items}</ol>)
      continue
    }

    // Empty line
    if (line.trim() === '') {
      i++
      continue
    }

    // Regular paragraph
    nodes.push(<p key={key++} className="text-sm leading-relaxed mb-1 break-words">{parseInline(line)}</p>)
    i++
  }

  return nodes
}

function parseInline(text: string): React.ReactNode[] {
  const parts: React.ReactNode[] = []
  let remaining = text
  let key = 0

  while (remaining.length > 0) {
    let match: RegExpMatchArray | null

    // Inline code
    match = remaining.match(/`(.+?)`/)
    if (match) {
      const idx = remaining.indexOf(match[0])
      if (idx > 0) {
        parts.push(remaining.slice(0, idx))
        remaining = remaining.slice(idx)
      }
      parts.push(<InlineCode key={key++}>{match[1]}</InlineCode>)
      remaining = remaining.slice(match[0].length)
      continue
    }

    // Bold
    match = remaining.match(/\*\*(.+?)\*\*/)
    if (match) {
      const idx = remaining.indexOf(match[0])
      if (idx > 0) {
        parts.push(remaining.slice(0, idx))
        remaining = remaining.slice(idx)
      }
      parts.push(<strong key={key++}>{match[1]}</strong>)
      remaining = remaining.slice(match[0].length)
      continue
    }

    // Strikethrough
    match = remaining.match(/~~(.+?)~~/)
    if (match) {
      const idx = remaining.indexOf(match[0])
      if (idx > 0) {
        parts.push(remaining.slice(0, idx))
        remaining = remaining.slice(idx)
      }
      parts.push(<del key={key++} className="text-muted-foreground/70">{match[1]}</del>)
      remaining = remaining.slice(match[0].length)
      continue
    }

    // Italic (must come after bold and strikethrough)
    match = remaining.match(/\*(.+?)\*/)
    if (match) {
      const idx = remaining.indexOf(match[0])
      if (idx > 0) {
        parts.push(remaining.slice(0, idx))
        remaining = remaining.slice(idx)
      }
      parts.push(<em key={key++}>{match[1]}</em>)
      remaining = remaining.slice(match[0].length)
      continue
    }

    // Link
    match = remaining.match(/\[(.+?)\]\((.+?)\)/)
    if (match) {
      const idx = remaining.indexOf(match[0])
      if (idx > 0) {
        parts.push(remaining.slice(0, idx))
        remaining = remaining.slice(idx)
      }
      parts.push(
        <a
          key={key++}
          href={match[2]}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary underline underline-offset-2 hover:text-primary/80"
        >
          {match[1]}
        </a>
      )
      remaining = remaining.slice(match[0].length)
      continue
    }

    parts.push(remaining)
    break
  }

  return parts
}

export const Markdown = memo(function Markdown({ content, className }: MarkdownProps) {
  const lastParsedRef = useRef(content)
  const lastRenderedRef = useRef<React.ReactNode[]>(parseMarkdown(content))
  const throttleRef = useRef(0)

  const rendered = useMemo(() => {
    const now = Date.now()
    const timeSinceLastParse = now - throttleRef.current

    // During streaming, throttle full re-parsing to every 300ms
    // Only re-parse immediately if content shrank (edit/regenerate) or 300ms elapsed
    const contentShrunk = content.length < lastParsedRef.current.length
    const throttled = !contentShrunk && timeSinceLastParse < 300 && lastParsedRef.current.length > 0

    if (throttled) {
      return lastRenderedRef.current
    }

    lastParsedRef.current = content
    throttleRef.current = now
    lastRenderedRef.current = parseMarkdown(content)
    return lastRenderedRef.current
  }, [content])

  return (
    <div className={cn("space-y-0 break-words", className)}>
      {rendered}
    </div>
  )
})
