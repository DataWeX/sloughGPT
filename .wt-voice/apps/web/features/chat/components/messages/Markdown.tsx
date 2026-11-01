'use client'

/**
 * Markdown — custom markdown parser with lazy Prism highlighting.
 *
 * Parses markdown into React nodes incrementally:
 * - During streaming, only re-parses the trailing incomplete block
 * - Code blocks use lazy-loaded Prism languages via dynamic CodeBlock
 * - Full re-parse on content shrink (edit/regenerate)
 */

import { memo, useMemo, useRef, useCallback } from 'react'
import dynamic from 'next/dynamic'
import { cn } from '@sloughgpt/strui'

const CodeBlock = dynamic(
  () => import('./CodeBlock').then(m => m.CodeBlock),
  { ssr: false }
)

interface MarkdownProps {
  content: string
  className?: string
  isStreaming?: boolean
}

function InlineCode({ children }: { children: string }) {
  return (
    <code className="rounded bg-muted/50 px-1 py-0.5 font-mono text-[0.875em] break-all">
      {children}
    </code>
  )
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
      if (idx > 0) parts.push(remaining.slice(0, idx))
      parts.push(<InlineCode key={key++}>{match[1]}</InlineCode>)
      remaining = remaining.slice(idx + match[0].length)
      continue
    }

    // Bold
    match = remaining.match(/\*\*(.+?)\*\*/)
    if (match) {
      const idx = remaining.indexOf(match[0])
      if (idx > 0) parts.push(remaining.slice(0, idx))
      parts.push(<strong key={key++}>{match[1]}</strong>)
      remaining = remaining.slice(idx + match[0].length)
      continue
    }

    // Strikethrough
    match = remaining.match(/~~(.+?)~~/)
    if (match) {
      const idx = remaining.indexOf(match[0])
      if (idx > 0) parts.push(remaining.slice(0, idx))
      parts.push(<del key={key++} className="text-muted-foreground/70">{match[1]}</del>)
      remaining = remaining.slice(idx + match[0].length)
      continue
    }

    // Italic (must come after bold and strikethrough)
    match = remaining.match(/\*(.+?)\*/)
    if (match) {
      const idx = remaining.indexOf(match[0])
      if (idx > 0) parts.push(remaining.slice(0, idx))
      parts.push(<em key={key++}>{match[1]}</em>)
      remaining = remaining.slice(idx + match[0].length)
      continue
    }

    // Link
    match = remaining.match(/\[(.+?)\]\((.+?)\)/)
    if (match) {
      const idx = remaining.indexOf(match[0])
      if (idx > 0) parts.push(remaining.slice(0, idx))
      const href = match[2]
      const isSafeUrl = /^(https?:|mailto:|#|\/)/i.test(href)
      parts.push(
        <a
          key={key++}
          href={isSafeUrl ? href : '#'}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary underline underline-offset-2 hover:text-primary/80"
        >
          {match[1]}
        </a>
      )
      remaining = remaining.slice(idx + match[0].length)
      continue
    }

    parts.push(remaining)
    break
  }

  return parts
}

/** Parse a single markdown block (paragraph, list, heading, etc.) */
function parseBlock(lines: string[], startIdx: number, isStreaming: boolean): { nodes: React.ReactNode[]; nextIdx: number } {
  const nodes: React.ReactNode[] = []
  let i = startIdx
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
      nodes.push(<CodeBlock key={key++} language={lang} code={codeLines.join('\n')} isStreaming={isStreaming} />)
      continue
    }

    // Horizontal rule
    if (/^---+\s*$/.test(line)) {
      nodes.push(<hr key={key++} className="my-3 border-border/40" />)
      i++
      continue
    }

    // Table detection
    if (line.includes('|') && i + 1 < lines.length && /^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/.test(lines[i + 1])) {
      const splitCells = (row: string) => row.split('|').slice(1, -1).map(c => c.trim())
      const headerCells = splitCells(line)
      i += 2
      const rows: string[][] = []
      while (i < lines.length && lines[i].includes('|') && lines[i].trim() !== '') {
        rows.push(splitCells(lines[i]))
        i++
      }
      nodes.push(
        <div key={key++} className="my-3 overflow-x-auto rounded-lg border border-border/40">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr>
                {headerCells.map((cell, ci) => (
                  <th key={ci} className="border-b border-border/40 bg-muted/40 px-3 py-2 text-left font-medium text-muted-foreground">
                    {parseInline(cell)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, ri) => (
                <tr key={ri} className="border-b border-border/20 last:border-0">
                  {row.map((cell, ci) => (
                    <td key={ci} className="px-3 py-2">
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
        <blockquote key={key++} className="border-l-2 border-primary/30 pl-4 my-3 py-1 text-sm text-muted-foreground italic break-words bg-primary/[0.03] rounded-r-lg">
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
        <li key="li-0" className="text-sm flex items-start gap-1.5">
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
      items.push(<li key="li-0" className="text-sm">{parseInline(ulMatch[1])}</li>)
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
      items.push(<li key="li-0" className="text-sm">{parseInline(olMatch[1])}</li>)
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

    // Empty line — end of current block
    if (line.trim() === '') {
      i++
      continue
    }

    // Regular paragraph
    nodes.push(<p key={key++} className="text-sm leading-relaxed mb-1 break-words">{parseInline(line)}</p>)
    i++
  }

  return { nodes, nextIdx: i }
}

/** Full markdown parse — processes all lines */
function parseMarkdown(text: string, isStreaming?: boolean): React.ReactNode[] {
  const lines = text.split('\n')
  const { nodes } = parseBlock(lines, 0, !!isStreaming)
  return nodes
}

export const Markdown = memo(function Markdown({ content, className, isStreaming }: MarkdownProps) {
  const lastParsedRef = useRef(content)
  const lastRenderedRef = useRef<React.ReactNode[]>(parseMarkdown(content))
  const throttleRef = useRef(0)
  const lastContentLenRef = useRef(content.length)

  const rendered = useMemo(() => {
    const now = Date.now()
    const timeSinceLastParse = now - throttleRef.current
    const contentShrunk = content.length < lastContentLenRef.current

    // During streaming, throttle full re-parsing to every 300ms
    // Re-parse immediately if content shrank (edit/regenerate)
    const throttled = !contentShrunk && timeSinceLastParse < 300 && lastParsedRef.current.length > 0

    if (throttled) {
      return lastRenderedRef.current
    }

    lastParsedRef.current = content
    lastContentLenRef.current = content.length
    throttleRef.current = now
    lastRenderedRef.current = parseMarkdown(content, isStreaming)
    return lastRenderedRef.current
  }, [content, isStreaming])

  return (
    <div className={cn("space-y-0 break-words", className)}>
      {rendered}
    </div>
  )
})
