import { describe, it, expect } from 'vitest'
import {
  formatHTML,
  formatMarkdown,
  formatJSON,
  formatCSV,
  filterMessages,
  DEFAULT_TEMPLATES,
  type ExportTemplate,
} from './useExportTemplates'
import type { ChatMessage } from '@/lib/chat-utils'

const template = (over: Partial<ExportTemplate> = {}): ExportTemplate => ({
  id: 't',
  name: 'Test',
  format: 'html',
  includeMetadata: true,
  includeTimestamps: true,
  includeSystemMessages: true,
  messageFilter: 'all',
  ...over,
})

const messages: ChatMessage[] = [
  { id: 'm1', role: 'user', content: 'hello', timestamp: new Date(1700000000000) },
  { id: 'm2', role: 'assistant', content: 'hi\nthere', timestamp: new Date(1700000001000) },
]

describe('formatHTML', () => {
  it('emits a standalone document with no raw hex colours', () => {
    const html = formatHTML(messages, template(), 'My Chat')
    expect(html).toContain('<!DOCTYPE html>')
    expect(html).toContain('<title>My Chat</title>')
    expect(html).not.toMatch(/#[0-9a-fA-F]{3,8}\b/)
  })

  it('references design tokens with numeric fallbacks', () => {
    const html = formatHTML(messages, template())
    expect(html).toContain('rgb(var(--muted,')
    expect(html).toContain('rgb(var(--secondary,')
    expect(html).toContain('rgb(var(--accent,')
    expect(html).toContain('rgb(var(--muted-foreground,')
  })

  it('escapes newlines in message bodies', () => {
    const html = formatHTML(messages, template({ includeMetadata: false }))
    expect(html).toContain('hi<br>there')
  })

  it('honours includeTimestamps=false', () => {
    const html = formatHTML(messages, template({ includeTimestamps: false }))
    expect(html).not.toContain('class="meta"')
  })
})

describe('formatters', () => {
  it('formatMarkdown emits role headings', () => {
    const md = formatMarkdown(messages, template({ format: 'markdown' }), 'T')
    expect(md).toContain('# T')
    expect(md).toContain('### You')
    expect(md).toContain('### Assistant')
  })

  it('formatJSON round-trips roles', () => {
    const data = JSON.parse(formatJSON(messages, template({ format: 'json' })))
    expect(data.messages.map((m: { role: string }) => m.role)).toEqual(['user', 'assistant'])
  })

  it('formatCSV escapes quotes', () => {
    const csv = formatCSV(
      [{ id: 'm3', role: 'user', content: 'say "hi"', timestamp: new Date(1700000002000) }],
      template({ format: 'csv', includeTimestamps: false }),
    )
    expect(csv).toContain('"say ""hi"""')
  })

  it('filterMessages applies the role filter', () => {
    expect(filterMessages(messages, template({ messageFilter: 'user' }))).toHaveLength(1)
    expect(filterMessages(messages, template({ messageFilter: 'assistant' }))).toHaveLength(1)
    expect(filterMessages(messages, template({ messageFilter: 'all' }))).toHaveLength(2)
  })

  it('ships four default templates', () => {
    expect(DEFAULT_TEMPLATES.map((t) => t.format)).toEqual([
      'markdown',
      'markdown',
      'markdown',
      'json',
    ])
  })
})
