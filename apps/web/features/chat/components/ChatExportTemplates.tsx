'use client'

import { useState, useCallback, memo } from 'react'
import { Button, IconX, IconCheck, IconDownload } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import type { ChatMessage } from '@/lib/chat-utils'

interface ExportTemplate {
  id: string
  name: string
  format: 'markdown' | 'json' | 'csv' | 'html'
  includeMetadata: boolean
  includeTimestamps: boolean
  includeSystemMessages: boolean
  messageFilter: 'all' | 'user' | 'assistant' | 'custom'
}

interface ChatExportTemplatesProps {
  messages: ChatMessage[]
  sessionTitle?: string
  className?: string
}

const DEFAULT_TEMPLATES: ExportTemplate[] = [
  {
    id: 'full',
    name: 'Full Export',
    format: 'markdown',
    includeMetadata: true,
    includeTimestamps: true,
    includeSystemMessages: true,
    messageFilter: 'all',
  },
  {
    id: 'conversation',
    name: 'Conversation Only',
    format: 'markdown',
    includeMetadata: false,
    includeTimestamps: false,
    includeSystemMessages: false,
    messageFilter: 'all',
  },
  {
    id: 'code',
    name: 'Code Snippets',
    format: 'markdown',
    includeMetadata: false,
    includeTimestamps: false,
    includeSystemMessages: false,
    messageFilter: 'assistant',
  },
  {
    id: 'data',
    name: 'Structured Data',
    format: 'json',
    includeMetadata: true,
    includeTimestamps: true,
    includeSystemMessages: true,
    messageFilter: 'all',
  },
]

const STORAGE_KEY = 'chat-export-templates'

function loadTemplates(): ExportTemplate[] {
  if (typeof window === 'undefined') return DEFAULT_TEMPLATES
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : DEFAULT_TEMPLATES
  } catch {
    return DEFAULT_TEMPLATES
  }
}

function saveTemplates(templates: ExportTemplate[]) {
  if (typeof window === 'undefined') return
  localStorage.setItem(STORAGE_KEY, JSON.stringify(templates))
}

function filterMessages(messages: ChatMessage[], template: ExportTemplate): ChatMessage[] {
  return messages.filter(msg => {
    if (template.messageFilter === 'user' && msg.role !== 'user') return false
    if (template.messageFilter === 'assistant' && msg.role !== 'assistant') return false
    return true
  })
}

function formatMarkdown(messages: ChatMessage[], template: ExportTemplate, title?: string): string {
  const lines: string[] = []

  if (template.includeMetadata && title) {
    lines.push(`# ${title}`)
    lines.push('')
    lines.push(`Exported: ${new Date().toLocaleString()}`)
    lines.push(`Messages: ${messages.length}`)
    lines.push('')
    lines.push('---')
    lines.push('')
  }

  for (const msg of messages) {
    const role = msg.role === 'user' ? 'You' : msg.role === 'assistant' ? 'Assistant' : 'System'
    const timestamp = template.includeTimestamps
      ? ` *(${new Date(msg.timestamp).toLocaleString()})*`
      : ''

    lines.push(`**${role}**${timestamp}`)
    lines.push('')
    lines.push(msg.content)
    lines.push('')
    lines.push('---')
    lines.push('')
  }

  return lines.join('\n')
}

function formatJSON(messages: ChatMessage[], template: ExportTemplate, title?: string): string {
  const data = {
    metadata: template.includeMetadata ? {
      title: title || 'Chat Export',
      exportedAt: new Date().toISOString(),
      messageCount: messages.length,
    } : undefined,
    messages: messages.map(msg => ({
      role: msg.role,
      content: msg.content,
      timestamp: template.includeTimestamps ? new Date(msg.timestamp).toISOString() : undefined,
    })),
  }
  return JSON.stringify(data, null, 2)
}

function formatCSV(messages: ChatMessage[], template: ExportTemplate): string {
  const rows: string[] = []
  const header = template.includeTimestamps
    ? 'Role,Content,Timestamp'
    : 'Role,Content'
  rows.push(header)

  for (const msg of messages) {
    const content = msg.content.replace(/"/g, '""')
    const row = template.includeTimestamps
      ? `"${msg.role}","${content}","${new Date(msg.timestamp).toISOString()}"`
      : `"${msg.role}","${content}"`
    rows.push(row)
  }

  return rows.join('\n')
}

function formatHTML(messages: ChatMessage[], template: ExportTemplate, title?: string): string {
  const lines: string[] = [
    '<!DOCTYPE html>',
    '<html lang="en">',
    '<head>',
    '  <meta charset="UTF-8">',
    '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
    `  <title>${title || 'Chat Export'}</title>`,
    '  <style>',
    '    body { font-family: system-ui; max-width: 800px; margin: 0 auto; padding: 20px; }',
    '    .message { margin: 16px 0; padding: 12px; border-radius: 8px; }',
    '    .user { background: #e3f2fd; }',
    '    .assistant { background: #f5f5f5; }',
    '    .system { background: #fff3e0; }',
    '    .meta { font-size: 12px; color: #666; margin-bottom: 4px; }',
    '    pre { white-space: pre-wrap; }',
    '  </style>',
    '</head>',
    '<body>',
  ]

  if (template.includeMetadata && title) {
    lines.push(`  <h1>${title}</h1>`)
    lines.push(`  <p>Exported: ${new Date().toLocaleString()}</p>`)
  }

  for (const msg of messages) {
    const role = msg.role
    const timestamp = template.includeTimestamps
      ? `<div class="meta">${new Date(msg.timestamp).toLocaleString()}</div>`
      : ''
    lines.push(`  <div class="message ${role}">`)
    lines.push(`    <strong>${role.charAt(0).toUpperCase() + role.slice(1)}</strong>`)
    lines.push(timestamp)
    lines.push(`    <pre>${msg.content}</pre>`)
    lines.push('  </div>')
  }

  lines.push('</body>')
  lines.push('</html>')
  return lines.join('\n')
}

export const ChatExportTemplates = memo(function ChatExportTemplates({
  messages,
  sessionTitle,
  className,
}: ChatExportTemplatesProps) {
  const [templates, setTemplates] = useState<ExportTemplate[]>(loadTemplates)
  const [selectedId, setSelectedId] = useState<string>('full')
  const [showCustom, setShowCustom] = useState(false)
  const [customName, setCustomName] = useState('')
  const [customFormat, setCustomFormat] = useState<ExportTemplate['format']>('markdown')

  const selectedTemplate = templates.find(t => t.id === selectedId)

  const handleExport = useCallback(() => {
    if (!selectedTemplate) return

    const filtered = filterMessages(messages, selectedTemplate)
    let content: string
    let filename: string
    let mimeType: string

    switch (selectedTemplate.format) {
      case 'json':
        content = formatJSON(filtered, selectedTemplate, sessionTitle)
        filename = `chat-export.json`
        mimeType = 'application/json'
        break
      case 'csv':
        content = formatCSV(filtered, selectedTemplate)
        filename = `chat-export.csv`
        mimeType = 'text/csv'
        break
      case 'html':
        content = formatHTML(filtered, selectedTemplate, sessionTitle)
        filename = `chat-export.html`
        mimeType = 'text/html'
        break
      default:
        content = formatMarkdown(filtered, selectedTemplate, sessionTitle)
        filename = `chat-export.md`
        mimeType = 'text/markdown'
    }

    const blob = new Blob([content], { type: mimeType })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [selectedTemplate, messages, sessionTitle])

  const handleCopy = useCallback(async () => {
    if (!selectedTemplate) return

    const filtered = filterMessages(messages, selectedTemplate)
    let content: string

    switch (selectedTemplate.format) {
      case 'json':
        content = formatJSON(filtered, selectedTemplate, sessionTitle)
        break
      case 'csv':
        content = formatCSV(filtered, selectedTemplate)
        break
      case 'html':
        content = formatHTML(filtered, selectedTemplate, sessionTitle)
        break
      default:
        content = formatMarkdown(filtered, selectedTemplate, sessionTitle)
    }

    await navigator.clipboard.writeText(content)
  }, [selectedTemplate, messages, sessionTitle])

  const handleSaveCustom = useCallback(() => {
    const trimmed = customName.trim()
    if (!trimmed) return

    const newTemplate: ExportTemplate = {
      id: crypto.randomUUID(),
      name: trimmed,
      format: customFormat,
      includeMetadata: true,
      includeTimestamps: true,
      includeSystemMessages: true,
      messageFilter: 'all',
    }

    const next = [...templates, newTemplate]
    setTemplates(next)
    saveTemplates(next)
    setSelectedId(newTemplate.id)
    setCustomName('')
    setShowCustom(false)
  }, [customName, customFormat, templates])

  const handleDeleteTemplate = useCallback((id: string) => {
    const next = templates.filter(t => t.id !== id)
    setTemplates(next)
    saveTemplates(next)
    if (selectedId === id) {
      setSelectedId(next[0]?.id || '')
    }
  }, [templates, selectedId])

  return (
    <div className={cn('border rounded-lg bg-card overflow-hidden', className)}>
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30">
        <div className="flex items-center gap-2">
          <IconDownload className="h-3 w-3 text-muted-foreground" />
          <span className="text-xs font-medium">Export Templates</span>
        </div>
        <Button
          variant="ghost"
          size="icon-sm"
          className="h-5 w-5"
          onClick={() => setShowCustom(!showCustom)}
          aria-label="Create template"
        >
          <IconDownload className="h-3 w-3" />
        </Button>
      </div>

      {showCustom && (
        <div className="p-2 border-b space-y-2">
          <input
            type="text"
            value={customName}
            onChange={(e) => setCustomName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSaveCustom()}
            placeholder="Template name..."
            className="w-full text-xs bg-transparent border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary/50"
          />
          <select
            value={customFormat}
            onChange={(e) => setCustomFormat(e.target.value as ExportTemplate['format'])}
            className="w-full text-xs bg-transparent border rounded px-2 py-1"
          >
            <option value="markdown">Markdown</option>
            <option value="json">JSON</option>
            <option value="csv">CSV</option>
            <option value="html">HTML</option>
          </select>
          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="sm"
              className="text-[10px] h-6"
              onClick={handleSaveCustom}
              disabled={!customName.trim()}
            >
              <IconCheck className="h-3 w-3 mr-1" />
              Save
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="text-[10px] h-6"
              onClick={() => setShowCustom(false)}
            >
              Cancel
            </Button>
          </div>
        </div>
      )}

      <div className="max-h-[300px] overflow-y-auto">
        {templates.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">No templates</p>
        ) : (
          <div className="divide-y">
            {templates.map(template => (
              <div
                key={template.id}
                className={cn(
                  'flex items-center gap-2 px-3 py-2 hover:bg-muted/30 group',
                  selectedId === template.id && 'bg-primary/5',
                )}
              >
                <button
                  type="button"
                  className="flex-1 text-left min-w-0"
                  onClick={() => setSelectedId(template.id)}
                >
                  <div className="text-xs font-medium">{template.name}</div>
                  <div className="text-[10px] text-muted-foreground uppercase">
                    {template.format} · {template.messageFilter}
                  </div>
                </button>
                {!DEFAULT_TEMPLATES.find(t => t.id === template.id) && (
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    className="h-5 w-5 opacity-0 group-hover:opacity-100"
                    onClick={() => handleDeleteTemplate(template.id)}
                    title="Delete template"
                  >
                    <IconX className="h-3 w-3" />
                  </Button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="p-2 border-t flex gap-1">
        <Button
          variant="ghost"
          size="sm"
          className="text-[10px] h-6 flex-1"
          onClick={handleExport}
          disabled={!selectedTemplate || messages.length === 0}
        >
          <IconDownload className="h-3 w-3 mr-1" />
          Download
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="text-[10px] h-6 flex-1"
          onClick={handleCopy}
          disabled={!selectedTemplate || messages.length === 0}
        >
          Copy
        </Button>
      </div>
    </div>
  )
})