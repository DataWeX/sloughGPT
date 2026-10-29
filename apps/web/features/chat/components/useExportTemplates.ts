'use client'

import { useState, useCallback } from 'react'
import type { ChatMessage } from '@/lib/chat-utils'

export interface ExportTemplate {
  id: string
  name: string
  format: 'markdown' | 'json' | 'csv' | 'html'
  includeMetadata: boolean
  includeTimestamps: boolean
  includeSystemMessages: boolean
  messageFilter: 'all' | 'user' | 'assistant' | 'custom'
}

const STORAGE_KEY = 'chat-export-templates'

export const DEFAULT_TEMPLATES: ExportTemplate[] = [
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

export function loadTemplates(): ExportTemplate[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return [...DEFAULT_TEMPLATES]
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return [...DEFAULT_TEMPLATES]
    return parsed
  } catch {
    return [...DEFAULT_TEMPLATES]
  }
}

export function saveTemplates(templates: ExportTemplate[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(templates))
  } catch {
    // ignore
  }
}

export function filterMessages(messages: ChatMessage[], template: ExportTemplate): ChatMessage[] {
  return messages.filter(msg => {
    if (template.messageFilter === 'all') return true
    if (template.messageFilter === 'user') return msg.role === 'user'
    if (template.messageFilter === 'assistant') return msg.role === 'assistant'
    return true
  })
}

export function formatMarkdown(messages: ChatMessage[], template: ExportTemplate, title?: string): string {
  const lines: string[] = []
  if (template.includeMetadata && title) {
    lines.push(`# ${title}`)
    lines.push('')
    lines.push(`*Exported ${new Date().toLocaleString()}*`)
    lines.push('')
  }
  for (const msg of messages) {
    const role = msg.role === 'user' ? 'You' : msg.role === 'assistant' ? 'Assistant' : 'System'
    const timestamp = template.includeTimestamps && msg.timestamp
      ? ` _${new Date(msg.timestamp).toLocaleString()}_`
      : ''
    lines.push(`### ${role}${timestamp}`)
    lines.push('')
    lines.push(msg.content)
    lines.push('')
  }
  return lines.join('\n')
}

export function formatJSON(messages: ChatMessage[], template: ExportTemplate, title?: string): string {
  const data = {
    ...(template.includeMetadata && { title, exportedAt: new Date().toISOString() }),
    messages: messages.map(msg => ({
      role: msg.role,
      content: msg.content,
      ...(template.includeTimestamps && msg.timestamp && { timestamp: new Date(msg.timestamp).toISOString() }),
    })),
  }
  return JSON.stringify(data, null, 2)
}

export function formatCSV(messages: ChatMessage[], template: ExportTemplate): string {
  const rows: string[] = []
  const header = template.includeTimestamps ? 'role,content,timestamp' : 'role,content'
  rows.push(header)
  for (const msg of messages) {
    const content = msg.content.replace(/"/g, '""')
    const row = template.includeTimestamps && msg.timestamp
      ? `"${msg.role}","${content}","${new Date(msg.timestamp).toISOString()}"`
      : `"${msg.role}","${content}"`
    rows.push(row)
  }
  return rows.join('\n')
}

export function formatHTML(messages: ChatMessage[], template: ExportTemplate, title?: string): string {
  const lines: string[] = [
    '<!DOCTYPE html>',
    '<html><head><meta charset="utf-8">',
    `<title>${title || 'Chat Export'}</title>`,
    '<style>body{font-family:sans-serif;max-width:800px;margin:0 auto;padding:20px}',
    '.message{margin:16px 0;padding:12px;border-radius:8px}',
    '.user{background:#f0f0f0}',
    '.assistant{background:#e8f4fd}',
    '.system{background:#fff3cd}',
    '.meta{font-size:12px;color:#666;margin-bottom:4px}</style>',
    '</head><body>',
  ]
  if (template.includeMetadata && title) {
    lines.push(`<h1>${title}</h1>`)
    lines.push(`<p>Exported ${new Date().toLocaleString()}</p>`)
  }
  for (const msg of messages) {
    const roleClass = msg.role
    const timestamp = template.includeTimestamps && msg.timestamp
      ? `<div class="meta">${new Date(msg.timestamp).toLocaleString()}</div>`
      : ''
    lines.push(`<div class="message ${roleClass}">`)
    lines.push(timestamp)
    lines.push(`<div>${msg.content.replace(/\n/g, '<br>')}</div>`)
    lines.push('</div>')
  }
  lines.push('</body></html>')
  return lines.join('\n')
}

export interface UseExportTemplatesReturn {
  templates: ExportTemplate[]
  selectedId: string | null
  selectedTemplate: ExportTemplate | null
  showCustom: boolean
  customName: string
  customFormat: ExportTemplate['format']
  setSelectedId: (id: string) => void
  setShowCustom: (v: boolean) => void
  setCustomName: (v: string) => void
  setCustomFormat: (f: ExportTemplate['format']) => void
  handleSaveCustom: () => void
  handleDeleteTemplate: (id: string) => void
  handleExport: () => void
  handleCopy: () => void
}

export function useExportTemplates(
  messages: ChatMessage[],
  sessionTitle?: string
): UseExportTemplatesReturn {
  const [templates, setTemplates] = useState<ExportTemplate[]>(() => loadTemplates())
  const [selectedId, setSelectedId] = useState<string | null>('full')
  const [showCustom, setShowCustom] = useState(false)
  const [customName, setCustomName] = useState('')
  const [customFormat, setCustomFormat] = useState<ExportTemplate['format']>('markdown')

  const selectedTemplate = templates.find(t => t.id === selectedId) || null

  const handleSaveCustom = useCallback(() => {
    if (!customName.trim()) return
    const newTemplate: ExportTemplate = {
      id: `custom-${Date.now()}`,
      name: customName.trim(),
      format: customFormat,
      includeMetadata: true,
      includeTimestamps: true,
      includeSystemMessages: true,
      messageFilter: 'all',
    }
    const updated = [...templates, newTemplate]
    setTemplates(updated)
    saveTemplates(updated)
    setSelectedId(newTemplate.id)
    setCustomName('')
    setShowCustom(false)
  }, [customName, customFormat, templates])

  const handleDeleteTemplate = useCallback((id: string) => {
    const updated = templates.filter(t => t.id !== id)
    setTemplates(updated)
    saveTemplates(updated)
    if (selectedId === id) setSelectedId('full')
  }, [templates, selectedId])

  const handleExport = useCallback(() => {
    if (!selectedTemplate || messages.length === 0) return
    const filtered = filterMessages(messages, selectedTemplate)
    let content: string
    let filename: string
    let mimeType: string

    switch (selectedTemplate.format) {
      case 'markdown':
        content = formatMarkdown(filtered, selectedTemplate, sessionTitle)
        filename = `${sessionTitle || 'chat'}.md`
        mimeType = 'text/markdown'
        break
      case 'json':
        content = formatJSON(filtered, selectedTemplate, sessionTitle)
        filename = `${sessionTitle || 'chat'}.json`
        mimeType = 'application/json'
        break
      case 'csv':
        content = formatCSV(filtered, selectedTemplate)
        filename = `${sessionTitle || 'chat'}.csv`
        mimeType = 'text/csv'
        break
      case 'html':
        content = formatHTML(filtered, selectedTemplate, sessionTitle)
        filename = `${sessionTitle || 'chat'}.html`
        mimeType = 'text/html'
        break
    }

    const blob = new Blob([content], { type: mimeType })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }, [selectedTemplate, messages, sessionTitle])

  const handleCopy = useCallback(async () => {
    if (!selectedTemplate || messages.length === 0) return
    const filtered = filterMessages(messages, selectedTemplate)
    let content: string

    switch (selectedTemplate.format) {
      case 'markdown':
        content = formatMarkdown(filtered, selectedTemplate, sessionTitle)
        break
      case 'json':
        content = formatJSON(filtered, selectedTemplate, sessionTitle)
        break
      case 'csv':
        content = formatCSV(filtered, selectedTemplate)
        break
      case 'html':
        content = formatHTML(filtered, selectedTemplate, sessionTitle)
        break
    }

    await navigator.clipboard.writeText(content)
  }, [selectedTemplate, messages, sessionTitle])

  return {
    templates, selectedId, selectedTemplate,
    showCustom, customName, customFormat,
    setSelectedId, setShowCustom, setCustomName, setCustomFormat,
    handleSaveCustom, handleDeleteTemplate, handleExport, handleCopy,
  }
}
