export interface MemoryImportEntry {
  content: string
  topic: string
}

/**
 * Canonical memory-import parser (was duplicated in
 * components/knowledge/memory-card-utils.ts — merged here 2026-10-24).
 *
 * JSON: tolerant of malformed input (returns []), accepts top-level arrays
 * plus `{items|facts|records}` wrappers, string or object entries
 * (`content`/`text`/`fact` aliases), trims and defaults topic to `manual`.
 * CSV: a header row is detected by content/text/fact/topic column names and
 * indexed by position; without one every line is data.
 * Anything else (txt): one entry per line, `fact [topic]` bracket syntax.
 */
export function parseMemoryImport(text: string, filename: string): MemoryImportEntry[] {
  const entries: MemoryImportEntry[] = []
  const ext = filename.split('.').pop()?.toLowerCase() || ''
  const stripQuotes = (v: string) => v.trim().replace(/^"|"$/g, '')

  if (ext === 'json') {
    try {
      const data = JSON.parse(text)
      const items = Array.isArray(data) ? data : data.items || data.facts || data.records || []
      for (const item of items) {
        if (typeof item === 'string') {
          const content = item.trim()
          if (content) entries.push({ content, topic: 'manual' })
          continue
        }
        if (item && typeof item === 'object') {
          const content = (item.content || item.text || item.fact) ?? ''
          if (typeof content === 'string' && content.trim()) {
            entries.push({
              content: content.trim(),
              topic:
                typeof item.topic === 'string' && item.topic.trim() ? item.topic.trim() : 'manual',
            })
          }
        }
      }
    } catch {
      return []
    }
  } else if (ext === 'csv') {
    const lines = text.split('\n').filter((l) => l.trim())
    if (lines.length === 0) return []
    const header = lines[0].toLowerCase()
    const hasHeader =
      header.includes('content') ||
      header.includes('text') ||
      header.includes('fact') ||
      header.includes('topic')
    if (hasHeader) {
      const cols = lines[0].split(',').map((h) => h.toLowerCase())
      const contentIdx = cols.findIndex(
        (h) => h.includes('content') || h.includes('text') || h.includes('fact'),
      )
      const topicIdx = cols.findIndex((h) => h.includes('topic'))
      for (const line of lines.slice(1)) {
        const parts = line.split(',').map(stripQuotes)
        const content = contentIdx >= 0 ? parts[contentIdx] : parts[0]
        if (content && content.trim()) {
          entries.push({
            content: content.trim(),
            topic: (topicIdx >= 0 && parts[topicIdx]) || 'manual',
          })
        }
      }
    } else {
      for (const line of lines) {
        const parts = line.split(',').map(stripQuotes)
        if (parts[0]?.trim()) {
          entries.push({ content: parts[0].trim(), topic: parts[1]?.trim() || 'manual' })
        }
      }
    }
  } else {
    for (const line of text.split('\n')) {
      const trimmed = line.trim()
      if (!trimmed) continue
      const match = trimmed.match(/^(.*?)\s*\[([^\]]+)\]$/)
      if (match) entries.push({ content: match[1].trim(), topic: match[2].trim() })
      else entries.push({ content: trimmed, topic: 'manual' })
    }
  }

  return entries
}

export function archiveTypeLabel(taskType: string): string {
  const labels: Record<string, string> = {
    'memory.store': 'stored',
    'memory.remember': 'remember',
    'memory.consolidate': 'consolidate',
  }
  return labels[taskType] || taskType.replace(/^[^.]+\./, '').replace(/_/g, ' ')
}

export function archiveBadgeClass(taskType: string): string {
  const classes: Record<string, string> = {
    'memory.store': 'bg-success/15 text-success',
    'memory.remember': 'bg-primary/10 text-primary',
    'memory.consolidate': 'bg-warning/15 text-warning',
  }
  return classes[taskType] || 'bg-muted text-muted-foreground'
}

export function archiveSummary(record: Record<string, unknown>): { text: string; detail: string } {
  const taskType = record.task_type as string
  if (taskType === 'memory.remember') {
    const userMsg = (record.user_message as string) || ''
    const assistantMsg = (record.assistant_response as string) || ''
    const text = userMsg.length > 60 ? userMsg.slice(0, 60) + '…' : userMsg || '(no message)'
    const detail = assistantMsg.length > 80 ? assistantMsg.slice(0, 80) + '…' : assistantMsg
    return { text, detail }
  }
  if (taskType === 'memory.store') {
    const content = (record.content as string) || ''
    const topic = (record.topic as string) || ''
    const text = content.length > 60 ? content.slice(0, 60) + '…' : content || '(no content)'
    return { text, detail: topic ? `Topic: ${topic}` : '' }
  }
  if (taskType === 'memory.consolidate') {
    const removed = record.removed ?? 0
    const kept = record.kept ?? 0
    return { text: `Consolidated ${removed} duplicate(s), kept ${kept}`, detail: '' }
  }
  return { text: '', detail: '' }
}
