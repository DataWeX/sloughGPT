export interface MemoryImportEntry {
  content: string
  topic: string
}

export function parseMemoryImport(text: string, filename: string): MemoryImportEntry[] {
  const entries: MemoryImportEntry[] = []
  const ext = filename.split('.').pop()?.toLowerCase() || ''

  if (ext === 'json') {
    try {
      const data = JSON.parse(text)
      const items = Array.isArray(data) ? data : data.items || data.facts || data.records || []
      for (const item of items) {
        if (item.content || item.text || item.fact) {
          entries.push({
            content: item.content || item.text || item.fact,
            topic: item.topic || 'manual',
          })
        }
      }
    } catch {
      return []
    }
  } else if (ext === 'csv') {
    const lines = text.split('\n').filter(l => l.trim())
    if (lines.length === 0) return []
    const header = lines[0].toLowerCase()
    const hasHeader = header.includes('content') || header.includes('text') || header.includes('fact')
    const start = hasHeader ? 1 : 0
    for (let i = start; i < lines.length; i++) {
      const parts = lines[i].split(',')
      if (parts.length >= 1 && parts[0].trim()) {
        const content = parts[0].trim().replace(/^"(.*)"$/, '$1')
        const topic = parts.length >= 2 && parts[1].trim() ? parts[1].trim().replace(/^"(.*)"$/, '$1') : 'manual'
        entries.push({ content, topic })
      }
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
    'memory.store': 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300',
    'memory.remember': 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300',
    'memory.consolidate': 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300',
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
