import type { MemoryArchiveRecord } from '@/lib/memory-controller'

export interface MemoryImportEntry {
  content: string
  topic: string
}

export function parseMemoryImport(text: string, filename: string): MemoryImportEntry[] {
  const splitLines = (t: string) => t.split('\n').map(l => l.trim()).filter(Boolean)
  if (filename.toLowerCase().endsWith('.json')) {
    const parsed = JSON.parse(text)
    if (!Array.isArray(parsed)) return []
    return parsed.flatMap((entry: unknown) => {
      if (typeof entry === 'string') return entry.trim() ? [{ content: entry.trim(), topic: 'manual' }] : []
      if (entry && typeof entry === 'object') {
        const e = entry as Record<string, unknown>
        const content = typeof e.content === 'string' ? e.content.trim() : ''
        const topic = typeof e.topic === 'string' && e.topic.trim() ? e.topic.trim() : 'manual'
        return content ? [{ content, topic }] : []
      }
      return []
    })
  }
  if (filename.toLowerCase().endsWith('.csv')) {
    const lines = splitLines(text)
    if (lines.length === 0) return []
    const header = lines[0].toLowerCase()
    const contentIdx = header.split(',').findIndex(h => h.includes('content'))
    const topicIdx = header.split(',').findIndex(h => h.includes('topic'))
    return lines.slice(1).flatMap(line => {
      const cols = line.split(',').map(c => c.trim().replace(/^"|"$/g, ''))
      const content = contentIdx >= 0 ? cols[contentIdx] : cols[0]
      const topic = topicIdx >= 0 && cols[topicIdx] ? cols[topicIdx] : 'manual'
      return content ? [{ content, topic }] : []
    })
  }
  return splitLines(text).flatMap(line => {
    const match = line.match(/^(.*?)\s*\[([^\]]+)\]$/)
    if (match) return [{ content: match[1], topic: match[2] }]
    return [{ content: line, topic: 'manual' }]
  })
}

export function archiveTypeLabel(taskType: string): string {
  if (taskType === 'memory.remember') return 'remember'
  if (taskType === 'memory.store') return 'store'
  if (taskType === 'memory.consolidate') return 'consolidate'
  return (taskType || 'task').replace(/^memory\./, '')
}

export function archiveBadgeClass(taskType: string): string {
  if (taskType === 'memory.remember') return 'bg-primary/10 text-primary'
  if (taskType === 'memory.store') return 'bg-success/15 text-success'
  if (taskType === 'memory.consolidate') return 'bg-warning/15 text-warning'
  return 'bg-muted text-muted-foreground'
}

export function archiveSummary(record: MemoryArchiveRecord): { text: string; detail: string } {
  if (record.task_type === 'memory.remember') {
    return { text: String(record.user_message ?? ''), detail: 'Learned from a conversation' }
  }
  if (record.task_type === 'memory.store') {
    return { text: String(record.content ?? ''), detail: record.topic ? `Topic: ${record.topic}` : 'Stored fact' }
  }
  if (record.task_type === 'memory.consolidate') {
    const removed = Number(record.removed ?? 0)
    const kept = Number(record.kept ?? 0)
    return { text: `Consolidated ${removed} duplicate(s), kept ${kept}`, detail: `Threshold ${Number(record.threshold ?? 0.8).toFixed(2)}` }
  }
  const { ts: _ts, task_id: _id, ...rest } = record as Record<string, unknown>
  const snippet = JSON.stringify(rest)
  return { text: snippet && snippet !== '{}' ? snippet : archiveTypeLabel(record.task_type), detail: 'Task record' }
}
