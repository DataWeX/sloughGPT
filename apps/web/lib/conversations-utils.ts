export function formatDate(dateStr: string | undefined): string {
  if (!dateStr) return ''
  const date = new Date(dateStr)
  if (isNaN(date.getTime())) return ''
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)
  if (diffMins < 1) return 'Just now'
  if (diffMins < 60) return `${diffMins}m`
  if (diffHours < 24) return `${diffHours}h`
  if (diffDays < 7) return `${diffDays}d`
  return date.toLocaleDateString()
}

export function truncateMessage(content: string, maxLen = 60): string {
  if (!content) return ''
  const firstLine = content.split('\n')[0]
  return firstLine.length > maxLen ? firstLine.slice(0, maxLen) + '…' : firstLine
}

export function parseConversationJSON(data: any): { name: string; messages: { role: string; content: string }[] }[] {
  const arr = Array.isArray(data) ? data : [data]
  return arr.flatMap((item: any) => {
    if (!item.messages || !Array.isArray(item.messages)) return []
    return [{
      name: item.name || item.id || `Imported ${new Date().toLocaleDateString()}`,
      messages: item.messages.map((m: any) => ({
        role: m.role === 'user' || m.role === 'assistant' ? m.role : 'user',
        content: typeof m.content === 'string' ? m.content : '',
      })),
    }]
  })
}

export function parseConversationMD(text: string): { name: string; messages: { role: string; content: string }[] }[] {
  const blocks = text.split(/(?=^# )/m)
  return blocks.filter(b => b.trim()).map(block => {
    const lines = block.split('\n')
    const name = lines[0].replace(/^#\s*/, '').trim() || 'Imported'
    const messages: { role: string; content: string }[] = []
    let currentRole: 'user' | 'assistant' | null = null
    let currentContent: string[] = []
    for (const line of lines.slice(1)) {
      const userMatch = line.match(/^\*\*(user|User)\*\*:\s*(.*)/)
      const asstMatch = line.match(/^\*\*(assistant|Assistant)\*\*:\s*(.*)/)
      if (userMatch || asstMatch) {
        if (currentRole && currentContent.length > 0) {
          messages.push({ role: currentRole, content: currentContent.join('\n').trim() })
        }
        currentRole = userMatch ? 'user' : 'assistant'
        currentContent = [userMatch ? userMatch[2] : asstMatch![2]]
      } else if (currentRole) {
        currentContent.push(line)
      }
    }
    if (currentRole && currentContent.length > 0) {
      messages.push({ role: currentRole, content: currentContent.join('\n').trim() })
    }
    return { name, messages }
  }).filter(c => c.messages.length > 0)
}
