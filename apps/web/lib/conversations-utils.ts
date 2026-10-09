import { formatShortRelative } from './time-format'

export function formatDate(dateStr: string | undefined): string {
  return formatShortRelative(dateStr)
}

export function truncate(text: string, maxLen = 80): string {
  return text.length > maxLen ? text.slice(0, maxLen) + '…' : text
}

export function truncateMessage(content: string, maxLen = 60): string {
  if (!content) return 'Empty conversation'
  const firstLine = content.split('\n')[0]
  return truncate(firstLine, maxLen)
}
