'use client'

export const CURRENT_SESSION_KEY = 'man_current_conversation'
export const USER_ID_KEY = 'man_user_id'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  images?: ImageAttachment[]
}

export interface ImageAttachment {
  id: string
  dataUrl: string
  name: string
}

export interface ChatSession {
  id: string
  name: string
  messages: ChatMessage[]
  createdAt: string
  updatedAt: string
  synced: boolean
  starred: boolean
  pinned: boolean
}

export interface DownloadProgressInfo {
  percentage: number
  status: string
  speed_mb_per_sec?: number
  eta_seconds?: number
  bytes_downloaded?: number
  total_bytes?: number
  current_file?: string
  files_completed?: number
  files_total?: number
}

export function cleanStreamedContent(text: string): string {
  if (!text) return text
  let cleaned = text.replace(/^\s*>\s*/gm, '')
  cleaned = cleaned.replace(/^(Assistant:\s*)+/i, '')
  cleaned = cleaned.replace(/^\s+/, '')
  return cleaned
}

export function stripAssistantPrefix(text: string): string {
  if (!text) return text
  const prefixes: RegExp[] = [
    /^Assistant:\s*/i,
    /^\n?Assistant:\s*/i,
    /^\s*Assistant:\s*/i,
    /^\s*>\s*Assistant:\s*/i,
  ]
  for (const prefix of prefixes) {
    if (prefix.test(text)) {
      text = text.replace(prefix, '')
    }
  }
  return text
}

export function getOrCreateUserId(): string {
  if (typeof window === 'undefined') return 'default'
  const existing = localStorage.getItem(USER_ID_KEY)
  if (existing) return existing
  const id = `user_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
  localStorage.setItem(USER_ID_KEY, id)
  return id
}

export function generateSessionId(): string {
  const chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
  let hash = ''
  for (let i = 0; i < 8; i++) {
    hash += chars[Math.floor(Math.random() * chars.length)]
  }
  return `chat_${hash}`
}

export function buildLocalPrompt(messages: ChatMessage[], systemPrompt: string): string {
  let prompt = systemPrompt ? `System: ${systemPrompt}\n` : ''
  for (const m of messages) {
    if (m.role === 'user') prompt += `User: ${m.content}\n`
    else prompt += `Assistant: ${m.content}\n`
  }
  prompt += 'Assistant:'
  return prompt
}

export function exportConversationAsMarkdown(messages: ChatMessage[]): void {
  let md = '# Chat Export\n\n'
  for (const m of messages) {
    const role = m.role === 'user' ? '**You**' : '**Assistant**'
    md += `### ${role}\n${m.content}\n\n`
  }
  const blob = new Blob([md], { type: 'text/markdown' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `chat-${new Date().toISOString().slice(0, 10)}.md`
  a.click()
  URL.revokeObjectURL(url)
}

export async function copyConversationAsMarkdown(messages: ChatMessage[]): Promise<boolean> {
  let md = ''
  for (const m of messages) {
    const role = m.role === 'user' ? 'You' : 'Assistant'
    md += `**${role}:**\n${m.content}\n\n`
  }
  try {
    await navigator.clipboard.writeText(md)
    return true
  } catch {
    return false
  }
}

export function computeSearchMatches(messages: ChatMessage[], query: string): { matchIds: string[]; matchCount: number } {
  if (!query) return { matchIds: [], matchCount: 0 }
  const q = query.toLowerCase()
  const matchIds = messages.filter(m => m.content.toLowerCase().includes(q)).map(m => m.id)
  return { matchIds, matchCount: matchIds.length }
}

export function formatSize(bytes: number): string {
  if (bytes >= 1073741824) return `${(bytes / 1073741824).toFixed(2)} GB`
  if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${bytes} B`
}

export function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400)
  const h = Math.floor((seconds % 86400) / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (d > 0) return `${d}d ${h}h ${m}m`
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

export function getErrorInfo(err: unknown): { title: string; description: string; retryable: boolean } {
  if (!err) return { title: 'Unknown Error', description: 'An unexpected error occurred.', retryable: false }
  const msg = err instanceof Error ? err.message : String(err)
  if (msg.includes('no model loaded') || msg.includes('model is not loaded')) {
    return { title: 'Model Not Loaded', description: 'No model is currently loaded. Select a model from the dropdown.', retryable: true }
  }
  if (msg.includes('Failed to fetch') || msg.includes('NetworkError') || msg.includes('ERR_CONNECTION_REFUSED')) {
    return { title: 'Connection Error', description: 'Cannot reach the server. Make sure the API is running.', retryable: true }
  }
  if (msg.includes('timeout') || msg.includes('timed out')) {
    return { title: 'Request Timeout', description: 'The request took too long. The model may be overloaded.', retryable: true }
  }
  if (msg.includes('rate limit') || msg.includes('too many requests')) {
    return { title: 'Rate Limited', description: 'Too many requests. Please wait a moment.', retryable: true }
  }
  if (msg.includes('401') || msg.includes('unauthorized') || msg.includes('Unauthorized')) {
    return { title: 'Unauthorized', description: 'Authentication failed. Please log in again.', retryable: false }
  }
  if (msg.includes('503') || msg.includes('Service Unavailable')) {
    return { title: 'Service Unavailable', description: 'The server is temporarily unavailable. Please try again.', retryable: true }
  }
  return { title: 'Error', description: msg, retryable: true }
}
