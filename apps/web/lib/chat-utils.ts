'use client'

import { chatDB } from '@/lib/db'
import { todayDateString } from '@/lib/format-bytes'

export const CURRENT_SESSION_KEY = 'man_current_conversation'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  images?: ImageAttachment[]
  audio?: AudioAttachment
  reactions?: Record<string, number>
  isError?: boolean
  pinned?: boolean
  toolCalls?: Array<{ id?: string; name?: string; arguments?: string; status?: string }>
}

export interface ImageAttachment {
  id: string
  dataUrl: string
  name: string
}

export interface AudioAttachment {
  id: string
  url: string
  durationMs: number
  waveform?: number[]
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
  archived?: boolean
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

const USER_ID_KEY = 'man_user_id'

export async function getOrCreateUserId(): Promise<string> {
  if (typeof window === 'undefined') return 'default'
  const existing = await chatDB.getKV<string>(USER_ID_KEY)
  if (existing) return existing
  const id = `user_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
  await chatDB.setKV(USER_ID_KEY, id)
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
  a.download = `chat-${todayDateString()}.md`
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

export function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400)
  const h = Math.floor((seconds % 86400) / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (d > 0) return `${d}d ${h}h ${m}m`
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

export function shortModelName(id: string): string {
  return id.includes('/') ? id.split('/').pop() || id : id
}
