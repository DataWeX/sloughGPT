import type { ImageAttachment } from './input/ImageUpload'
import type { AudioAttachment } from '@/lib/chat-utils'

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
  /** Per-turn stream metrics — live throttled ticks while streaming, final on completion. `thinking` marks pre-first-token reasoning ticks. */
  metrics?: { tokens?: number; elapsedMs?: number; maxTokens?: number; thinking?: boolean }
}
