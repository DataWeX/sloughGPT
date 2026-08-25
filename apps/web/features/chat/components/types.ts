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
  toolCalls?: Array<{ id?: string; name?: string; arguments?: string }>
}
