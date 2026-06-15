import type { ImageAttachment } from './ImageUpload'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  images?: ImageAttachment[]
}
