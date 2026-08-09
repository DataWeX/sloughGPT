import type { ImageAttachment } from './input/ImageUpload'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  images?: ImageAttachment[]
  isError?: boolean
}
