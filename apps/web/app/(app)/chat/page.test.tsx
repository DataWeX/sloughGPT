import { describe, it, expect } from 'vitest'
import * as page from './page'
import ChatPage from '@/features/chat/ChatPage'

describe('chat route page', () => {
  it('stays force-dynamic for streaming', () => {
    expect(page.dynamic).toBe('force-dynamic')
  })

  it('re-exports the chat feature page as default', () => {
    expect(page.default).toBe(ChatPage)
  })
})
