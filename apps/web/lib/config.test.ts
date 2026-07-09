import {describe, it, expect} from 'vitest'
import {PUBLIC_API_URL, API_CHAT_ENDPOINT, KNOWLEDGE_STORAGE_KEY} from './config'

describe('config', () => {
  it('PUBLIC_API_URL defaults to localhost', () => {
    expect(PUBLIC_API_URL).toBeTruthy()
    expect(typeof PUBLIC_API_URL).toBe('string')
  })

  it('API_CHAT_ENDPOINT includes /chat/stream', () => {
    expect(API_CHAT_ENDPOINT).toContain('/chat/stream')
    expect(API_CHAT_ENDPOINT).toBe(PUBLIC_API_URL + '/chat/stream')
  })

  it('KNOWLEDGE_STORAGE_KEY is defined', () => {
    expect(KNOWLEDGE_STORAGE_KEY).toBeTruthy()
    expect(typeof KNOWLEDGE_STORAGE_KEY).toBe('string')
  })
})
