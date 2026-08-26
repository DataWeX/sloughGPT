import { describe, it, expect, afterEach } from 'vitest'
import { renderHook, act, cleanup } from '@testing-library/react'
import { useMessageThreads } from './useMessageThreads'
import type { ChatMessage } from '@/lib/chat-utils'

afterEach(cleanup)

const mockMessages: ChatMessage[] = [
  { id: '1', role: 'user', content: 'Hello', timestamp: Date.now() },
  { id: '2', role: 'assistant', content: 'Hi there!', timestamp: Date.now() },
]

describe('useMessageThreads', () => {
  it('initializes with empty threads', () => {
    const { result } = renderHook(() => useMessageThreads({ messages: mockMessages }))
    expect(result.current.threadCount).toBe(0)
    expect(result.current.threads).toEqual({})
  })

  it('creates a thread', () => {
    const { result } = renderHook(() => useMessageThreads({ messages: mockMessages }))
    
    let threadId: string
    act(() => {
      threadId = result.current.createThread('1')
    })

    expect(result.current.threadCount).toBe(1)
    expect(result.current.hasThread('1')).toBe(true)
    expect(result.current.getThread('1')).toBeDefined()
  })

  it('adds message to thread', () => {
    const { result } = renderHook(() => useMessageThreads({ messages: mockMessages }))
    
    let threadId: string
    act(() => {
      threadId = result.current.createThread('1')
    })

    const reply: ChatMessage = { id: '3', role: 'user', content: 'Reply', timestamp: Date.now() }
    act(() => {
      result.current.addToThread(threadId!, reply)
    })

    expect(result.current.getThreadMessages(threadId!)).toHaveLength(1)
    expect(result.current.getThreadMessages(threadId!)[0].content).toBe('Reply')
  })

  it('returns empty array for non-existent thread', () => {
    const { result } = renderHook(() => useMessageThreads({ messages: mockMessages }))
    expect(result.current.getThreadMessages('nonexistent')).toEqual([])
  })

  it('returns undefined for non-existent thread', () => {
    const { result } = renderHook(() => useMessageThreads({ messages: mockMessages }))
    expect(result.current.getThread('nonexistent')).toBeUndefined()
  })

  it('checks if message has thread', () => {
    const { result } = renderHook(() => useMessageThreads({ messages: mockMessages }))
    expect(result.current.hasThread('1')).toBe(false)
    
    act(() => {
      result.current.createThread('1')
    })
    
    expect(result.current.hasThread('1')).toBe(true)
  })
})