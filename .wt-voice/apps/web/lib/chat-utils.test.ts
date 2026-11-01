import { describe, expect, it } from 'vitest'

import {
  cleanStreamedContent,
  stripAssistantPrefix,
  computeSearchMatches,
  formatUptime,
  buildLocalPrompt,
} from './chat-utils'

describe('cleanStreamedContent', () => {
  it('removes leading > quote markers', () => {
    expect(cleanStreamedContent('> hello\n> world')).toBe('hello\nworld')
  })

  it('removes Assistant: prefixes', () => {
    expect(cleanStreamedContent('Assistant: Hello there')).toBe('Hello there')
  })

  it('removes multiple Assistant: prefixes', () => {
    expect(cleanStreamedContent('Assistant: Assistant: Hi')).toBe('Hi')
  })

  it('trims leading whitespace', () => {
    expect(cleanStreamedContent('  hello')).toBe('hello')
  })

  it('returns empty string unchanged', () => {
    expect(cleanStreamedContent('')).toBe('')
  })

  it('handles nullish via falsy guard', () => {
    expect(cleanStreamedContent('')).toBe('')
  })

  it('preserves content with no markers', () => {
    expect(cleanStreamedContent('plain text')).toBe('plain text')
  })
})

describe('stripAssistantPrefix', () => {
  it('strips Assistant: prefix', () => {
    expect(stripAssistantPrefix('Assistant: Hello')).toBe('Hello')
  })

  it('strips newline-prefixed Assistant:', () => {
    expect(stripAssistantPrefix('\nAssistant: Hello')).toBe('Hello')
  })

  it('strips whitespace-prefixed Assistant:', () => {
    expect(stripAssistantPrefix('  Assistant: Hello')).toBe('Hello')
  })

  it('strips quote-prefixed Assistant:', () => {
    expect(stripAssistantPrefix('> Assistant: Hello')).toBe('Hello')
  })

  it('returns text without prefix unchanged', () => {
    expect(stripAssistantPrefix('Hello')).toBe('Hello')
  })

  it('returns empty string unchanged', () => {
    expect(stripAssistantPrefix('')).toBe('')
  })
})

describe('computeSearchMatches', () => {
  const msgs = [
    { id: '1', role: 'user' as const, content: 'Hello world', timestamp: new Date() },
    { id: '2', role: 'assistant' as const, content: 'Hi there', timestamp: new Date() },
    { id: '3', role: 'user' as const, content: 'How are you?', timestamp: new Date() },
  ]

  it('finds matching messages', () => {
    const r = computeSearchMatches(msgs, 'hello')
    expect(r.matchIds).toEqual(['1'])
    expect(r.matchCount).toBe(1)
  })

  it('is case-insensitive', () => {
    const r = computeSearchMatches(msgs, 'HELLO')
    expect(r.matchCount).toBe(1)
  })

  it('returns empty for no match', () => {
    const r = computeSearchMatches(msgs, 'xyz')
    expect(r.matchCount).toBe(0)
    expect(r.matchIds).toEqual([])
  })

  it('returns empty for empty query', () => {
    const r = computeSearchMatches(msgs, '')
    expect(r.matchCount).toBe(0)
  })

  it('returns empty for null query', () => {
    const r = computeSearchMatches(msgs, '')
    expect(r.matchCount).toBe(0)
  })
})

describe('formatUptime', () => {
  it('formats minutes', () => {
    expect(formatUptime(300)).toBe('5m')
  })

  it('formats hours and minutes', () => {
    expect(formatUptime(3660)).toBe('1h 1m')
  })

  it('formats days', () => {
    expect(formatUptime(90000)).toBe('1d 1h 0m')
  })

  it('handles zero', () => {
    expect(formatUptime(0)).toBe('0m')
  })
})

describe('buildLocalPrompt', () => {
  it('builds prompt with system prompt', () => {
    const p = buildLocalPrompt([
      { id: '1', role: 'user', content: 'Hi', timestamp: new Date() },
      { id: '2', role: 'assistant', content: 'Hello', timestamp: new Date() },
    ], 'You are a bot.')
    expect(p).toBe('System: You are a bot.\nUser: Hi\nAssistant: Hello\nAssistant:')
  })

  it('builds prompt without system prompt', () => {
    const p = buildLocalPrompt([
      { id: '1', role: 'user', content: 'Hi', timestamp: new Date() },
    ], '')
    expect(p).toBe('User: Hi\nAssistant:')
  })
})
