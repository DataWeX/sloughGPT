import { describe, expect, it } from 'vitest'

import {
  cleanStreamedContent,
  stripAssistantPrefix,
  computeSearchMatches,
  formatSize,
  formatUptime,
  getErrorInfo,
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

describe('formatSize', () => {
  it('formats bytes', () => {
    expect(formatSize(500)).toBe('500 B')
  })

  it('formats KB', () => {
    expect(formatSize(2048)).toBe('2.0 KB')
  })

  it('formats MB', () => {
    expect(formatSize(1048576 * 3)).toBe('3.0 MB')
  })

  it('formats GB', () => {
    expect(formatSize(1073741824 * 2)).toBe('2.00 GB')
  })

  it('handles zero', () => {
    expect(formatSize(0)).toBe('0 B')
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

describe('getErrorInfo', () => {
  it('detects model not loaded', () => {
    const r = getErrorInfo(new Error('no model loaded'))
    expect(r.title).toBe('Model Not Loaded')
    expect(r.retryable).toBe(true)
  })

  it('detects connection error from Failed to fetch', () => {
    const r = getErrorInfo('Failed to fetch')
    expect(r.title).toBe('Connection Error')
    expect(r.retryable).toBe(true)
  })

  it('detects connection error from NetworkError', () => {
    const r = getErrorInfo(new Error('NetworkError'))
    expect(r.title).toBe('Connection Error')
  })

  it('detects timeout', () => {
    const r = getErrorInfo('request timed out')
    expect(r.title).toBe('Request Timeout')
  })

  it('detects rate limit', () => {
    const r = getErrorInfo('rate limit exceeded')
    expect(r.title).toBe('Rate Limited')
  })

  it('detects unauthorized', () => {
    const r = getErrorInfo('401 Unauthorized')
    expect(r.title).toBe('Unauthorized')
    expect(r.retryable).toBe(false)
  })

  it('detects 503', () => {
    const r = getErrorInfo('503 Service Unavailable')
    expect(r.title).toBe('Service Unavailable')
  })

  it('falls through to generic error', () => {
    const r = getErrorInfo('something weird happened')
    expect(r.title).toBe('Error')
    expect(r.retryable).toBe(true)
  })

  it('handles null gracefully', () => {
    const r = getErrorInfo(null)
    expect(r.title).toBe('Unknown Error')
    expect(r.retryable).toBe(false)
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
