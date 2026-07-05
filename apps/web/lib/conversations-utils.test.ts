import { describe, expect, it } from 'vitest'
import { formatDate, truncateMessage, parseConversationJSON, parseConversationMD } from './conversations-utils'

describe('formatDate', () => {
  it('returns empty for undefined', () => expect(formatDate(undefined)).toBe(''))
  it('returns empty for empty string', () => expect(formatDate('')).toBe(''))
  it('returns empty for invalid date', () => expect(formatDate('not-a-date')).toBe(''))
  it('returns "Just now" for < 1 min', () => {
    const now = new Date().toISOString()
    expect(formatDate(now)).toBe('Just now')
  })
  it('returns minutes for < 60 min', () => {
    const d = new Date(Date.now() - 5 * 60000).toISOString()
    expect(formatDate(d)).toBe('5m')
  })
  it('returns hours for < 24h', () => {
    const d = new Date(Date.now() - 3 * 3600000).toISOString()
    expect(formatDate(d)).toBe('3h')
  })
  it('returns days for < 7d', () => {
    const d = new Date(Date.now() - 2 * 86400000).toISOString()
    expect(formatDate(d)).toBe('2d')
  })
  it('returns locale date for >= 7d', () => {
    const d = new Date(Date.now() - 10 * 86400000).toISOString()
    expect(formatDate(d)).toBe(new Date(d).toLocaleDateString())
  })
})

describe('truncateMessage', () => {
  it('returns empty for falsy content', () => {
    expect(truncateMessage('')).toBe('')
    expect(truncateMessage(null as unknown as string)).toBe('')
    expect(truncateMessage(undefined as unknown as string)).toBe('')
  })
  it('returns first line if short enough', () => {
    expect(truncateMessage('Hello world')).toBe('Hello world')
  })
  it('truncates long first line at maxLen', () => {
    const long = 'a'.repeat(100)
    expect(truncateMessage(long, 10)).toBe('aaaaaaaaaa…')
  })
  it('uses default maxLen of 60', () => {
    const long = 'a'.repeat(100)
    const result = truncateMessage(long)
    expect(result.length).toBe(61)
    expect(result.endsWith('…')).toBe(true)
  })
  it('stops at newline', () => {
    expect(truncateMessage('first line\nsecond line', 100)).toBe('first line')
  })
})

describe('parseConversationJSON', () => {
  it('returns empty for non-array without messages', () => {
    expect(parseConversationJSON({})).toEqual([])
  })
  it('parses single conversation', () => {
    const data = { name: 'test', messages: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'hello' }] }
    const result = parseConversationJSON(data)
    expect(result.length).toBe(1)
    expect(result[0].name).toBe('test')
    expect(result[0].messages.length).toBe(2)
  })
  it('parses array of conversations', () => {
    const data = [{ name: 'a', messages: [{ role: 'user', content: 'hi' }] }, { name: 'b', messages: [{ role: 'user', content: 'hey' }] }]
    const result = parseConversationJSON(data)
    expect(result.length).toBe(2)
  })
  it('filters out invalid messages', () => {
    const data = { name: 'test', messages: [{ role: 'unknown', content: 'x' }] }
    const result = parseConversationJSON(data)
    expect(result[0].messages[0].role).toBe('user')
  })
  it('handles non-string content', () => {
    const data = { messages: [{ role: 'user', content: { nested: true } }] }
    const result = parseConversationJSON(data)
    expect(result[0].messages[0].content).toBe('')
  })
  it('skips items without messages array', () => {
    const data = [{ name: 'b' }]
    const result = parseConversationJSON(data)
    expect(result.length).toBe(0)
  })
  it('includes item with empty messages array', () => {
    const data = [{ name: 'a', messages: [] }]
    const result = parseConversationJSON(data)
    expect(result.length).toBe(1)
    expect(result[0].messages).toEqual([])
  })
  it('generates name fallback', () => {
    const data = { messages: [{ role: 'user', content: 'hi' }] }
    const result = parseConversationJSON(data)
    expect(result[0].name).toMatch(/^Imported/)
  })
})

describe('parseConversationMD', () => {
  it('returns empty for empty text', () => {
    expect(parseConversationMD('')).toEqual([])
  })
  it('parses single conversation with user/assistant messages', () => {
    const md = `# My Chat
**User**: Hello
**Assistant**: Hi there
**User**: How are you?
**Assistant**: I'm good`
    const result = parseConversationMD(md)
    expect(result.length).toBe(1)
    expect(result[0].name).toBe('My Chat')
    expect(result[0].messages.length).toBe(4)
    expect(result[0].messages[0].role).toBe('user')
    expect(result[0].messages[0].content).toBe('Hello')
  })
  it('parses multi-line content', () => {
    const md = `# Test
**User**: Line 1
Line 2
**Assistant**: Response`
    const result = parseConversationMD(md)
    expect(result[0].messages.length).toBe(2)
    expect(result[0].messages[0].content).toBe('Line 1\nLine 2')
  })
  it('parses lowercase user/assistant', () => {
    const md = `# Chat
**user**: hello
**assistant**: world`
    const result = parseConversationMD(md)
    expect(result[0].messages.length).toBe(2)
  })
  it('filters conversations with no messages', () => {
    expect(parseConversationMD('# Empty\n')).toEqual([])
  })
})
