import { describe, it, expect } from 'vitest'
import {
  parseMemoryImport,
  archiveTypeLabel,
  archiveBadgeClass,
  archiveSummary,
} from './memory-card-utils'

// ── parseMemoryImport ──────────────────────────────────────────────

describe('parseMemoryImport', () => {
  describe('JSON files', () => {
    it('parses array of strings', () => {
      const result = parseMemoryImport('["hello", "world"]', 'data.json')
      expect(result).toEqual([
        { content: 'hello', topic: 'manual' },
        { content: 'world', topic: 'manual' },
      ])
    })

    it('parses array of objects', () => {
      const input = JSON.stringify([
        { content: 'fact 1', topic: 'science' },
        { content: 'fact 2', topic: 'history' },
      ])
      expect(parseMemoryImport(input, 'data.json')).toEqual([
        { content: 'fact 1', topic: 'science' },
        { content: 'fact 2', topic: 'history' },
      ])
    })

    it('parses mixed strings and objects', () => {
      const input = JSON.stringify(['plain text', { content: 'with topic', topic: 'math' }])
      expect(parseMemoryImport(input, 'data.json')).toEqual([
        { content: 'plain text', topic: 'manual' },
        { content: 'with topic', topic: 'math' },
      ])
    })

    it('parses {items|facts|records} wrapper objects', () => {
      const input = JSON.stringify({ records: [{ content: 'wrapped', topic: 'w' }] })
      expect(parseMemoryImport(input, 'data.json')).toEqual([{ content: 'wrapped', topic: 'w' }])
    })

    it('accepts text/fact aliases for content', () => {
      const input = JSON.stringify([{ text: 'via text' }, { fact: 'via fact' }])
      const result = parseMemoryImport(input, 'data.json')
      expect(result.map((e) => e.content)).toEqual(['via text', 'via fact'])
    })

    it('skips empty strings and objects without content', () => {
      const input = JSON.stringify(['hello', '', { topic: 'science' }, { content: 'valid' }])
      expect(parseMemoryImport(input, 'data.json')).toEqual([
        { content: 'hello', topic: 'manual' },
        { content: 'valid', topic: 'manual' },
      ])
    })

    it('returns empty array for non-array JSON', () => {
      expect(parseMemoryImport('{"key": "value"}', 'data.json')).toEqual([])
    })

    it('returns empty array for malformed JSON instead of throwing', () => {
      expect(parseMemoryImport('{nope', 'data.json')).toEqual([])
    })

    it('trims whitespace in content and topic', () => {
      const input = JSON.stringify(['  hello  ', { content: ' x ', topic: '  t ' }])
      const result = parseMemoryImport(input, 'data.json')
      expect(result[0].content).toBe('hello')
      expect(result[1]).toEqual({ content: 'x', topic: 't' })
    })
  })

  describe('CSV files', () => {
    it('parses CSV with content and topic columns', () => {
      const csv = 'content,topic\nfact 1,science\nfact 2,history'
      expect(parseMemoryImport(csv, 'data.csv')).toEqual([
        { content: 'fact 1', topic: 'science' },
        { content: 'fact 2', topic: 'history' },
      ])
    })

    it('parses CSV with only content column', () => {
      const csv = 'content\nfact 1\nfact 2'
      expect(parseMemoryImport(csv, 'data.csv')).toEqual([
        { content: 'fact 1', topic: 'manual' },
        { content: 'fact 2', topic: 'manual' },
      ])
    })

    it('handles quoted values', () => {
      const csv = 'content,topic\n"fact, with comma",science'
      expect(parseMemoryImport(csv, 'data.csv')[0].content).toBeDefined()
    })

    it('returns empty for CSV with only header', () => {
      expect(parseMemoryImport('content,topic', 'data.csv')).toEqual([])
    })

    it('returns empty for empty CSV', () => {
      expect(parseMemoryImport('', 'data.csv')).toEqual([])
    })

    it('treats every line as data when the header has no content column', () => {
      const csv = 'name,value\nhello,world'
      const result = parseMemoryImport(csv, 'data.csv')
      expect(result[0].content).toBe('name')
      expect(result).toHaveLength(2)
    })
  })

  describe('plain text files', () => {
    it('parses plain text lines', () => {
      expect(parseMemoryImport('line 1\nline 2\nline 3', 'notes.txt')).toEqual([
        { content: 'line 1', topic: 'manual' },
        { content: 'line 2', topic: 'manual' },
        { content: 'line 3', topic: 'manual' },
      ])
    })

    it('parses bracket-annotated topics', () => {
      const result = parseMemoryImport(
        'fact about cats [animals]\nfact about dogs [animals]',
        'notes.txt',
      )
      expect(result).toEqual([
        { content: 'fact about cats', topic: 'animals' },
        { content: 'fact about dogs', topic: 'animals' },
      ])
    })

    it('handles mixed annotated and plain', () => {
      const result = parseMemoryImport('plain text\nannotated [science]', 'notes.txt')
      expect(result).toEqual([
        { content: 'plain text', topic: 'manual' },
        { content: 'annotated', topic: 'science' },
      ])
    })

    it('skips empty lines', () => {
      expect(parseMemoryImport('line 1\n\n\nline 2', 'notes.txt')).toHaveLength(2)
    })

    it('returns empty for empty text', () => {
      expect(parseMemoryImport('', 'notes.txt')).toEqual([])
    })
  })

  describe('filename detection', () => {
    it('is case-insensitive for .json', () => {
      expect(parseMemoryImport('["test"]', 'DATA.JSON')).toHaveLength(1)
    })

    it('is case-insensitive for .csv', () => {
      expect(parseMemoryImport('content\nhello', 'DATA.CSV')).toHaveLength(1)
    })
  })
})

// ── archiveTypeLabel ───────────────────────────────────────────────

describe('archiveTypeLabel', () => {
  it('returns remember for memory.remember', () => {
    expect(archiveTypeLabel('memory.remember')).toBe('remember')
  })

  it('returns stored for memory.store', () => {
    expect(archiveTypeLabel('memory.store')).toBe('stored')
  })

  it('returns consolidate for memory.consolidate', () => {
    expect(archiveTypeLabel('memory.consolidate')).toBe('consolidate')
  })

  it('strips the first dotted segment for unknown memory types', () => {
    expect(archiveTypeLabel('memory.custom')).toBe('custom')
  })

  it('strips the namespace prefix for task types', () => {
    expect(archiveTypeLabel('task.cleanup')).toBe('cleanup')
  })

  it('returns empty string for empty type', () => {
    expect(archiveTypeLabel('')).toBe('')
  })
})

// ── archiveBadgeClass ──────────────────────────────────────────────

describe('archiveBadgeClass', () => {
  it('returns primary class for remember', () => {
    expect(archiveBadgeClass('memory.remember')).toContain('primary')
  })

  it('returns success class for store', () => {
    expect(archiveBadgeClass('memory.store')).toContain('success')
  })

  it('returns warning class for consolidate', () => {
    expect(archiveBadgeClass('memory.consolidate')).toContain('warning')
  })

  it('returns muted class for unknown types', () => {
    expect(archiveBadgeClass('memory.unknown')).toContain('muted')
  })
})

// ── archiveSummary ─────────────────────────────────────────────────

describe('archiveSummary', () => {
  it('summarizes remember records with truncation', () => {
    const long = 'x'.repeat(70)
    const result = archiveSummary({
      task_type: 'memory.remember',
      user_message: long,
      assistant_response: 'short',
    } as Record<string, unknown>)
    expect(result.text).toBe('x'.repeat(60) + '…')
    expect(result.detail).toBe('short')
  })

  it('falls back to (no message) for empty remember records', () => {
    const result = archiveSummary({ task_type: 'memory.remember' } as Record<string, unknown>)
    expect(result.text).toBe('(no message)')
    expect(result.detail).toBe('')
  })

  it('summarizes store records with topic', () => {
    const result = archiveSummary({
      task_type: 'memory.store',
      content: 'Cats are felines',
      topic: 'animals',
    } as Record<string, unknown>)
    expect(result.text).toBe('Cats are felines')
    expect(result.detail).toBe('Topic: animals')
  })

  it('summarizes store records without topic', () => {
    const result = archiveSummary({
      task_type: 'memory.store',
      content: 'Some fact',
    } as Record<string, unknown>)
    expect(result.detail).toBe('')
  })

  it('summarizes consolidate records', () => {
    const result = archiveSummary({
      task_type: 'memory.consolidate',
      removed: 3,
      kept: 5,
    } as Record<string, unknown>)
    expect(result.text).toBe('Consolidated 3 duplicate(s), kept 5')
    expect(result.detail).toBe('')
  })

  it('returns empty summary for unknown record types', () => {
    const result = archiveSummary({
      task_type: 'memory.unknown',
      foo: 'bar',
    } as Record<string, unknown>)
    expect(result).toEqual({ text: '', detail: '' })
  })
})
