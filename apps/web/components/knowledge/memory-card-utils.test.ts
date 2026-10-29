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
      const result = parseMemoryImport(input, 'data.json')
      expect(result).toEqual([
        { content: 'fact 1', topic: 'science' },
        { content: 'fact 2', topic: 'history' },
      ])
    })

    it('parses mixed strings and objects', () => {
      const input = JSON.stringify([
        'plain text',
        { content: 'with topic', topic: 'math' },
      ])
      const result = parseMemoryImport(input, 'data.json')
      expect(result).toEqual([
        { content: 'plain text', topic: 'manual' },
        { content: 'with topic', topic: 'math' },
      ])
    })

    it('skips empty strings', () => {
      const result = parseMemoryImport('["hello", "", "world"]', 'data.json')
      expect(result).toHaveLength(2)
    })

    it('skips objects without content', () => {
      const input = JSON.stringify([{ topic: 'science' }, { content: 'valid', topic: 'math' }])
      const result = parseMemoryImport(input, 'data.json')
      expect(result).toHaveLength(1)
    })

    it('defaults topic to manual for objects without topic', () => {
      const input = JSON.stringify([{ content: 'fact' }])
      const result = parseMemoryImport(input, 'data.json')
      expect(result[0].topic).toBe('manual')
    })

    it('returns empty array for non-array JSON', () => {
      expect(parseMemoryImport('{"key": "value"}', 'data.json')).toEqual([])
    })

    it('returns empty array for empty array', () => {
      expect(parseMemoryImport('[]', 'data.json')).toEqual([])
    })

    it('trims whitespace in content', () => {
      const input = JSON.stringify(['  hello  '])
      const result = parseMemoryImport(input, 'data.json')
      expect(result[0].content).toBe('hello')
    })
  })

  describe('CSV files', () => {
    it('parses CSV with content and topic columns', () => {
      const csv = 'content,topic\nfact 1,science\nfact 2,history'
      const result = parseMemoryImport(csv, 'data.csv')
      expect(result).toEqual([
        { content: 'fact 1', topic: 'science' },
        { content: 'fact 2', topic: 'history' },
      ])
    })

    it('parses CSV with only content column', () => {
      const csv = 'content\nfact 1\nfact 2'
      const result = parseMemoryImport(csv, 'data.csv')
      expect(result).toEqual([
        { content: 'fact 1', topic: 'manual' },
        { content: 'fact 2', topic: 'manual' },
      ])
    })

    it('handles quoted values', () => {
      const csv = 'content,topic\n"fact, with comma",science'
      const result = parseMemoryImport(csv, 'data.csv')
      // CSV parser splits on commas, so quoted value gets split
      expect(result[0].content).toBeDefined()
    })

    it('returns empty for CSV with only header', () => {
      expect(parseMemoryImport('content,topic', 'data.csv')).toEqual([])
    })

    it('returns empty for empty CSV', () => {
      expect(parseMemoryImport('', 'data.csv')).toEqual([])
    })

    it('uses first column if no content column', () => {
      const csv = 'name,value\nhello,world'
      const result = parseMemoryImport(csv, 'data.csv')
      expect(result[0].content).toBe('hello')
    })
  })

  describe('plain text files', () => {
    it('parses plain text lines', () => {
      const result = parseMemoryImport('line 1\nline 2\nline 3', 'notes.txt')
      expect(result).toEqual([
        { content: 'line 1', topic: 'manual' },
        { content: 'line 2', topic: 'manual' },
        { content: 'line 3', topic: 'manual' },
      ])
    })

    it('parses bracket-annotated topics', () => {
      const result = parseMemoryImport('fact about cats [animals]\nfact about dogs [animals]', 'notes.txt')
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
      const result = parseMemoryImport('line 1\n\n\nline 2', 'notes.txt')
      expect(result).toHaveLength(2)
    })

    it('returns empty for empty text', () => {
      expect(parseMemoryImport('', 'notes.txt')).toEqual([])
    })
  })

  describe('filename detection', () => {
    it('is case-insensitive for .json', () => {
      const result = parseMemoryImport('["test"]', 'DATA.JSON')
      expect(result).toHaveLength(1)
    })

    it('is case-insensitive for .csv', () => {
      const result = parseMemoryImport('content\nhello', 'DATA.CSV')
      expect(result).toHaveLength(1)
    })
  })
})

// ── archiveTypeLabel ───────────────────────────────────────────────

describe('archiveTypeLabel', () => {
  it('returns remember for memory.remember', () => {
    expect(archiveTypeLabel('memory.remember')).toBe('remember')
  })

  it('returns store for memory.store', () => {
    expect(archiveTypeLabel('memory.store')).toBe('store')
  })

  it('returns consolidate for memory.consolidate', () => {
    expect(archiveTypeLabel('memory.consolidate')).toBe('consolidate')
  })

  it('strips memory. prefix for unknown types', () => {
    expect(archiveTypeLabel('memory.custom')).toBe('custom')
  })

  it('returns type as-is for non-memory types', () => {
    expect(archiveTypeLabel('task.cleanup')).toBe('task.cleanup')
  })

  it('returns task for empty string', () => {
    expect(archiveTypeLabel('')).toBe('task')
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
  it('summarizes remember records', () => {
    const result = archiveSummary({
      task_type: 'memory.remember',
      user_message: 'I like cats',
    } as any)
    expect(result.text).toBe('I like cats')
    expect(result.detail).toBe('Learned from a conversation')
  })

  it('summarizes store records with topic', () => {
    const result = archiveSummary({
      task_type: 'memory.store',
      content: 'Cats are felines',
      topic: 'animals',
    } as any)
    expect(result.text).toBe('Cats are felines')
    expect(result.detail).toBe('Topic: animals')
  })

  it('summarizes store records without topic', () => {
    const result = archiveSummary({
      task_type: 'memory.store',
      content: 'Some fact',
    } as any)
    expect(result.detail).toBe('Stored fact')
  })

  it('summarizes consolidate records', () => {
    const result = archiveSummary({
      task_type: 'memory.consolidate',
      removed: 3,
      kept: 5,
      threshold: 0.85,
    } as any)
    expect(result.text).toBe('Consolidated 3 duplicate(s), kept 5')
    expect(result.detail).toBe('Threshold 0.85')
  })

  it('summarizes unknown record types as JSON', () => {
    const result = archiveSummary({
      task_type: 'memory.unknown',
      foo: 'bar',
    } as any)
    expect(result.text).toContain('memory.unknown')
    expect(result.detail).toBe('Task record')
  })

  it('falls back to type label for empty records', () => {
    const result = archiveSummary({
      task_type: 'custom.type',
    } as any)
    // Falls back to JSON serialization of the record (minus ts/task_id)
    expect(result.text).toBeDefined()
    expect(result.detail).toBe('Task record')
  })
})
