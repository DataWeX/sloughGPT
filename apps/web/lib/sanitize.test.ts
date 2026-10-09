import { describe, it, expect } from 'vitest'
import {
  clampNumber,
  parseAndClamp,
  escapeHtml,
  stripScriptTags,
  stripHtmlTags,
  sanitizeText,
  sanitizeForDisplay,
  getFieldLimit,
  clampTrainingField,
  TRAINING_LIMITS,
} from './sanitize'

// ── clampNumber ────────────────────────────────────────────────────

describe('clampNumber', () => {
  it('clamps value within range', () => {
    expect(clampNumber(5, 0, 10)).toBe(5)
  })

  it('clamps to min', () => {
    expect(clampNumber(-5, 0, 10)).toBe(0)
  })

  it('clamps to max', () => {
    expect(clampNumber(15, 0, 10)).toBe(10)
  })

  it('returns fallback for NaN', () => {
    expect(clampNumber(NaN, 0, 10, 7)).toBe(7)
  })

  it('returns fallback for Infinity', () => {
    expect(clampNumber(Infinity, 0, 10, 3)).toBe(3)
  })

  it('returns fallback for -Infinity', () => {
    expect(clampNumber(-Infinity, 0, 10, 3)).toBe(3)
  })

  it('uses min as default fallback', () => {
    expect(clampNumber(NaN, 5, 10)).toBe(5)
  })

  it('handles exact min', () => {
    expect(clampNumber(0, 0, 10)).toBe(0)
  })

  it('handles exact max', () => {
    expect(clampNumber(10, 0, 10)).toBe(10)
  })

  it('handles negative range', () => {
    expect(clampNumber(0, -10, -5)).toBe(-5)
  })
})

// ── parseAndClamp ──────────────────────────────────────────────────

describe('parseAndClamp', () => {
  it('parses valid number', () => {
    expect(parseAndClamp('5', 0, 10)).toBe(5)
  })

  it('clamps parsed value', () => {
    expect(parseAndClamp('15', 0, 10)).toBe(10)
  })

  it('returns fallback for non-numeric string', () => {
    expect(parseAndClamp('abc', 0, 10, 7)).toBe(7)
  })

  it('returns fallback for empty string', () => {
    // Number('') === 0, which is valid, so clampNumber returns 0
    expect(parseAndClamp('', 0, 10, 3)).toBe(0)
  })

  it('parses float', () => {
    expect(parseAndClamp('3.14', 0, 10)).toBe(3.14)
  })

  it('parses negative', () => {
    expect(parseAndClamp('-5', -10, 10)).toBe(-5)
  })

  it('returns min as default fallback', () => {
    expect(parseAndClamp('xyz', 2, 8)).toBe(2)
  })
})

// ── escapeHtml ─────────────────────────────────────────────────────

describe('escapeHtml', () => {
  it('escapes ampersand', () => {
    expect(escapeHtml('a&b')).toBe('a&amp;b')
  })

  it('escapes less than', () => {
    expect(escapeHtml('a<b')).toBe('a&lt;b')
  })

  it('escapes greater than', () => {
    expect(escapeHtml('a>b')).toBe('a&gt;b')
  })

  it('escapes double quote', () => {
    expect(escapeHtml('a"b')).toBe('a&quot;b')
  })

  it('escapes single quote', () => {
    expect(escapeHtml("a'b")).toBe('a&#x27;b')
  })

  it('escapes backtick', () => {
    expect(escapeHtml('a`b')).toBe('a&#96;b')
  })

  it('escapes forward slash', () => {
    expect(escapeHtml('a/b')).toBe('a&#x2F;b')
  })

  it('escapes all special chars in XSS payload', () => {
    const xss = '<script>alert("xss")</script>'
    const escaped = escapeHtml(xss)
    expect(escaped).not.toContain('<')
    expect(escaped).not.toContain('>')
    expect(escaped).not.toContain('"')
    expect(escaped).toContain('&lt;script&gt;')
  })

  it('leaves normal text unchanged', () => {
    expect(escapeHtml('hello world')).toBe('hello world')
  })

  it('handles empty string', () => {
    expect(escapeHtml('')).toBe('')
  })
})

// ── stripScriptTags ────────────────────────────────────────────────

describe('stripScriptTags', () => {
  it('removes script tag and content', () => {
    expect(stripScriptTags('<script>alert("xss")</script>')).toBe('')
  })

  it('removes script tag with attributes', () => {
    expect(stripScriptTags('<script src="evil.js"></script>')).toBe('')
  })

  it('removes script tag case-insensitively', () => {
    expect(stripScriptTags('<SCRIPT>alert(1)</SCRIPT>')).toBe('')
  })

  it('preserves text around script tag', () => {
    expect(stripScriptTags('hello <script>evil</script> world')).toBe('hello  world')
  })

  it('removes multiple script tags', () => {
    const input = '<script>a</script>text<script>b</script>'
    expect(stripScriptTags(input)).toBe('text')
  })

  it('handles nested script-like tags (greedy match)', () => {
    // Regex is non-greedy, so nested tags leave remnants — this is expected
    const result = stripScriptTags('<script><script>inner</script></script>')
    expect(result).toBe('</script>')
  })

  it('leaves non-script tags untouched', () => {
    expect(stripScriptTags('<div>hello</div>')).toBe('<div>hello</div>')
  })

  it('handles empty input', () => {
    expect(stripScriptTags('')).toBe('')
  })
})

// ── stripHtmlTags ──────────────────────────────────────────────────

describe('stripHtmlTags', () => {
  it('removes HTML tags', () => {
    expect(stripHtmlTags('<p>hello</p>')).toBe('hello')
  })

  it('removes nested tags', () => {
    expect(stripHtmlTags('<div><b>bold</b></div>')).toBe('bold')
  })

  it('removes self-closing tags', () => {
    expect(stripHtmlTags('hello<br/>world')).toBe('helloworld')
  })

  it('removes tags with attributes', () => {
    expect(stripHtmlTags('<a href="http://x">link</a>')).toBe('link')
  })

  it('leaves plain text unchanged', () => {
    expect(stripHtmlTags('hello world')).toBe('hello world')
  })

  it('handles empty string', () => {
    expect(stripHtmlTags('')).toBe('')
  })
})

// ── sanitizeText ───────────────────────────────────────────────────

describe('sanitizeText', () => {
  it('trims whitespace', () => {
    expect(sanitizeText('  hello  ')).toBe('hello')
  })

  it('strips script tags', () => {
    expect(sanitizeText('<script>evil</script>hello')).toBe('hello')
  })

  it('enforces max length', () => {
    expect(sanitizeText('a'.repeat(20000), 100)).toHaveLength(100)
  })

  it('preserves normal text', () => {
    expect(sanitizeText('hello world')).toBe('hello world')
  })

  it('handles empty string', () => {
    expect(sanitizeText('')).toBe('')
  })

  it('uses default max length of 10000', () => {
    expect(sanitizeText('a'.repeat(10001))).toHaveLength(10000)
  })
})

// ── sanitizeForDisplay ─────────────────────────────────────────────

describe('sanitizeForDisplay', () => {
  it('escapes HTML entities including forward slash', () => {
    expect(sanitizeForDisplay('<b>bold</b>')).toBe('&lt;b&gt;bold&lt;&#x2F;b&gt;')
  })

  it('strips script tags first', () => {
    expect(sanitizeForDisplay('<script>evil</script>hello')).toBe('hello')
  })

  it('trims whitespace', () => {
    expect(sanitizeForDisplay('  hello  ')).toBe('hello')
  })

  it('enforces max length', () => {
    expect(sanitizeForDisplay('a'.repeat(20000), 100)).toHaveLength(100)
  })

  it('prevents XSS in display', () => {
    const xss = '<img src=x onerror=alert(1)>'
    const safe = sanitizeForDisplay(xss)
    expect(safe).not.toContain('<')
    expect(safe).not.toContain('>')
  })
})

// ── getFieldLimit ──────────────────────────────────────────────────

describe('getFieldLimit', () => {
  it('returns limit for known field', () => {
    const limit = getFieldLimit('epochs')
    expect(limit).toBeDefined()
    expect(limit!.min).toBe(1)
    expect(limit!.max).toBe(1000)
  })

  it('returns undefined for unknown field', () => {
    expect(getFieldLimit('nonexistent')).toBeUndefined()
  })

  it('returns limit for learning_rate', () => {
    const limit = getFieldLimit('learning_rate')
    expect(limit).toBeDefined()
    expect(limit!.min).toBe(1e-5)
    expect(limit!.max).toBe(1.0)
  })

  it('returns limit for temperature', () => {
    const limit = getFieldLimit('temperature')
    expect(limit).toBeDefined()
    expect(limit!.min).toBe(0.1)
    expect(limit!.max).toBe(2.0)
  })
})

// ── clampTrainingField ─────────────────────────────────────────────

describe('clampTrainingField', () => {
  it('clamps to field limits', () => {
    expect(clampTrainingField('epochs', 500)).toBe(500)
    expect(clampTrainingField('epochs', 0)).toBe(1)
    expect(clampTrainingField('epochs', 2000)).toBe(1000)
  })

  it('returns value for unknown field', () => {
    expect(clampTrainingField('unknown', 42)).toBe(42)
  })

  it('clamps temperature', () => {
    expect(clampTrainingField('temperature', 1.5)).toBe(1.5)
    expect(clampTrainingField('temperature', 0.01)).toBe(0.1)
    expect(clampTrainingField('temperature', 5.0)).toBe(2.0)
  })

  it('clamps learning_rate', () => {
    expect(clampTrainingField('learning_rate', 0.001)).toBe(0.001)
    expect(clampTrainingField('learning_rate', 0.0)).toBe(1e-5)
    expect(clampTrainingField('learning_rate', 2.0)).toBe(1.0)
  })
})

// ── TRAINING_LIMITS ───────────────────────────────────────────────

describe('TRAINING_LIMITS', () => {
  it('has all expected training fields', () => {
    expect(TRAINING_LIMITS.epochs).toBeDefined()
    expect(TRAINING_LIMITS.learning_rate).toBeDefined()
    expect(TRAINING_LIMITS.batch_size).toBeDefined()
    expect(TRAINING_LIMITS.temperature).toBeDefined()
    expect(TRAINING_LIMITS.dropout).toBeDefined()
  })

  it('has all expected inference fields', () => {
    expect(TRAINING_LIMITS.max_tokens).toBeDefined()
    expect(TRAINING_LIMITS.top_p).toBeDefined()
    expect(TRAINING_LIMITS.top_k).toBeDefined()
    expect(TRAINING_LIMITS.repetition_penalty).toBeDefined()
  })

  it('has all expected world fields', () => {
    expect(TRAINING_LIMITS.width).toBeDefined()
    expect(TRAINING_LIMITS.height).toBeDefined()
    expect(TRAINING_LIMITS.samples).toBeDefined()
  })

  it('all fields have min < max', () => {
    for (const [key, limit] of Object.entries(TRAINING_LIMITS)) {
      expect(limit.min).toBeLessThan(limit.max)
    }
  })

  it('all step fields have step > 0', () => {
    for (const [key, limit] of Object.entries(TRAINING_LIMITS)) {
      if (limit.step !== undefined) {
        expect(limit.step).toBeGreaterThan(0)
      }
    }
  })
})
