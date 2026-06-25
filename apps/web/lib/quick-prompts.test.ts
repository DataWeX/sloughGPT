import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mockStore = new Map<string, string>()
vi.stubGlobal('localStorage', {
  getItem: (key: string) => mockStore.get(key) ?? null,
  setItem: (key: string, val: string) => mockStore.set(key, val),
  removeItem: (key: string) => mockStore.delete(key),
  clear: () => mockStore.clear(),
  get length() { return mockStore.size },
  key: (i: number) => [...mockStore.keys()][i] ?? null,
})

import { applyPrompt, createPrompt, deletePrompt, getPrompt, listPrompts, listPromptsByCategory, resetToDefaults, updatePrompt } from './quick-prompts'

beforeEach(() => {
  vi.useFakeTimers()
  vi.setSystemTime(1000000)
  mockStore.clear()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('applyPrompt', () => {
  it('replaces {{text}} placeholder', () => {
    expect(applyPrompt('Explain: {{text}}', 'hello')).toBe('Explain: hello')
  })

  it('replaces multiple occurrences', () => {
    expect(applyPrompt('{{text}} is {{text}}', 'x')).toBe('x is x')
  })

  it('returns template unchanged if no placeholder', () => {
    expect(applyPrompt('hello', 'x')).toBe('hello')
  })
})

describe('listPrompts', () => {
  it('returns default prompts when no stored data', () => {
    const prompts = listPrompts()
    expect(prompts.length).toBeGreaterThan(5)
    expect(prompts[0].id).toBe('default-0')
    expect(prompts[0].name).toBe('Explain simply')
    expect(prompts[0].category).toBe('learning')
  })

  it('caches result on second call', () => {
    const a = listPrompts()
    const b = listPrompts()
    expect(a).toBe(b)
  })
})

describe('listPromptsByCategory', () => {
  it('groups prompts by category', () => {
    const grouped = listPromptsByCategory()
    expect(grouped.learning).toBeDefined()
    expect(grouped.writing).toBeDefined()
    expect(grouped.coding).toBeDefined()
    expect(grouped.planning).toBeDefined()
    expect(grouped.learning!.length).toBeGreaterThan(0)
  })
})

describe('getPrompt', () => {
  it('finds prompt by id', () => {
    const prompts = listPrompts()
    const found = getPrompt(prompts[0].id)
    expect(found).toBeDefined()
    expect(found!.name).toBe(prompts[0].name)
  })

  it('returns undefined for unknown id', () => {
    expect(getPrompt('nonexistent')).toBeUndefined()
  })
})

describe('createPrompt', () => {
  it('adds a new prompt with generated id', () => {
    const p = createPrompt({ name: 'Test', description: 'desc', prompt: 'hello', icon: '?', category: 'custom' })
    expect(p.id).toMatch(/^prompt_\d+_/)
    expect(p.name).toBe('Test')
    expect(p.createdAt).toBe(1000000)
    expect(listPrompts().length).toBeGreaterThan(5)
  })

  it('persists to localStorage', () => {
    createPrompt({ name: 'T', description: 'd', prompt: 'p', icon: '?', category: 'custom' })
    expect(mockStore.has('man_quick_prompts')).toBe(true)
  })
})

describe('updatePrompt', () => {
  it('updates existing prompt', () => {
    const prompts = listPrompts()
    const id = prompts[0].id
    const updated = updatePrompt(id, { name: 'Updated' })
    expect(updated).toBeDefined()
    expect(updated!.name).toBe('Updated')
    expect(getPrompt(id)!.name).toBe('Updated')
  })

  it('returns undefined for unknown id', () => {
    expect(updatePrompt('unknown', { name: 'x' })).toBeUndefined()
  })
})

describe('deletePrompt', () => {
  it('removes prompt by id', () => {
    const prompts = listPrompts()
    const count = prompts.length
    expect(deletePrompt(prompts[0].id)).toBe(true)
    expect(listPrompts()).toHaveLength(count - 1)
  })

  it('returns false for unknown id', () => {
    expect(deletePrompt('unknown')).toBe(false)
  })
})

describe('resetToDefaults', () => {
  it('clears stored data and persists empty to localStorage', () => {
    createPrompt({ name: 'X', description: '', prompt: '', icon: '?', category: 'custom' })
    expect(listPrompts().length).toBeGreaterThan(5)
    resetToDefaults()
    expect(listPrompts().length).toBe(0)
    expect(JSON.parse(mockStore.get('man_quick_prompts') || '[]')).toEqual([])
  })
})
