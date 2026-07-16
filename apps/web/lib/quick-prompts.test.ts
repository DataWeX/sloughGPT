import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { mockPrompts, mockChatDB } = vi.hoisted(() => {
  const mockPrompts = new Map<string, any>()
  const mockChatDB = {
    getPrompts: vi.fn(async () => [...mockPrompts.values()]),
    savePrompt: vi.fn(async (p: any) => { mockPrompts.set(p.id, p) }),
    deletePrompt: vi.fn(async (id: string) => { mockPrompts.delete(id) }),
    clearPrompts: vi.fn(async () => { mockPrompts.clear() }),
    importPrompts: vi.fn(async (items: any[]) => { for (const p of items) mockPrompts.set(p.id, p) }),
  }
  return { mockPrompts, mockChatDB }
})

vi.mock('@/lib/db', () => ({
  chatDB: mockChatDB,
}))

import { applyPrompt, createPrompt, deletePrompt, getPrompt, listPrompts, listPromptsByCategory, resetToDefaults, updatePrompt, __resetPromptCache, initPrompts } from './quick-prompts'

beforeEach(async () => {
  vi.useFakeTimers()
  vi.setSystemTime(1000000)
  vi.clearAllMocks()
  __resetPromptCache()
  mockPrompts.clear()
  mockChatDB.getPrompts.mockImplementation(async () => [...mockPrompts.values()])
  await initPrompts()
  __resetPromptCache()
  vi.clearAllMocks()
  mockChatDB.getPrompts.mockImplementation(async () => [...mockPrompts.values()])
  mockChatDB.savePrompt.mockImplementation(async (p: any) => { mockPrompts.set(p.id, p) })
  mockChatDB.deletePrompt.mockImplementation(async (id: string) => { mockPrompts.delete(id) })
  mockChatDB.clearPrompts.mockImplementation(async () => { mockPrompts.clear() })
  mockChatDB.importPrompts.mockImplementation(async (items: any[]) => { for (const p of items) mockPrompts.set(p.id, p) })
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
  it('returns default prompts when no stored data', async () => {
    const prompts = await listPrompts()
    expect(prompts.length).toBeGreaterThan(5)
    expect(prompts[0].id).toBe('default-0')
    expect(prompts[0].name).toBe('Explain simply')
    expect(prompts[0].category).toBe('learning')
  })

  it('caches result on second call', async () => {
    const a = await listPrompts()
    const b = await listPrompts()
    expect(a).toBe(b)
    expect(a.length).toBeGreaterThan(0)
  })
})

describe('listPromptsByCategory', () => {
  it('groups prompts by category', async () => {
    const grouped = await listPromptsByCategory()
    expect(grouped.learning).toBeDefined()
    expect(grouped.writing).toBeDefined()
    expect(grouped.coding).toBeDefined()
    expect(grouped.planning).toBeDefined()
    expect(grouped.learning!.length).toBeGreaterThan(0)
  })
})

describe('getPrompt', () => {
  it('finds prompt by id', async () => {
    const prompts = await listPrompts()
    const found = await getPrompt(prompts[0].id)
    expect(found).toBeDefined()
    expect(found!.name).toBe(prompts[0].name)
  })

  it('returns undefined for unknown id', async () => {
    expect(await getPrompt('nonexistent')).toBeUndefined()
  })
})

describe('createPrompt', () => {
  it('adds a new prompt with generated id', async () => {
    const p = await createPrompt({ name: 'Test', description: 'desc', prompt: 'hello', icon: '?', category: 'custom' })
    expect(p.id).toMatch(/^prompt_\d+_/)
    expect(p.name).toBe('Test')
    expect(p.createdAt).toBe(1000000)
    const all = await listPrompts()
    expect(all.length).toBeGreaterThan(5)
  })

  it('saves via chatDB', async () => {
    await createPrompt({ name: 'T', description: 'd', prompt: 'p', icon: '?', category: 'custom' })
    expect(mockChatDB.savePrompt).toHaveBeenCalled()
  })
})

describe('updatePrompt', () => {
  it('updates existing prompt', async () => {
    const prompts = await listPrompts()
    const id = prompts[0].id
    const updated = await updatePrompt(id, { name: 'Updated' })
    expect(updated).toBeDefined()
    expect(updated!.name).toBe('Updated')
    expect((await getPrompt(id))!.name).toBe('Updated')
  })

  it('returns undefined for unknown id', async () => {
    expect(await updatePrompt('unknown', { name: 'x' })).toBeUndefined()
  })
})

describe('deletePrompt', () => {
  it('removes prompt by id', async () => {
    const prompts = await listPrompts()
    const count = prompts.length
    expect(await deletePrompt(prompts[0].id)).toBe(true)
    expect(await listPrompts()).toHaveLength(count - 1)
  })

  it('returns false for unknown id', async () => {
    expect(await deletePrompt('unknown')).toBe(false)
  })
})

describe('resetToDefaults', () => {
  it('clears stored data via chatDB', async () => {
    await createPrompt({ name: 'X', description: '', prompt: '', icon: '?', category: 'custom' })
    const before = await listPrompts()
    expect(before.length).toBeGreaterThan(5)
    await resetToDefaults()
    expect(mockChatDB.clearPrompts).toHaveBeenCalled()
    const after = await listPrompts()
    expect(after.every(p => p.id.startsWith('default-'))).toBe(true)
  })
})
