import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// ── Mock the HTTP client with an in-memory DocStore ───────────────────
// Mirrors the backend contract (/docstore/{collection}[/{id}]) so the
// db.ts client can be tested end-to-end without a live server.

const { store, apiGet, apiPut, apiPatch, apiDelete, apiPost } = vi.hoisted(() => {
  const store = new Map<string, Map<string, unknown>>()

  function coll(url: string): Map<string, unknown> {
    const name = url.split('?')[0].split('/').filter(Boolean)[1]
    if (!store.has(name)) store.set(name, new Map())
    return store.get(name)!
  }

  const apiGet = vi.fn(async (url: string): Promise<any> => {
    const parts = url.split('?')[0].split('/').filter(Boolean)
    const id = parts[2]
    if (id) return coll(url).get(id) ?? null
    let list = [...coll(url).values()]
    const qs = new URLSearchParams(url.split('?')[1] ?? '')
    if (qs.get('sort')) {
      const field = qs.get('sort')!
      const dir = Number(qs.get('dir') ?? -1)
      list = list.sort((a: any, b: any) => {
        const av = a[field] ?? ''
        const bv = b[field] ?? ''
        return av < bv ? -dir : av > bv ? dir : 0
      })
    }
    if (qs.get('limit')) list = list.slice(0, Number(qs.get('limit')))
    return list
  })

  const apiPut = vi.fn(async (url: string, body?: any): Promise<any> => {
    const parts = url.split('?')[0].split('/').filter(Boolean)
    const id = decodeURIComponent(parts[2])
    const m = coll(url)
    const created = !m.has(id)
    m.set(id, { ...body, id })
    return { id, created }
  })

  const apiPatch = vi.fn(async (url: string, body?: any): Promise<any> => {
    const parts = url.split('?')[0].split('/').filter(Boolean)
    const id = decodeURIComponent(parts[2])
    const m = coll(url)
    const existing = m.get(id)
    if (!existing) return { modified: 0 }
    m.set(id, { ...(existing as object), ...(body ?? {}) })
    return { modified: 1 }
  })

  const apiDelete = vi.fn(async (url: string): Promise<any> => {
    const parts = url.split('?')[0].split('/').filter(Boolean)
    if (parts.length === 2) {
      coll(url).clear()
      return { cleared: true }
    }
    const id = decodeURIComponent(parts[2])
    const deleted = coll(url).delete(id)
    return { deleted }
  })

  const apiPost = vi.fn(async (url: string, body?: any): Promise<any> => {
    const m = coll(url)
    let count = 0
    for (const doc of body?.docs ?? []) {
      if (!doc?.id) continue
      m.set(doc.id, { ...doc })
      count++
    }
    return { imported: count }
  })

  return { store, apiGet, apiPut, apiPatch, apiDelete, apiPost }
})

vi.mock('@/lib/http-client', () => ({ apiGet, apiPut, apiPatch, apiDelete, apiPost }))

// Override the global @/lib/db mock with the real module (which uses the mocked http-client)
vi.mock('@/lib/db', async () => {
  const actual = await vi.importActual<typeof import('./db')>('./db')
  return actual
})

import { chatDB, createChatDB, DbCircuitBreaker, type ChatSession, type ChatMessage } from './db'

beforeEach(() => { store.clear() })
afterEach(() => { store.clear() })

const testMsg: ChatMessage = { id: 'm1', role: 'user', content: 'hello', timestamp: new Date('2024-01-01') }

const testSession: ChatSession = {
  id: 's1',
  name: 'Test Chat',
  messages: [testMsg],
  createdAt: '2024-01-01T00:00:00Z',
  updatedAt: '2024-01-01T00:00:00Z',
  synced: false,
  starred: false,
  pinned: false,
}

describe('chatDB', () => {
  describe('saveSession / loadSession', () => {
    it('saves and loads a session', async () => {
      await chatDB.saveSession(testSession)
      const loaded = await chatDB.loadSession('s1')
      expect(loaded).toBeDefined()
      expect(loaded!.id).toBe('s1')
      expect(loaded!.name).toBe('Test Chat')
    })

    it('loadSession returns undefined for missing id', async () => {
      const loaded = await chatDB.loadSession('nonexistent')
      expect(loaded).toBeUndefined()
    })

    it('converts stored timestamps back to Date objects', async () => {
      await chatDB.saveSession(testSession)
      const loaded = await chatDB.loadSession('s1')
      expect(loaded!.messages[0].timestamp).toBeInstanceOf(Date)
      expect(loaded!.messages[0].timestamp.toISOString()).toBe('2024-01-01T00:00:00.000Z')
    })
  })

  describe('loadSessions', () => {
    it('returns sessions sorted by updatedAt descending', async () => {
      vi.useFakeTimers()
      vi.setSystemTime(new Date('2024-01-03T12:00:00Z'))
      await chatDB.saveSession({ ...testSession, id: 'a' })
      vi.setSystemTime(new Date('2024-01-01T12:00:00Z'))
      await chatDB.saveSession({ ...testSession, id: 'b' })
      vi.setSystemTime(new Date('2024-01-02T12:00:00Z'))
      await chatDB.saveSession({ ...testSession, id: 'c' })
      vi.useRealTimers()
      const all = await chatDB.loadSessions()
      expect(all.map(s => s.id)).toEqual(['a', 'c', 'b'])
    })

    it('returns empty array when no sessions', async () => {
      const all = await chatDB.loadSessions()
      expect(all).toEqual([])
    })
  })

  describe('deleteSession', () => {
    it('removes a session by id', async () => {
      await chatDB.saveSession(testSession)
      await chatDB.deleteSession('s1')
      expect(await chatDB.loadSession('s1')).toBeUndefined()
    })
  })

  describe('updateSession', () => {
    it('updates starred status', async () => {
      await chatDB.saveSession(testSession)
      await chatDB.updateSession('s1', { starred: true })
      const loaded = await chatDB.loadSession('s1')
      expect(loaded!.starred).toBe(true)
    })

    it('updates name', async () => {
      await chatDB.saveSession(testSession)
      await chatDB.updateSession('s1', { name: 'Renamed' })
      const loaded = await chatDB.loadSession('s1')
      expect(loaded!.name).toBe('Renamed')
    })

    it('does nothing for nonexistent id', async () => {
      await chatDB.updateSession('nonexistent', { starred: true })
      // no error
    })
  })

  describe('clearAllSessions', () => {
    it('removes all sessions', async () => {
      await chatDB.saveSession(testSession)
      await chatDB.saveSession({ ...testSession, id: 's2' })
      await chatDB.clearAllSessions()
      expect(await chatDB.loadSessions()).toEqual([])
    })
  })

  describe('getUnsyncedSessions', () => {
    it('returns only sessions with synced=false', async () => {
      await chatDB.saveSession({ ...testSession, id: 's1', synced: false })
      await chatDB.saveSession({ ...testSession, id: 's2', synced: true })
      await chatDB.markSynced('s2')
      const unsynced = await chatDB.getUnsyncedSessions()
      expect(unsynced).toHaveLength(1)
      expect(unsynced[0].id).toBe('s1')
    })
  })

  describe('markSynced', () => {
    it('sets synced to true', async () => {
      await chatDB.saveSession(testSession)
      await chatDB.markSynced('s1')
      const loaded = await chatDB.loadSession('s1')
      expect(loaded!.synced).toBe(true)
    })
  })

  describe('pending messages', () => {
    it('savePendingMessage stores a message', async () => {
      await chatDB.savePendingMessage({ id: 'p1', sessionId: 's1', content: 'test', createdAt: '2024-01-01', retries: 0 })
      const all = await chatDB.getPendingMessages()
      expect(all).toHaveLength(1)
    })

    it('getPendingMessages returns sorted by createdAt', async () => {
      await chatDB.savePendingMessage({ id: 'p2', sessionId: 's1', content: 'b', createdAt: '2024-01-02', retries: 0 })
      await chatDB.savePendingMessage({ id: 'p1', sessionId: 's1', content: 'a', createdAt: '2024-01-01', retries: 0 })
      const all = await chatDB.getPendingMessages()
      expect(all.map(p => p.id)).toEqual(['p1', 'p2'])
    })

    it('deletePendingMessage removes by id', async () => {
      await chatDB.savePendingMessage({ id: 'p1', sessionId: 's1', content: 'x', createdAt: '2024-01-01', retries: 0 })
      await chatDB.deletePendingMessage('p1')
      expect(await chatDB.getPendingMessages()).toEqual([])
    })

    it('clearPendingMessages removes all', async () => {
      await chatDB.savePendingMessage({ id: 'p1', sessionId: 's1', content: 'x', createdAt: '2024-01-01', retries: 0 })
      await chatDB.savePendingMessage({ id: 'p2', sessionId: 's2', content: 'y', createdAt: '2024-01-02', retries: 0 })
      await chatDB.clearPendingMessages()
      expect(await chatDB.getPendingMessages()).toEqual([])
    })
  })

  describe('searchAllSessions', () => {
    it('finds sessions by message content', async () => {
      await chatDB.saveSession(testSession)
      const results = await chatDB.searchAllSessions('hello')
      expect(results).toHaveLength(1)
      expect(results[0].session.id).toBe('s1')
      expect(results[0].matches).toHaveLength(1)
    })

    it('finds sessions by name', async () => {
      await chatDB.saveSession(testSession)
      const results = await chatDB.searchAllSessions('Test')
      expect(results).toHaveLength(1)
    })

    it('returns empty for no match', async () => {
      await chatDB.saveSession(testSession)
      const results = await chatDB.searchAllSessions('zzzz')
      expect(results).toEqual([])
    })

    it('returns empty for empty query', async () => {
      const results = await chatDB.searchAllSessions('')
      expect(results).toEqual([])
    })

    it('returns empty for whitespace-only query', async () => {
      const results = await chatDB.searchAllSessions('   ')
      expect(results).toEqual([])
    })

    it('sorts results by match count descending', async () => {
      const sMany = {
        ...testSession,
        id: 'many',
        messages: [
          { id: 'm1', role: 'user' as const, content: 'hello world', timestamp: new Date('2024-01-01') },
          { id: 'm2', role: 'assistant' as const, content: 'hello again', timestamp: new Date('2024-01-01') },
        ],
      }
      const sOne = { ...testSession, id: 'one', messages: [{ id: 'm3', role: 'user' as const, content: 'hello', timestamp: new Date('2024-01-01') }] }
      await chatDB.saveSession(sMany)
      await chatDB.saveSession(sOne)
      const results = await chatDB.searchAllSessions('hello')
      expect(results[0].session.id).toBe('many')
      expect(results[1].session.id).toBe('one')
    })
  })

  describe('knowledge', () => {
    it('addKnowledge then getKnowledge sorted by timestamp desc', async () => {
      await chatDB.addKnowledge({ id: 'k1', content: 'old', timestamp: 1 })
      await chatDB.addKnowledge({ id: 'k2', content: 'new', timestamp: 2 })
      const all = await chatDB.getKnowledge()
      expect(all.map(k => k.id)).toEqual(['k2', 'k1'])
    })

    it('updateKnowledge merges content', async () => {
      await chatDB.addKnowledge({ id: 'k1', content: 'a', timestamp: 1 })
      await chatDB.updateKnowledge('k1', { content: 'b' })
      const item = await chatDB.getKnowledge()
      expect(item[0].content).toBe('b')
    })

    it('importKnowledge bulk upserts', async () => {
      await chatDB.importKnowledge([{ id: 'k1', content: 'x', timestamp: 0 }])
      expect(await chatDB.getKnowledge()).toHaveLength(1)
    })

    it('clearKnowledge removes all', async () => {
      await chatDB.addKnowledge({ id: 'k1', content: 'x', timestamp: 1 })
      await chatDB.clearKnowledge()
      expect(await chatDB.getKnowledge()).toEqual([])
    })
  })

  describe('bookmarks', () => {
    it('addBookmark then removeBookmark', async () => {
      await chatDB.addBookmark({ id: 'b1', content: 'c', role: 'user', timestamp: 1 })
      expect(await chatDB.getBookmarks()).toHaveLength(1)
      await chatDB.removeBookmark('b1')
      expect(await chatDB.getBookmarks()).toEqual([])
    })
  })

  describe('prompts', () => {
    it('savePrompt then importPrompts then getPrompts', async () => {
      await chatDB.savePrompt({ id: 'p1', name: 'n', description: 'd', prompt: 'p', icon: '', category: 'a', createdAt: 1, updatedAt: 1 })
      await chatDB.importPrompts([{ id: 'p2', name: 'n2', description: 'd2', prompt: 'p2', icon: '', category: 'b', createdAt: 2, updatedAt: 2 }])
      const all = await chatDB.getPrompts()
      expect(all.map(p => p.id).sort()).toEqual(['p1', 'p2'])
      await chatDB.deletePrompt('p1')
      expect(await chatDB.getPrompts()).toHaveLength(1)
    })
  })

  describe('drafts & kv', () => {
    it('saveDraft, getDraft, deleteDraft', async () => {
      expect(await chatDB.getDraft('s1')).toBe('')
      await chatDB.saveDraft('s1', 'hello draft')
      expect(await chatDB.getDraft('s1')).toBe('hello draft')
      await chatDB.saveDraft('s1', '')
      expect(await chatDB.getDraft('s1')).toBe('')
    })

    it('setKV/getKV/deleteKV round trip', async () => {
      expect(await chatDB.getKV('theme')).toBeUndefined()
      await chatDB.setKV('theme', 'dark')
      expect(await chatDB.getKV('theme')).toBe('dark')
      await chatDB.deleteKV('theme')
      expect(await chatDB.getKV('theme')).toBeUndefined()
    })
  })
})

describe('chatDB — circuit breaker', () => {
  const realPut = apiPut.getMockImplementation()!

  // Each test gets a fresh breaker + handle so the closed (false) state is
  // guaranteed without module-level resets.
  function freshHandle() {
    const breaker = new DbCircuitBreaker()
    return { breaker, chatDB: createChatDB(breaker) }
  }

  function failErrorsPut() {
    apiPut.mockImplementation(async (url: string, body?: any) => {
      if (url.startsWith('/docstore/errors')) throw new Error('DocStore unavailable')
      return realPut(url, body)
    })
  }

  afterEach(() => { apiPut.mockImplementation(realPut) })

  it('addError triggers the circuit breaker on server failure, then short-circuits', async () => {
    const { breaker, chatDB: fresh } = freshHandle()
    expect(breaker.isDead()).toBe(false)
    failErrorsPut()
    await fresh.addError('test-trigger')
    expect(breaker.isDead()).toBe(true)

    const putSpy = vi.fn()
    apiPut.mockImplementation(putSpy)
    await fresh.addError('test-after-dead')
    expect(putSpy).not.toHaveBeenCalled()
  })

  it('getErrors returns empty when the DB is dead', async () => {
    const { breaker, chatDB: fresh } = freshHandle()
    failErrorsPut()
    await fresh.addError('x')
    expect(breaker.isDead()).toBe(true)

    expect(await fresh.getErrors()).toEqual([])
  })

  it('clearErrors is a no-op when the DB is dead', async () => {
    const { breaker, chatDB: fresh } = freshHandle()
    failErrorsPut()
    await fresh.addError('x')
    expect(breaker.isDead()).toBe(true)

    apiDelete.mockClear()
    await fresh.clearErrors()
    expect(apiDelete).not.toHaveBeenCalled()
  })

  it('does not leak dead state across handles', async () => {
    const a = freshHandle()
    failErrorsPut()
    await a.chatDB.addError('x')
    expect(a.breaker.isDead()).toBe(true)

    apiPut.mockImplementation(realPut)
    apiPut.mockClear()
    const b = freshHandle()
    expect(b.breaker.isDead()).toBe(false)
    await b.chatDB.addError('ok')
    expect(apiPut).toHaveBeenCalledTimes(1)
  })
})
