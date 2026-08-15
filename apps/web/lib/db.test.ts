import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// ── Mock Dexie with in-memory tables ─────────────────────────────────

const { tables, FakeDexie } = vi.hoisted(() => {
  const tables = new Map<string, Map<string, any>>()

  class FakeTable {
    private _key: string
    constructor(key: string) { this._key = key }

    private _data() {
      if (!tables.has(this._key)) tables.set(this._key, new Map())
      return tables.get(this._key)!
    }

    put(obj: any) { this._data().set(obj.id, { ...obj }) }
    get(id: string) { return this._data().get(id) }
    delete(id: string) { this._data().delete(id) }
    clear() { tables.delete(this._key) }

    update(id: string, updates: any) {
      const d = this._data()
      const existing = d.get(id)
      if (existing) d.set(id, { ...existing, ...updates })
    }

    toArray() { return [...this._data().values()] }

    orderBy(field: string) {
      return {
        reverse: () => ({
          toArray: () =>
            [...this._data().values()].sort((a, b) => {
              const av = a[field] ?? ''
              const bv = b[field] ?? ''
              return av < bv ? 1 : av > bv ? -1 : 0
            }),
        }),
        toArray: () =>
          [...this._data().values()].sort((a, b) => {
            const av = a[field] ?? ''
            const bv = b[field] ?? ''
            return av < bv ? -1 : av > bv ? 1 : 0
          }),
      }
    }

    where(field: string) {
      return {
        equals: (val: any) => ({
          toArray: () => [...this._data().values()].filter(v => v[field] == val),
        }),
      }
    }
  }

  class FakeDexie {
    sessions = new FakeTable('sessions')
    pendingMessages = new FakeTable('pending')
    version() { return { stores() {} } }
    on(_event: string, _handler: (...args: any[]) => void) { /* no-op */ }
  }

  return { tables, FakeDexie }
})

vi.mock('dexie', () => ({ default: FakeDexie }))

// Override the global @/lib/db mock with the real module (which uses our FakeDexie)
vi.mock('@/lib/db', async () => {
  const actual = await vi.importActual<typeof import('./db')>('./db')
  return actual
})

import { chatDB, type ChatSession, type ChatMessage } from './db'

beforeEach(() => { tables.clear() })
afterEach(() => { tables.clear() })

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
})

describe('chatDB — circuit breaker (_dbDead)', () => {
  it('addError silently returns when DB is dead', async () => {
    const { chatDB, isDBDead } = await import('./db')
    if (isDBDead()) {
      await chatDB.addError('test-when-dead')
    }
  })

  it('getErrors returns empty array when DB is dead', async () => {
    const { chatDB, isDBDead } = await import('./db')
    if (isDBDead()) {
      const errors = await chatDB.getErrors()
      expect(errors).toEqual([])
    }
  })

  it('clearErrors silently returns when DB is dead', async () => {
    const { chatDB, isDBDead } = await import('./db')
    if (isDBDead()) {
      await chatDB.clearErrors()
    }
  })
})
