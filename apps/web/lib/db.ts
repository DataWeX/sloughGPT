import { apiGet, apiPut, apiPatch, apiDelete, apiPost } from './http-client'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

export interface ChatSession {
  id: string
  name: string
  messages: ChatMessage[]
  createdAt: string
  updatedAt: string
  synced: boolean
  starred: boolean
  pinned: boolean
  archived?: boolean
  unread?: boolean
}

export interface KnowledgeItem {
  id: string
  content: string
  timestamp: number
}

export interface BookmarkedMessage {
  id: string
  content: string
  role: 'user' | 'assistant'
  timestamp: number
  sessionTitle?: string
}

export interface QuickPrompt {
  id: string
  name: string
  description: string
  prompt: string
  icon: string
  category: string
  createdAt: number
  updatedAt: number
}

interface StoredChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
}

interface StoredChatSession {
  id: string
  name: string
  messages: StoredChatMessage[]
  createdAt: string
  updatedAt: string
  synced: boolean
  pinned: boolean
  starred: boolean
  archived?: boolean
  unread?: boolean
}

interface PendingMessage {
  id: string
  sessionId: string
  content: string
  createdAt: string
  retries: number
}

interface Draft {
  sessionId: string
  text: string
  updatedAt: number
}

interface KVEntry {
  key: string
  value: unknown
}

interface ErrorEntry {
  id: string
  message: string
  stack?: string
  timestamp: number
}

const DOCSTORE = '/docstore'

function docUrl(collection: string, id?: string): string {
  const base = `${DOCSTORE}/${collection}`
  return id ? `${base}/${encodeURIComponent(id)}` : base
}

// ── Server circuit breaker ────────────────────────────────────────────
// When the API server is unreachable every failed write re-throws, which
// fires window.onerror, which calls chatDB.addError(), which fails again —
// infinite cascade.  This flag breaks the cycle.

/** Per-handle circuit breaker; a fresh instance isolates tests from shared state. */
export class DbCircuitBreaker {
  private _dead = false

  isDead(): boolean {
    return this._dead
  }

  markDead(err: unknown): void {
    if (this._dead) return
    this._dead = true
    const msg = err instanceof Error ? err.message : String(err)
    console.warn('[ManDB] DocStore marked dead — all further error writes will be skipped:', msg)
  }
}

const _defaultBreaker = new DbCircuitBreaker()

/** Returns true if the DocStore is known to be unavailable. */
export function isDBDead(): boolean { return _defaultBreaker.isDead() }

function toStored(session: ChatSession): StoredChatSession {
  return {
    ...session,
    messages: session.messages.map(m => ({
      ...m,
      timestamp: typeof m.timestamp === 'string' ? m.timestamp : m.timestamp.toISOString(),
    })),
  }
}

function fromStored(session: StoredChatSession): ChatSession {
  return {
    ...session,
    messages: session.messages.map(m => ({
      ...m,
      timestamp: new Date(m.timestamp),
    })),
  }
}

/** Create a chatDB handle over the shared DocStore REST API with an isolated breaker. */
export function createChatDB(breaker: DbCircuitBreaker = _defaultBreaker) {
  return {
    async saveSession(session: ChatSession): Promise<void> {
      const stored = toStored(session)
      stored.updatedAt = new Date().toISOString()
      stored.synced = false
      await apiPut(docUrl('sessions', session.id), stored)
    },

    async loadSessions(): Promise<ChatSession[]> {
      const sessions = await apiGet<StoredChatSession[]>(docUrl('sessions'))
      return sessions
        .slice()
        .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
        .map(fromStored)
    },

    async loadSession(id: string): Promise<ChatSession | undefined> {
      const session = await apiGet<StoredChatSession | null>(docUrl('sessions', id))
      return session ? fromStored(session) : undefined
    },

    async deleteSession(id: string): Promise<void> {
      await apiDelete(docUrl('sessions', id))
    },

    async updateSession(id: string, updates: { starred?: boolean; name?: string; pinned?: boolean; archived?: boolean }): Promise<void> {
      await apiPatch(docUrl('sessions', id), {
        ...(updates.starred !== undefined && { starred: updates.starred }),
        ...(updates.name !== undefined && { name: updates.name }),
        ...(updates.pinned !== undefined && { pinned: updates.pinned }),
        ...(updates.archived !== undefined && { archived: updates.archived }),
        updatedAt: new Date().toISOString(),
      })
    },

    async clearAllSessions(): Promise<void> {
      await apiDelete(docUrl('sessions'))
    },

    async getUnsyncedSessions(): Promise<ChatSession[]> {
      const sessions = await apiGet<StoredChatSession[]>(docUrl('sessions'))
      return sessions.filter(s => s.synced === false).map(fromStored)
    },

    async markSynced(id: string): Promise<void> {
      await apiPatch(docUrl('sessions', id), { synced: true })
    },

    async markUnread(id: string, unread: boolean): Promise<void> {
      await apiPatch(docUrl('sessions', id), { unread })
    },

    async savePendingMessage(msg: PendingMessage): Promise<void> {
      await apiPut(docUrl('pendingMessages', msg.id), msg)
    },

    async getPendingMessages(): Promise<PendingMessage[]> {
      const msgs = await apiGet<PendingMessage[]>(docUrl('pendingMessages'))
      return msgs.slice().sort((a, b) => a.createdAt.localeCompare(b.createdAt))
    },

    async deletePendingMessage(id: string): Promise<void> {
      await apiDelete(docUrl('pendingMessages', id))
    },

    async clearPendingMessages(): Promise<void> {
      await apiDelete(docUrl('pendingMessages'))
    },

    async searchAllSessions(query: string): Promise<Array<{ session: ChatSession; matches: ChatMessage[] }>> {
      if (!query.trim()) return []
      const q = query.toLowerCase()
      const all = await apiGet<StoredChatSession[]>(docUrl('sessions'))
      const results: Array<{ session: ChatSession; matches: ChatMessage[] }> = []
      for (const stored of all) {
        const session = fromStored(stored)
        const matches = session.messages.filter(m => m.content.toLowerCase().includes(q))
        if (matches.length > 0 || session.name.toLowerCase().includes(q)) {
          results.push({ session, matches })
        }
      }
      return results.sort((a, b) => b.matches.length - a.matches.length)
    },

    async getKnowledge(): Promise<KnowledgeItem[]> {
      const items = await apiGet<KnowledgeItem[]>(docUrl('knowledge'))
      return items.slice().sort((a, b) => b.timestamp - a.timestamp)
    },

    async addKnowledge(item: KnowledgeItem): Promise<void> {
      await apiPut(docUrl('knowledge', item.id), item)
    },

    async updateKnowledge(id: string, updates: { content?: string }): Promise<void> {
      await apiPatch(docUrl('knowledge', id), updates)
    },

    async deleteKnowledge(id: string): Promise<void> {
      await apiDelete(docUrl('knowledge', id))
    },

    async clearKnowledge(): Promise<void> {
      await apiDelete(docUrl('knowledge'))
    },

    async importKnowledge(items: KnowledgeItem[]): Promise<void> {
      await apiPost(docUrl('knowledge', 'bulk'), { docs: items })
    },

    async getBookmarks(): Promise<BookmarkedMessage[]> {
      const items = await apiGet<BookmarkedMessage[]>(docUrl('bookmarks'))
      return items.slice().sort((a, b) => b.timestamp - a.timestamp)
    },

    async addBookmark(item: BookmarkedMessage): Promise<void> {
      await apiPut(docUrl('bookmarks', item.id), item)
    },

    async removeBookmark(id: string): Promise<void> {
      await apiDelete(docUrl('bookmarks', id))
    },

    async clearBookmarks(): Promise<void> {
      await apiDelete(docUrl('bookmarks'))
    },

    async getPrompts(): Promise<QuickPrompt[]> {
      const prompts = await apiGet<QuickPrompt[]>(docUrl('prompts'))
      return prompts.slice().sort((a, b) => b.createdAt - a.createdAt)
    },

    async savePrompt(prompt: QuickPrompt): Promise<void> {
      await apiPut(docUrl('prompts', prompt.id), prompt)
    },

    async deletePrompt(id: string): Promise<void> {
      await apiDelete(docUrl('prompts', id))
    },

    async clearPrompts(): Promise<void> {
      await apiDelete(docUrl('prompts'))
    },

    async importPrompts(prompts: QuickPrompt[]): Promise<void> {
      await apiPost(docUrl('prompts', 'bulk'), { docs: prompts })
    },

    async getDraft(sessionId: string): Promise<string> {
      const draft = await apiGet<Draft | null>(docUrl('drafts', sessionId))
      return draft?.text ?? ''
    },

    async saveDraft(sessionId: string, text: string): Promise<void> {
      if (!text) {
        await apiDelete(docUrl('drafts', sessionId))
      } else {
        await apiPut(docUrl('drafts', sessionId), { sessionId, text, updatedAt: Date.now() })
      }
    },

    async deleteDraft(sessionId: string): Promise<void> {
      await apiDelete(docUrl('drafts', sessionId))
    },

    async getKV<T = unknown>(key: string): Promise<T | undefined> {
      const entry = await apiGet<KVEntry | null>(docUrl('kv', key))
      return entry?.value as T | undefined
    },

    async setKV(key: string, value: unknown): Promise<void> {
      await apiPut(docUrl('kv', key), { key, value })
    },

    async deleteKV(key: string): Promise<void> {
      await apiDelete(docUrl('kv', key))
    },

    async addError(message: string, stack?: string): Promise<void> {
      if (breaker.isDead()) return // circuit breaker — don't cascade
      try {
        await apiPut(docUrl('errors', `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`), {
          id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
          message,
          stack,
          timestamp: Date.now(),
        })
      } catch (err: unknown) {
        breaker.markDead(err)
      }
    },

    async getErrors(limit = 20): Promise<ErrorEntry[]> {
      if (breaker.isDead()) return []
      try {
        return await apiGet<ErrorEntry[]>(docUrl('errors'), { sort: 'timestamp', dir: '-1', limit: String(limit) })
      } catch { return [] }
    },

    async clearErrors(): Promise<void> {
      if (breaker.isDead()) return
      try { await apiDelete(docUrl('errors')) } catch { /* ignore */ }
    },
  }
}

export const chatDB = createChatDB()
