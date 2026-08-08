import Dexie, { type Table } from 'dexie'

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

interface ManDB extends Dexie {
  sessions: Table<StoredChatSession, string>
  pendingMessages: Table<PendingMessage, string>
  knowledge: Table<KnowledgeItem, string>
  bookmarks: Table<BookmarkedMessage, string>
  prompts: Table<QuickPrompt, string>
  drafts: Table<Draft, string>
  kv: Table<KVEntry, string>
  errors: Table<ErrorEntry, string>
}

const db = new Dexie('ManDB') as ManDB

db.version(1).stores({
  sessions: 'id, name, updatedAt, synced',
  pendingMessages: 'id, sessionId, createdAt',
})

db.version(2).stores({
  sessions: 'id, name, updatedAt, synced',
  pendingMessages: 'id, sessionId, createdAt',
  knowledge: 'id, timestamp',
  bookmarks: 'id, timestamp, role',
  prompts: 'id, name, category, createdAt',
})

db.version(3).stores({
  sessions: 'id, name, updatedAt, synced',
  pendingMessages: 'id, sessionId, createdAt',
  knowledge: 'id, timestamp',
  bookmarks: 'id, timestamp, role',
  prompts: 'id, name, category, createdAt',
  drafts: 'sessionId',
  kv: 'key',
  errors: 'id, timestamp',
})

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

export const chatDB = {
  async saveSession(session: ChatSession): Promise<void> {
    const stored = toStored(session)
    stored.updatedAt = new Date().toISOString()
    stored.synced = false
    await db.sessions.put(stored)
  },

  async loadSessions(): Promise<ChatSession[]> {
    const sessions = await db.sessions.orderBy('updatedAt').reverse().toArray()
    return sessions.map(fromStored)
  },

  async loadSession(id: string): Promise<ChatSession | undefined> {
    const session = await db.sessions.get(id)
    return session ? fromStored(session) : undefined
  },

  async deleteSession(id: string): Promise<void> {
    await db.sessions.delete(id)
  },

  async updateSession(id: string, updates: { starred?: boolean; name?: string; pinned?: boolean; archived?: boolean }): Promise<void> {
    const session = await db.sessions.get(id)
    if (session) {
      await db.sessions.update(id, {
        ...(updates.starred !== undefined && { starred: updates.starred }),
        ...(updates.name !== undefined && { name: updates.name }),
        ...(updates.pinned !== undefined && { pinned: updates.pinned }),
        ...(updates.archived !== undefined && { archived: updates.archived }),
        updatedAt: new Date().toISOString(),
      })
    }
  },

  async clearAllSessions(): Promise<void> {
    await db.sessions.clear()
  },

  async getUnsyncedSessions(): Promise<ChatSession[]> {
    const sessions = await db.sessions.where('synced').equals(0).toArray()
    return sessions.map(fromStored)
  },

  async markSynced(id: string): Promise<void> {
    await db.sessions.update(id, { synced: true })
  },

  async markUnread(id: string, unread: boolean): Promise<void> {
    await db.sessions.update(id, { unread })
  },

  async savePendingMessage(msg: PendingMessage): Promise<void> {
    await db.pendingMessages.put(msg)
  },

  async getPendingMessages(): Promise<PendingMessage[]> {
    return db.pendingMessages.orderBy('createdAt').toArray()
  },

  async deletePendingMessage(id: string): Promise<void> {
    await db.pendingMessages.delete(id)
  },

  async clearPendingMessages(): Promise<void> {
    await db.pendingMessages.clear()
  },

  async searchAllSessions(query: string): Promise<Array<{ session: ChatSession; matches: ChatMessage[] }>> {
    if (!query.trim()) return []
    const q = query.toLowerCase()
    const all = await db.sessions.toArray()
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
    return db.knowledge.orderBy('timestamp').reverse().toArray()
  },

  async addKnowledge(item: KnowledgeItem): Promise<void> {
    await db.knowledge.put(item)
  },

  async updateKnowledge(id: string, updates: { content?: string }): Promise<void> {
    await db.knowledge.update(id, updates)
  },

  async deleteKnowledge(id: string): Promise<void> {
    await db.knowledge.delete(id)
  },

  async clearKnowledge(): Promise<void> {
    await db.knowledge.clear()
  },

  async importKnowledge(items: KnowledgeItem[]): Promise<void> {
    await db.knowledge.bulkPut(items)
  },

  async getBookmarks(): Promise<BookmarkedMessage[]> {
    return db.bookmarks.orderBy('timestamp').reverse().toArray()
  },

  async addBookmark(item: BookmarkedMessage): Promise<void> {
    await db.bookmarks.put(item)
  },

  async removeBookmark(id: string): Promise<void> {
    await db.bookmarks.delete(id)
  },

  async clearBookmarks(): Promise<void> {
    await db.bookmarks.clear()
  },

  async getPrompts(): Promise<QuickPrompt[]> {
    return db.prompts.orderBy('createdAt').reverse().toArray()
  },

  async savePrompt(prompt: QuickPrompt): Promise<void> {
    await db.prompts.put(prompt)
  },

  async deletePrompt(id: string): Promise<void> {
    await db.prompts.delete(id)
  },

  async clearPrompts(): Promise<void> {
    await db.prompts.clear()
  },

  async importPrompts(prompts: QuickPrompt[]): Promise<void> {
    await db.prompts.bulkPut(prompts)
  },

  async getDraft(sessionId: string): Promise<string> {
    const draft = await db.drafts.get(sessionId)
    return draft?.text ?? ''
  },

  async saveDraft(sessionId: string, text: string): Promise<void> {
    if (!text) {
      await db.drafts.delete(sessionId)
    } else {
      await db.drafts.put({ sessionId, text, updatedAt: Date.now() })
    }
  },

  async deleteDraft(sessionId: string): Promise<void> {
    await db.drafts.delete(sessionId)
  },

  async getKV<T = unknown>(key: string): Promise<T | undefined> {
    const entry = await db.kv.get(key)
    return entry?.value as T | undefined
  },

  async setKV(key: string, value: unknown): Promise<void> {
    await db.kv.put({ key, value })
  },

  async deleteKV(key: string): Promise<void> {
    await db.kv.delete(key)
  },

  async addError(message: string, stack?: string): Promise<void> {
    await db.errors.put({
      id: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      message,
      stack,
      timestamp: Date.now(),
    })
  },

  async getErrors(limit = 20): Promise<ErrorEntry[]> {
    return db.errors.orderBy('timestamp').reverse().limit(limit).toArray()
  },

  async clearErrors(): Promise<void> {
    await db.errors.clear()
  },
}
