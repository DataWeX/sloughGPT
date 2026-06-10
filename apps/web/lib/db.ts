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
}

interface PendingMessage {
  id: string
  sessionId: string
  content: string
  createdAt: string
  retries: number
}

interface ManDB extends Dexie {
  sessions: Table<StoredChatSession, string>
  pendingMessages: Table<PendingMessage, string>
}

const db = new Dexie('ManDB') as ManDB

db.version(1).stores({
  sessions: 'id, name, updatedAt, synced',
  pendingMessages: 'id, sessionId, createdAt',
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

  async updateSession(id: string, updates: { starred?: boolean; name?: string; pinned?: boolean }): Promise<void> {
    const session = await db.sessions.get(id)
    if (session) {
      await db.sessions.update(id, {
        ...(updates.starred !== undefined && { starred: updates.starred }),
        ...(updates.name !== undefined && { name: updates.name }),
        ...(updates.pinned !== undefined && { pinned: updates.pinned }),
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
}