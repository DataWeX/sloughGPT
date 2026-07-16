'use client'

import { useState, useCallback } from 'react'
import {
  CURRENT_SESSION_KEY, generateSessionId,
  type ChatMessage, type ChatSession,
} from '@/lib/chat-utils'
import { chatDB, type ChatSession as DBChatSession } from '@/lib/db'
import { sessionController } from '@/lib/session-controller'
import { addGlobalError } from '@/lib/error-store'
import type { Conversation } from '@/lib/session-controller'

const MAX_STORAGE_MESSAGES = 40
const DRAFT_PREFIX = 'man_draft_'

/** Module-level tracking: prevents duplicate backend session creation across
 *  React StrictMode double-mounts and concurrent hook instances. */
const _createdSessions = new Set<string>()

export interface UseChatSessionsReturn {
  sessions: ChatSession[]
  setSessions: React.Dispatch<React.SetStateAction<ChatSession[]>>
  sidebarConversations: Conversation[]
  archivedCount: number
  loadSession: (sessionId: string) => Promise<void>
  deleteSession: (sessionId: string) => Promise<void>
  starSession: (sessionId: string, starred: boolean) => Promise<void>
  pinSession: (sessionId: string, pinned: boolean) => Promise<void>
  archiveSession: (sessionId: string, archived: boolean) => Promise<void>
  renameSession: (sessionId: string, newName: string) => Promise<void>
  duplicateSession: (sessionId: string) => Promise<void>
  saveSessionToStorage: (msgs: ChatMessage[], sessionId: string) => Promise<void>
}

export function useChatSessions(opts: {
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>
  setInput: React.Dispatch<React.SetStateAction<string>>
  setSessionSaved: (v: boolean) => void
  setSessionLoading: (v: boolean) => void
  sessionIdRef: React.MutableRefObject<string>
  showToast: (message: string, type?: string) => void
}): UseChatSessionsReturn {
  const { setMessages, setInput, setSessionSaved, setSessionLoading, sessionIdRef, showToast } = opts

  const [sessions, setSessions] = useState<ChatSession[]>([])

  const saveSessionToStorage = useCallback(async (msgs: ChatMessage[], sessionId: string) => {
    const sessionName = (() => {
      const first = msgs.find(m => m.role === 'user')?.content || ''
      if (!first) return 'New Chat'
      const cleaned = first.replace(/^(hey|hi|hello|yo|sup)[\s,!.]*/i, '').trim()
      const sentence = cleaned.split(/[.!?\n]/).filter(Boolean)[0]?.trim() || cleaned
      const maxLen = 42
      return sentence.length > maxLen ? sentence.slice(0, maxLen).trimEnd() + '…' : sentence
    })()
    const storedMsgs = msgs.length > MAX_STORAGE_MESSAGES
      ? msgs.slice(-MAX_STORAGE_MESSAGES)
      : msgs
    const session: DBChatSession = {
      id: sessionId, name: sessionName, messages: storedMsgs,
      createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(),
      synced: false, starred: false, pinned: false,
    }
    await chatDB.saveSession(session)
    if (!_createdSessions.has(sessionId)) {
      _createdSessions.add(sessionId)
      sessionController.create(sessionName, sessionId).catch(err => {
        _createdSessions.delete(sessionId)
        addGlobalError(err, 'Chat:SessionCreate')
      })
    } else {
      sessionController.update(sessionId, { name: sessionName }).catch(err => addGlobalError(err, 'Chat:SessionUpdate'))
    }
    const newLocal = await chatDB.loadSessions()
    setSessions(newLocal)
  }, [])

  const loadSession = useCallback(async (sessionId: string) => {
    setSessionLoading(true)
    try {
      const session = await chatDB.loadSession(sessionId)
      if (session) {
        const filteredMessages = session.messages.filter(msg => {
          if (msg.role === 'assistant' && !msg.content.trim()) return false
          return true
        })
        try {
          const remoteMsgs = await sessionController.fetchMessages(sessionId)
          const remoteChatMsgs = remoteMsgs.map(m => ({
            id: `remote_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
            role: m.role as 'user' | 'assistant',
            content: m.content,
            timestamp: new Date(),
          }))
          const seen = new Set(remoteChatMsgs.map(m => `${m.role}:${m.content}`))
          const uniqueLocal = filteredMessages.filter(m => !seen.has(`${m.role}:${m.content}`))
          setMessages([...remoteChatMsgs, ...uniqueLocal])
        } catch {
          setMessages(filteredMessages)
        }
        localStorage.setItem(CURRENT_SESSION_KEY, sessionId)
        const isComplete = filteredMessages.length > 0 &&
          filteredMessages[filteredMessages.length - 1].role === 'assistant'
        setSessionSaved(isComplete)
        const savedDraft = localStorage.getItem(`${DRAFT_PREFIX}${sessionId}`)
        if (savedDraft) setInput(savedDraft)
        else setInput('')
        if (filteredMessages.length > 0) showToast(`Loaded: ${session.name}`)
      }
    } finally {
      setSessionLoading(false)
    }
  }, [showToast, setMessages, setInput, setSessionSaved, setSessionLoading])

  const deleteSession = useCallback(async (sessionId: string) => {
    await chatDB.deleteSession(sessionId)
    sessionController.delete(sessionId).catch(console.error)
    const newSessions = await chatDB.loadSessions()
    setSessions(newSessions)
    if (localStorage.getItem(CURRENT_SESSION_KEY) === sessionId) {
      setMessages([])
      setSessionSaved(false)
      localStorage.removeItem(CURRENT_SESSION_KEY)
    }
    showToast('Session deleted')
  }, [showToast, setMessages, setSessionSaved])

  const starSession = useCallback(async (sessionId: string, starred: boolean) => {
    await chatDB.updateSession(sessionId, { starred })
    sessionController.update(sessionId, { starred }).catch(console.error)
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, starred } : s))
    showToast(starred ? 'Conversation starred' : 'Conversation unstarred')
  }, [showToast])

  const pinSession = useCallback(async (sessionId: string, pinned: boolean) => {
    await chatDB.updateSession(sessionId, { pinned })
    sessionController.update(sessionId, { pinned }).catch(console.error)
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, pinned } : s))
    showToast(pinned ? 'Conversation pinned' : 'Conversation unpinned')
  }, [showToast])

  const archiveSession = useCallback(async (sessionId: string, archived: boolean) => {
    await chatDB.updateSession(sessionId, { archived })
    sessionController.update(sessionId, { archived }).catch(console.error)
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, archived } : s))
    showToast(archived ? 'Conversation archived' : 'Conversation restored')
  }, [showToast])

  const renameSession = useCallback(async (sessionId: string, newName: string) => {
    await chatDB.updateSession(sessionId, { name: newName })
    sessionController.update(sessionId, { name: newName }).catch(console.error)
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, name: newName } : s))
    showToast('Conversation renamed')
  }, [showToast])

  const duplicateSession = useCallback(async (sessionId: string) => {
    const session = await chatDB.loadSession(sessionId)
    if (session) {
      const newId = `chat_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
      const newSession = {
        ...session, id: newId, name: `${session.name} (copy)`,
        createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(),
        starred: false, pinned: false,
      }
      await chatDB.saveSession(newSession)
      const newSessions = await chatDB.loadSessions()
      setSessions(newSessions)
      showToast('Conversation duplicated')
    }
  }, [showToast])

  const archivedCount = (Array.isArray(sessions) ? sessions : []).filter(s => s.archived).length

  const sidebarConversations: Conversation[] = (Array.isArray(sessions) ? sessions : [])
    .filter(s => !s.archived)
    .map(s => ({
    id: s.id, name: s.name || 'Untitled', session_id: s.id,
    created_at: s.createdAt || new Date().toISOString(),
    updated_at: s.updatedAt || new Date().toISOString(),
    pinned: Boolean(s.pinned), starred: Boolean(s.starred),
    message_count: Array.isArray(s.messages) ? s.messages.length : 0,
    messages: (Array.isArray(s.messages) ? s.messages : []).map(m => ({
      id: String(m.id || `msg_${Date.now()}`), role: String(m.role || 'user'), content: String(m.content || ''),
      timestamp: m.timestamp instanceof Date ? m.timestamp.toISOString() : String(m.timestamp || Date.now()),
    })),
    synced: Boolean(s.synced),
  }))

  return {
    sessions, setSessions,
    sidebarConversations, archivedCount,
    loadSession,
    deleteSession, starSession, pinSession, archiveSession, renameSession, duplicateSession,
    saveSessionToStorage,
  }
}
