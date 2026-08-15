'use client'

import { useState, useCallback, useRef } from 'react'
import {
  CURRENT_SESSION_KEY, generateSessionId,
  type ChatMessage, type ChatSession,
} from '@/lib/chat-utils'
import { chatDB, type ChatSession as DBChatSession } from '@/lib/db'
import { sessionController } from '@/lib/session-controller'
import { addGlobalError } from '@/lib/error-store'
import type { Conversation } from '@/lib/session-controller'

const MAX_STORAGE_MESSAGES = 40

/** Upper bound on the backend remote-merge phase of loadSession. If the server
 *  is slow or offline, fetchMessages can otherwise hold sessionLoading true for
 *  minutes (http-client retries with a fresh 30s timeout per attempt), leaving
 *  the chat screen stuck on its loading skeleton instead of the conversation. */
const REMOTE_MERGE_TIMEOUT_MS = 8000

/** Module-level tracking: prevents duplicate backend session creation across
 *  React StrictMode double-mounts and concurrent hook instances. */
const _createdSessions = new Set<string>()

/** Runs `task` with an AbortSignal and rejects if `ms` elapses before it
 *  settles. On timeout the controller is aborted so the underlying request is
 *  cancelled, and the race guarantees the caller is never left awaiting a
 *  request that ignores its signal. The losing task keeps a handler attached
 *  (Promise.race) so its eventual outcome is never an unhandled rejection. */
async function withRemoteTimeout<T>(task: (signal: AbortSignal) => Promise<T>, ms: number): Promise<T> {
  const controller = new AbortController()
  let timer: ReturnType<typeof setTimeout> | undefined
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      controller.abort()
      reject(new Error('Remote merge timed out'))
    }, ms)
  })
  try {
    return await Promise.race([task(controller.signal), timeout])
  } finally {
    if (timer) clearTimeout(timer)
  }
}

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
  messagesRef: React.MutableRefObject<ChatMessage[]>
  showToast: (message: string, type?: 'success' | 'error' | 'info') => void
}): UseChatSessionsReturn {
  const { setMessages, setInput, setSessionSaved, setSessionLoading, sessionIdRef, messagesRef, showToast } = opts

  const [sessions, setSessions] = useState<ChatSession[]>([])

  /** Number of loadSession calls still in flight for this hook instance. Loading
   *  is released when the count reaches zero so a superseded load (e.g. the user
   *  clicked New Chat while an old session was still merging) can never leave the
   *  skeleton mounted: some later load may never come to clear it. */
  const inFlightLoadsRef = useRef(0)

  /** Session ids deleted while a loadSession for them may still be in flight.
   *  A delete does not change sessionIdRef, so the existing stale guards would
   *  otherwise let the load resurrect a deleted session (apply its messages and
   *  restore CURRENT_SESSION_KEY). Sending a new message re-creates the session
   *  via saveSessionToStorage, which unmarks the id again. */
  const deletedSessionsRef = useRef<Set<string>>(new Set())

  const saveSessionToStorage = useCallback(async (msgs: ChatMessage[], sessionId: string) => {
    const sessionName = (() => {
      const first = msgs.find(m => m.role === 'user')?.content || ''
      if (!first) return 'New Chat'
      const cleaned = first.replace(/^(hey|hi|hello|yo|sup)[\s,!.]*/i, '').trim()
      const sentence = cleaned.split(/[.!?\n]/).find(Boolean)?.trim() || cleaned
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
    deletedSessionsRef.current.delete(sessionId)
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
    sessionIdRef.current = sessionId
    setSessionLoading(true)
    inFlightLoadsRef.current++
    const baselineMsgCount = messagesRef.current.length
    const superseded = () =>
      sessionIdRef.current !== sessionId ||
      deletedSessionsRef.current.has(sessionId) ||
      messagesRef.current.length !== baselineMsgCount
    try {
      const session = await chatDB.loadSession(sessionId)
      if (session) {
        const filteredMessages = session.messages.filter(msg => {
          if (msg.role === 'assistant' && !msg.content.trim()) return false
          return true
        })
        try {
          const remoteMsgs = await withRemoteTimeout(
            (signal) => sessionController.fetchMessages(sessionId, { signal, silent: true }),
            REMOTE_MERGE_TIMEOUT_MS,
          )
          if (superseded()) return
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
          if (superseded()) return
          setMessages(filteredMessages)
        }
        if (superseded()) return
        await chatDB.setKV(CURRENT_SESSION_KEY, sessionId)
        const isComplete = filteredMessages.length > 0 &&
          filteredMessages[filteredMessages.length - 1].role === 'assistant'
        setSessionSaved(isComplete)
        const savedDraft = await chatDB.getDraft(sessionId)
        if (savedDraft) setInput(savedDraft)
        else setInput('')
        if (filteredMessages.length > 0) showToast(`Loaded: ${session.name}`)
      }
    } finally {
      inFlightLoadsRef.current--
      if (inFlightLoadsRef.current <= 0) setSessionLoading(false)
    }
  }, [showToast, setMessages, setInput, setSessionSaved, setSessionLoading, sessionIdRef])

  const deleteSession = useCallback(async (sessionId: string) => {
    deletedSessionsRef.current.add(sessionId)
    await chatDB.deleteSession(sessionId)
    sessionController.delete(sessionId).catch((e) => addGlobalError({ message: 'Session delete sync failed', source: 'useChatSessions', metadata: { sessionId, error: String(e) } }))
    const newSessions = await chatDB.loadSessions()
    setSessions(newSessions)
    if ((await chatDB.getKV<string>(CURRENT_SESSION_KEY)) === sessionId) {
      setMessages([])
      setSessionSaved(false)
      await chatDB.deleteKV(CURRENT_SESSION_KEY)
    }
    showToast('Session deleted')
  }, [showToast, setMessages, setSessionSaved])

  const starSession = useCallback(async (sessionId: string, starred: boolean) => {
    await chatDB.updateSession(sessionId, { starred })
    sessionController.update(sessionId, { starred }).catch((e) => addGlobalError({ message: 'Session star sync failed', source: 'useChatSessions', metadata: { sessionId, error: String(e) } }))
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, starred } : s))
    showToast(starred ? 'Conversation starred' : 'Conversation unstarred')
  }, [showToast])

  const pinSession = useCallback(async (sessionId: string, pinned: boolean) => {
    await chatDB.updateSession(sessionId, { pinned })
    sessionController.update(sessionId, { pinned }).catch((e) => addGlobalError({ message: 'Session pin sync failed', source: 'useChatSessions', metadata: { sessionId, error: String(e) } }))
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, pinned } : s))
    showToast(pinned ? 'Conversation pinned' : 'Conversation unpinned')
  }, [showToast])

  const archiveSession = useCallback(async (sessionId: string, archived: boolean) => {
    await chatDB.updateSession(sessionId, { archived })
    sessionController.update(sessionId, { archived }).catch((e) => addGlobalError({ message: 'Session archive sync failed', source: 'useChatSessions', metadata: { sessionId, error: String(e) } }))
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, archived } : s))
    showToast(archived ? 'Conversation archived' : 'Conversation restored')
  }, [showToast])

  const renameSession = useCallback(async (sessionId: string, newName: string) => {
    await chatDB.updateSession(sessionId, { name: newName })
    sessionController.update(sessionId, { name: newName }).catch((e) => addGlobalError({ message: 'Session rename sync failed', source: 'useChatSessions', metadata: { sessionId, error: String(e) } }))
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

  const archivedCount = sessions.filter(s => s.archived).length

  const sidebarConversations: Conversation[] = sessions
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
