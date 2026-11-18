/**
 * chat-merge — chronological local/remote merge for one session's messages.
 *
 * Both projections of a conversation are ordered arrays: the local copy
 * (IndexedDB via chatDB) and the backend copy (context store via
 * GET /session/{id}/messages). Neither is guaranteed to be a subset of the
 * other: the local copy drifts when stream cleanup rewrites content after a
 * save, and the backend copy drifts when a context save failed or a turn was
 * still unsynced.
 *
 * The old merge (`[...remote, ...localNotInRemote]`) appended every
 * local-only message to the END of the conversation regardless of its real
 * position — an early unsynced message landed after the newest reply, and
 * dedup by bare `role:content` collapsed repeated identical texts. This
 * module aligns the two lists by longest-common-subsequence instead: matched
 * messages keep their local identity (real timestamp/id), local-only messages
 * keep their chronological position, and remote-only messages are inserted
 * where they belong. Array order is the render order — timestamps are
 * display-only, so position is the invariant that matters.
 */

import { sessionController } from '@/lib/session-controller'
import type { ChatMessage } from '@/lib/chat-utils'

export type RemoteMessage = { role: string; content: string; id?: string }

/** Upper bound on the backend remote-merge phase of loadSession. If the server
 *  is slow or offline, fetchMessages can otherwise hold sessionLoading true for
 *  minutes (http-client retries with a fresh 30s timeout per attempt), leaving
 *  the chat screen stuck on its loading skeleton instead of the conversation. */
export const REMOTE_MERGE_TIMEOUT_MS = 8000

/** O(n*m) LCS memory guard: beyond ~4M cells (e.g. 2000x2000) fall back to the
 *  backend-authoritative order instead of allocating the DP table. Real chat
 *  sessions are orders of magnitude smaller. */
const MAX_LCS_CELLS = 4_000_000

/** Runs `task` with an AbortSignal and rejects if `ms` elapses before it
 *  settles. On timeout the controller is aborted so the underlying request is
 *  cancelled, and the race guarantees the caller is never left awaiting a
 *  request that ignores its signal. The losing task keeps a handler attached
 *  (Promise.race) so its eventual outcome is never an unhandled rejection. */
async function withRemoteTimeout<T>(
  task: (signal: AbortSignal) => Promise<T>,
  ms: number,
): Promise<T> {
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

/** Identity key for alignment. JSON-encoded pair so role participates in the
 *  identity (a user/assistant pair with identical text never matches across
 *  roles) and content cannot collide via an ad-hoc separator. */
function keyOf(role: string, content: string): string {
  return JSON.stringify([role, content])
}

function remoteToMessage(r: RemoteMessage, index: number): ChatMessage {
  return {
    // Backend ids survive reloads when the writer assigned one; otherwise a
    // position-stable synthetic id (the old `remote_${Date.now()}_...` changed
    // on every load, breaking message-keyed UI state across session loads).
    id: r.id || `remote_${index}`,
    role: r.role as ChatMessage['role'],
    content: r.content,
    timestamp: new Date(),
  }
}

/**
 * Align `local` and `remote` chronologically (LCS) and return the merged
 * conversation in true order.
 *
 * - matched (same role+content): keep the LOCAL message — it carries the real
 *   timestamp, original id, and any images/attachments.
 * - local-only: keep in place (unsynced turn or drifted content).
 * - remote-only: insert at its position.
 *
 * Beyond MAX_LCS_CELLS degrades to the legacy behaviour (remote first,
 * unmatched local appended) — order may be wrong there, never missing.
 */
export function mergeSessionMessages(local: ChatMessage[], remote: RemoteMessage[]): ChatMessage[] {
  if (remote.length === 0) return local
  if (local.length === 0) return remote.map(remoteToMessage)

  const n = local.length
  const m = remote.length
  const lkeys = local.map((msg) => keyOf(msg.role, msg.content))
  const rkeys = remote.map((msg) => keyOf(msg.role, msg.content))

  if (n * m > MAX_LCS_CELLS) {
    const seen = new Set(rkeys)
    return [...remote.map(remoteToMessage), ...local.filter((msg, i) => !seen.has(lkeys[i]))]
  }

  // LCS length table, filled back-to-front: dp[i][j] = LCS(local[i..], remote[j..]).
  const width = m + 1
  const dp = new Int32Array((n + 1) * width)
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i * width + j] =
        lkeys[i] === rkeys[j]
          ? dp[(i + 1) * width + j + 1] + 1
          : Math.max(dp[(i + 1) * width + j], dp[i * width + j + 1])
    }
  }

  const out: ChatMessage[] = []
  let i = 0
  let j = 0
  while (i < n && j < m) {
    if (lkeys[i] === rkeys[j]) {
      out.push(local[i])
      i++
      j++
    } else if (dp[(i + 1) * width + j] >= dp[i * width + j + 1]) {
      // Skipping the local message costs no LCS — it is local-only here.
      out.push(local[i])
      i++
    } else {
      out.push(remoteToMessage(remote[j], j))
      j++
    }
  }
  while (i < n) out.push(local[i++])
  while (j < m) out.push(remoteToMessage(remote[j], j++))
  return out
}

/** A persisted empty assistant message must render nothing, never a loading
 *  placeholder (AGENTS.md loading-state rule) — drop both sides' empties. */
function dropEmptyAssistant(msgs: ChatMessage[]): ChatMessage[] {
  return msgs.filter((m) => !(m.role === 'assistant' && !m.content.trim()))
}

/**
 * Load a session's full conversation: local copy merged with the backend
 * context store, chronologically. Backend failures/timeouts degrade to the
 * filtered local copy — loading a session must never throw.
 *
 * Shared by BOTH loadSession implementations (useChatSessions for the
 * controller path, useChatStore for the store path) so the two can never
 * diverge in ordering semantics.
 */
export async function loadMergedMessages(
  sessionId: string,
  localMessages: ChatMessage[],
): Promise<ChatMessage[]> {
  try {
    const remote = await withRemoteTimeout(
      (signal) => sessionController.fetchMessages(sessionId, { signal, silent: true }),
      REMOTE_MERGE_TIMEOUT_MS,
    )
    return dropEmptyAssistant(mergeSessionMessages(localMessages, remote))
  } catch {
    return dropEmptyAssistant(localMessages)
  }
}
