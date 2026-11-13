'use client'

/**
 * Conversation AppScope cache — the UI's mirror of each conversation's
 * per-instance config (Phase 2: a conversation is the instance boundary of
 * the app).
 *
 * Server truth lives on the session record (`SessionMeta.scope`, written via
 * `PUT /chat/sessions/{id}` and resolved once at the turn seam). This store is
 * the read/write mirror for surfaces that must show or apply a value *before*
 * a turn happens — the chat settings dialog and the offline local engine. It
 * is deliberately seeded from the session-index fetch the chat page already
 * performs (no extra request) and write-through on every scope PUT, kept
 * independent of the local session docstore (`chatDB`), whose full-record
 * PUTs rebuild rows without server-only fields.
 *
 * The precedence rule mirrored here is the server's, per field, never per
 * layer: conversation delta if set, else the global default.
 */

import { create } from 'zustand'
import { sessionController } from './session-controller'
import type { AppScopeDelta, SessionMeta } from './protocol/contracts.gen'

interface ConversationScopeState {
  /** session_id → stored delta. Absent = no override — inherit the global default. */
  scopes: Record<string, AppScopeDelta>
  /** Merge server rows into the cache (fed by the chat page's session-index fetch). */
  seed: (rows: Array<Pick<SessionMeta, 'id' | 'scope'>>) => void
  /** Write-through after a successful scope PUT. */
  setScope: (sessionId: string, scope: AppScopeDelta) => void
}

export const useConversationScope = create<ConversationScopeState>((set) => ({
  scopes: {},
  seed: (rows) =>
    set((state) => {
      const seeded: Record<string, AppScopeDelta> = {}
      for (const row of rows) {
        if (row.scope) seeded[row.id] = row.scope
      }
      // Local writes win: a PUT that landed after this index was fetched is
      // newer than the row it would otherwise overwrite.
      return { scopes: { ...seeded, ...state.scopes } }
    }),
  setScope: (sessionId, scope) =>
    set((state) => ({ scopes: { ...state.scopes, [sessionId]: scope } })),
}))

/**
 * Persist a conversation's delta: merge the patch over the cached one (the
 * endpoint replaces the stored delta wholesale), PUT, then write through so
 * the UI reflects the server's new truth without a refetch. Rejects without
 * touching the cache when the PUT fails — the display must never claim a
 * scope the server did not take.
 */
export async function saveConversationScope(
  sessionId: string,
  patch: Partial<AppScopeDelta>,
): Promise<void> {
  const current = useConversationScope.getState().scopes[sessionId]
  const next: AppScopeDelta = { ...current, ...patch }
  await sessionController.update(sessionId, { scope: next })
  useConversationScope.getState().setScope(sessionId, next)
}

/**
 * Effective generation params for one conversation — the offline twin of the
 * turn seam's precedence for the local (in-browser) engine, which has no
 * server to resolve its layers: the delta's field when set, else the caller's
 * global default, decided per field.
 */
export function effectiveGeneration(
  sessionId: string,
  fallback: { temperature: number; maxTokens: number },
): { temperature: number; maxTokens: number } {
  const delta = useConversationScope.getState().scopes[sessionId]
  return {
    temperature: delta?.temperature ?? fallback.temperature,
    maxTokens: delta?.max_tokens ?? fallback.maxTokens,
  }
}
