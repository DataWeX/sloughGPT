'use client'

/**
 * Process-notification projector store (card 20261102_014254).
 *
 * Third notification tier — multi-slot, process-backed, action-first.
 * Distinct from the single-slot banner store (journey blockers) and the
 * toast store (transient confirmations): entries here represent live
 * processes, carry the process's REAL command surface, and persist until
 * the process state changes or the user acts.
 *
 * The contract (training/process_events.py on the backend):
 * - `event_id` is deterministic per process+event and is THE dedupe key —
 *   the boot REST snapshot and a later live transition for the same
 *   underlying event collapse to one entry (this fixes the toast store's
 *   fragile type:message dedupe).
 * - `actions` is the process's current command surface, projected from
 *   state — never frozen buttons, never dead ones.
 * - Dispatch re-validates server-side (fail-closed): a stale click
 *   degrades to `stale: true` ("no longer available"), never a wrong
 *   action against a live job.
 *
 * Boot events cannot ride SSE (no client connected when they fire), so
 * the initial state is the REST snapshot at /recovery/events; live
 * transitions after load ingest through the same projector.
 */

import { createStore } from 'zustand/vanilla'
import { useStore } from 'zustand'

import { apiGet } from '@/lib/http-client'
import { trainingController } from '@/lib/training-controller'

/**
 * Contract shape — GENERATED from training/process_events.py
 * (EVENT_SCHEMA / ACTION_SCHEMA, card 20261102_110421) via
 * scripts/gen_ts_contracts.py. Never hand-edit these here; regenerate.
 */
export type {
  ProcessAction,
  ProcessActionIntent,
  ProcessEvent,
  ProcessEventName,
  ProcessState,
} from '@/lib/protocol/contracts.gen'
import type { ProcessEvent } from '@/lib/protocol/contracts.gen'

export interface ProcessNotification extends ProcessEvent {
  /** Set after a successful dispatch (action ran, awaiting state sync). */
  dispatched?: boolean
  /** Server refused the action — it is no longer available. */
  stale?: boolean
}

interface ProcessNotificationStore {
  notifications: ProcessNotification[]
  /** Project events in: dedupe by event_id, update in place, drop settled
   * (empty action surface — the backend's settled-guard). */
  ingest: (events: ProcessEvent[]) => void
  dismiss: (eventId: string) => void
  clear: () => void
}

const processNotificationStore = createStore<ProcessNotificationStore>(
  (set) => ({
    notifications: [],

    ingest: (events) => {
      if (!events?.length) return
      // Settled guard: a projection with no actions carries no command
      // surface — the process is done and renders nothing.
      const settledIds = new Set(
        events.filter((e) => e.actions.length === 0).map((e) => e.event_id),
      )
      set((prev) => {
        // Start from existing minus anything this batch settles or
        // re-projects with a fresh action surface.
        const incoming = new Map(events.map((e) => [e.event_id, e]))
        const kept = prev.notifications.filter(
          (n) => !incoming.has(n.event_id) && !settledIds.has(n.event_id),
        )
        const updated = events
          .filter((e) => e.actions.length > 0)
          .map((e) => {
            const existing = prev.notifications.find((n) => n.event_id === e.event_id)
            // A re-projection is fresh server truth: drop stale/dispatched
            // marks so the action surface reflects CURRENT state.
            return { ...existing, ...e, dispatched: false, stale: false }
          })
        return { notifications: [...kept, ...updated] }
      })
    },

    dismiss: (eventId) => {
      set((prev) => ({
        notifications: prev.notifications.filter((n) => n.event_id !== eventId),
      }))
    },

    clear: () => set({ notifications: [] }),
  }),
)

export const useProcessNotifications = Object.assign(
  <T,>(selector: (state: ProcessNotificationStore) => T): T =>
    useStore(processNotificationStore, selector),
  {
    getState: processNotificationStore.getState,
    subscribe: processNotificationStore.subscribe,
  },
)

export function getProcessNotifications(): ProcessNotification[] {
  return processNotificationStore.getState().notifications
}

export function ingestProcessEvents(events: ProcessEvent[]): void {
  processNotificationStore.getState().ingest(events)
}

export function clearProcessNotifications(): void {
  processNotificationStore.getState().clear()
}

/**
 * Boot source: REST snapshot of durable state. SSE cannot deliver events
 * that fired before this client connected (boot-time auto-resume), so
 * the dock's initial state comes from here.
 */
export async function loadProcessSnapshot(): Promise<void> {
  try {
    const data = await apiGet<{
      contract_version?: number
      events?: ProcessEvent[]
    }>('/recovery/events')
    // Unknown contract version -> render nothing (fail-closed; a future
    // server must not be mis-rendered by an old client).
    if (!data || data.contract_version !== 1 || !Array.isArray(data.events)) return
    ingestProcessEvents(data.events)
  } catch {
    // Notification outage never blocks the app.
  }
}

/**
 * Dispatch an action against its event. Routing rules:
 * - dismiss: local removal, no network.
 * - navigate: the dock renders a link; no mutation to dispatch.
 * - recover/cancel: through the controller (http-client underneath) —
 *   the SERVER re-validates against current state and may refuse; that
 *   refusal degrades the entry to `stale` instead of pretending success.
 */
export async function dispatchAction(
  eventId: string,
  actionId: string,
): Promise<void> {
  const state = processNotificationStore.getState()
  const note = state.notifications.find((n) => n.event_id === eventId)
  if (!note) return
  const action = note.actions.find((a) => a.id === actionId)
  if (!action) return

  if (action.intent === 'dismiss') {
    state.dismiss(eventId)
    return
  }
  if (action.intent === 'navigate') {
    // Link navigation — handled by the dock's anchor, nothing to dispatch.
    return
  }

  try {
    if (action.intent === 'recover') {
      await trainingController.recover(note.process_id)
    } else if (action.intent === 'cancel') {
      await trainingController.stop(note.process_id)
    }
    processNotificationStore.setState((prev) => ({
      notifications: prev.notifications.map((n) =>
        n.event_id === eventId ? { ...n, dispatched: true, stale: false } : n,
      ),
    }))
  } catch {
    // Server refused (state moved on) — degrade honestly.
    processNotificationStore.setState((prev) => ({
      notifications: prev.notifications.map((n) =>
        n.event_id === eventId ? { ...n, dispatched: false, stale: true } : n,
      ),
    }))
  }
}
