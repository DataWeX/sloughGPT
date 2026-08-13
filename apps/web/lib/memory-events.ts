/**
 * Lightweight pub/sub for memory lifecycle events.
 *
 * The chat loop publishes a memory event each time a chat turn stores (or
 * skips storing) a new fact. Panels that surface memory (e.g. the MemoryTab
 * tool panel) subscribe so their data refreshes without threading callbacks
 * through the component tree.
 */

export interface MemoryEventInfo {
  stored: boolean
  /** The newly stored fact text (first of the turn); undefined when skipped. */
  fact?: string
  /** All fact texts stored this turn (may be multiple); undefined when skipped. */
  facts?: string[]
}

type MemoryEventListener = (info: MemoryEventInfo) => void

const listeners = new Set<MemoryEventListener>()

/**
 * Subscribe to memory events.
 *
 * @param listener - called with each published event info.
 * @returns an unsubscribe function; call it on cleanup.
 */
export function subscribeMemoryEvents(listener: MemoryEventListener): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

/**
 * Publish a memory event to all subscribers.
 *
 * @param info - whether a new fact was stored this turn.
 * Side effects:
 * - invokes every subscribed listener; a throwing listener is isolated so it
 *   never breaks the remaining subscribers.
 */
export function publishMemoryEvent(info: MemoryEventInfo): void {
  for (const listener of listeners) {
    try {
      listener(info)
    } catch {
      // one broken listener must not break the others
    }
  }
}
