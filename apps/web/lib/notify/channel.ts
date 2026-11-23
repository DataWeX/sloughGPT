/**
 * channel — topology-aware store factory for the notify library.
 *
 * A channel wraps one zustand store whose state is `{ [key]: T[] }` plus its
 * dismiss/clear actions (exposed so existing selectors like `s.dismissToast`
 * keep resolving). Topology decides mutation shape:
 *
 *   stack — `push` enforces the bounded-stack rule via engine.makeRoom
 *   slot  — `put` replaces the whole channel (newest wins, single banner)
 *
 * Expiry for BOTH topologies runs on the engine's one shared ticker; a slot
 * entry with a deadline is swept exactly like a stack entry, which is what
 * makes supersede safety structural (the replacement carries its own
 * `dismissAt`) instead of timer bookkeeping.
 */

import { useStore } from 'zustand'
import { createStore } from 'zustand/vanilla'

import {
  isExpired,
  isTimed,
  makeRoom,
  registerProvider,
  syncTicker,
  type NotifyEntry,
  type Topology,
} from './engine'

export interface ChannelConfig<T> {
  /** State key the array is stored under (`toasts` / `banners`). */
  key: string
  topology: Topology
  /** Stack capacity (stack topology only). */
  max?: number
  /** State property name of the dismiss action (`dismissToast` …). */
  dismissName: string
  /** State property name of the clear action (`clearToasts` …). */
  clearName: string
  /** Phantom type for the entry — configures the public typings only. */
  phantom?: T
}

export interface Channel<T, S> {
  list(): T[]
  /** Stack: bounded append with eviction. */
  push(item: T): void
  /** Slot: replace everything (single-entry invariant). */
  put(item: T): void
  patch(id: string, patch: Partial<T>): void
  remove(id: string): void
  clear(): void
  getState(): S
  subscribe(listener: () => void): () => void
  /** Bound hook: selector sees the public state (key + dismiss/clear). */
  useChannel<Sel>(selector: (state: S) => Sel): Sel
}

export function createChannel<T extends NotifyEntry, S>(cfg: ChannelConfig<T>): Channel<T, S> {
  type Loose = S & Record<string, unknown>
  type State = Loose & { [key: string]: T[] | unknown }

  const items = (state: State): T[] => (state[cfg.key] as T[]) ?? []

  const store = createStore<State>((set) => {
    const remove = (id: string) => {
      set((prev) => ({ ...prev, [cfg.key]: items(prev).filter((e) => e.id !== id) }))
      syncTicker()
    }
    const clear = () => {
      set((prev) => ({ ...prev, [cfg.key]: [] }))
      syncTicker()
    }
    return {
      [cfg.key]: [],
      [cfg.dismissName]: remove,
      [cfg.clearName]: clear,
    } as State
  })

  // One provider per channel on the library's single ticker.
  registerProvider({
    needs: () => items(store.getState()).some(isTimed),
    sweep: (now) => {
      const current = items(store.getState())
      if (current.some((e) => isExpired(e, now))) {
        store.setState((prev) => ({
          ...prev,
          [cfg.key]: items(prev).filter((e) => !isExpired(e, now)),
        }))
      }
      syncTicker()
    },
  })

  const dismiss = store.getState()[cfg.dismissName] as (id: string) => void
  const clear = store.getState()[cfg.clearName] as () => void

  return {
    list: () => items(store.getState()),
    push: (item) => {
      store.setState((prev) => ({
        ...prev,
        [cfg.key]: [...makeRoom(items(prev), cfg.max ?? items(prev).length + 1), item],
      }))
      syncTicker()
    },

    put: (item) => {
      store.setState((prev) => ({ ...prev, [cfg.key]: [item] }))
      syncTicker()
    },

    patch: (id, patch) => {
      store.setState((prev) => ({
        ...prev,
        [cfg.key]: items(prev).map((e) => (e.id === id ? { ...e, ...patch } : e)),
      }))
      syncTicker()
    },

    remove: dismiss,
    clear,
    getState: () => store.getState() as S,
    subscribe: store.subscribe,

    useChannel: <Sel>(selector: (state: S) => Sel): Sel =>
      useStore(store, selector as (s: State) => Sel),
  } as Channel<T, S>
}
