'use client'

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from 'react'
import { chatDB } from '@/lib/db'

const CONV_COLLAPSED_KEY = 'sloughgpt:conv-sidebar-collapsed'
const NAV_COLLAPSED_KEY = 'sloughgpt:nav-sidebar-collapsed'

// Rapid toggles coalesce into a single PUT per key (matches store.ts).
const PERSIST_DEBOUNCE_MS = 300

async function readBool(key: string): Promise<boolean> {
  try {
    const entry = await chatDB.getKV<string>(key)
    return entry === 'true'
  } catch {
    return false
  }
}

/**
 * Boolean state persisted to the docstore KV collection.
 *
 * Guards against the mount-time write storm: the initial read marks the key
 * hydrated, so no PUT fires until (a) hydration has settled and (b) the value
 * actually differs from what was read. Writes are debounced per key, and a
 * pending write is flushed on unmount so the last toggle is not lost.
 */
function usePersistedBool(key: string): [boolean, Dispatch<SetStateAction<boolean>>] {
  const [value, setValue] = useState(false)
  // Hydration is STATE, not a ref: a toggle landing before the read resolves
  // must still schedule its write once hydration completes.
  const [hydrated, setHydrated] = useState(false)
  const persisted = useRef<string | null>(null)
  // A user edit before hydration wins over the stored value.
  const touched = useRef(false)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const valueRef = useRef(value)
  valueRef.current = value

  const set: Dispatch<SetStateAction<boolean>> = useCallback((next) => {
    touched.current = true
    setValue(next)
  }, [])

  const flush = useCallback(() => {
    if (timer.current) {
      clearTimeout(timer.current)
      timer.current = null
    }
    if (persisted.current === null) return
    const serialized = String(valueRef.current)
    if (persisted.current === serialized) return
    persisted.current = serialized
    chatDB.setKV(key, serialized).catch(() => {})
  }, [key])

  useEffect(() => {
    let cancelled = false
    readBool(key).then((v) => {
      if (cancelled) return
      persisted.current = String(v)
      // Pre-hydration edit: keep the user's value and write it back instead.
      if (!touched.current) setValue(v)
      setHydrated(true)
    })
    return () => {
      cancelled = true
    }
  }, [key])

  useEffect(() => {
    if (!hydrated) return
    const serialized = String(value)
    if (persisted.current === serialized) return
    if (timer.current) clearTimeout(timer.current)
    timer.current = setTimeout(flush, PERSIST_DEBOUNCE_MS)
  }, [value, hydrated, flush])

  // Unmount flush: a pending toggle must still reach the server.
  useEffect(() => () => flush(), [flush])

  return [value, set]
}

interface ConvSidebarContextValue {
  open: boolean
  toggle: () => void
  setOpen: (v: boolean) => void
  convCollapsed: boolean
  toggleConv: () => void
  setConvCollapsed: Dispatch<SetStateAction<boolean>>
  navCollapsed: boolean
  toggleNav: () => void
  setNavCollapsed: Dispatch<SetStateAction<boolean>>
}

const ConvSidebarContext = createContext<ConvSidebarContextValue | null>(null)

export function useConvSidebar() {
  const ctx = useContext(ConvSidebarContext)
  if (!ctx) throw new Error('useConvSidebar must be used within ConvSidebarProvider')
  return ctx
}

export function ConvSidebarProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false)
  const [convCollapsed, setConvCollapsed] = usePersistedBool(CONV_COLLAPSED_KEY)
  const [navCollapsed, setNavCollapsed] = usePersistedBool(NAV_COLLAPSED_KEY)
  const toggle = useCallback(() => setOpen((v) => !v), [])
  const toggleConv = useCallback(() => setConvCollapsed((v) => !v), [setConvCollapsed])
  const toggleNav = useCallback(() => setNavCollapsed((v) => !v), [setNavCollapsed])

  return (
    <ConvSidebarContext.Provider
      value={{
        open,
        toggle,
        setOpen,
        convCollapsed,
        toggleConv,
        setConvCollapsed,
        navCollapsed,
        toggleNav,
        setNavCollapsed,
      }}
    >
      {children}
    </ConvSidebarContext.Provider>
  )
}
