'use client'

import { createContext, useCallback, useContext, useState, useEffect, type ReactNode } from 'react'

const CONV_COLLAPSED_KEY = 'sloughgpt:conv-sidebar-collapsed'
const NAV_COLLAPSED_KEY = 'sloughgpt:nav-sidebar-collapsed'

function readBool(key: string): boolean {
  if (typeof window === 'undefined') return false
  try { return localStorage.getItem(key) === 'true' } catch { return false /* SSR or private browsing */ }
}

interface ConvSidebarContextValue {
  open: boolean
  toggle: () => void
  setOpen: (v: boolean) => void
  convCollapsed: boolean
  toggleConv: () => void
  setConvCollapsed: (v: boolean) => void
  navCollapsed: boolean
  toggleNav: () => void
  setNavCollapsed: (v: boolean) => void
}

const ConvSidebarContext = createContext<ConvSidebarContextValue | null>(null)

export function useConvSidebar() {
  const ctx = useContext(ConvSidebarContext)
  if (!ctx) throw new Error('useConvSidebar must be used within ConvSidebarProvider')
  return ctx
}

export function ConvSidebarProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false)
  const [convCollapsed, setConvCollapsed] = useState(() => readBool(CONV_COLLAPSED_KEY))
  const [navCollapsed, setNavCollapsed] = useState(() => readBool(NAV_COLLAPSED_KEY))
  const toggle = useCallback(() => setOpen(v => !v), [])
  const toggleConv = useCallback(() => setConvCollapsed(v => !v), [])
  const toggleNav = useCallback(() => setNavCollapsed(v => !v), [])

  useEffect(() => {
    try { localStorage.setItem(CONV_COLLAPSED_KEY, String(convCollapsed)) } catch { /* SSR or private browsing */ }
  }, [convCollapsed])

  useEffect(() => {
    try { localStorage.setItem(NAV_COLLAPSED_KEY, String(navCollapsed)) } catch { /* SSR or private browsing */ }
  }, [navCollapsed])

  return (
    <ConvSidebarContext.Provider value={{ open, toggle, setOpen, convCollapsed, toggleConv, setConvCollapsed, navCollapsed, toggleNav, setNavCollapsed }}>
      {children}
    </ConvSidebarContext.Provider>
  )
}
