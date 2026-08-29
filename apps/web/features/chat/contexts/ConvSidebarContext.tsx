'use client'

import { createContext, useCallback, useContext, useState, useEffect, type ReactNode } from 'react'
import { chatDB } from '@/lib/db'

const CONV_COLLAPSED_KEY = 'sloughgpt:conv-sidebar-collapsed'
const NAV_COLLAPSED_KEY = 'sloughgpt:nav-sidebar-collapsed'

async function readBool(key: string): Promise<boolean> {
  try {
    const entry = await chatDB.getKV<string>(key)
    return entry === 'true'
  } catch {
    return false
  }
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
  const [convCollapsed, setConvCollapsed] = useState(false)
  const [navCollapsed, setNavCollapsed] = useState(false)
  const toggle = useCallback(() => setOpen(v => !v), [])
  const toggleConv = useCallback(() => setConvCollapsed(v => !v), [])
  const toggleNav = useCallback(() => setNavCollapsed(v => !v), [])

  useEffect(() => {
    let cancelled = false
    readBool(CONV_COLLAPSED_KEY).then(v => { if (!cancelled) setConvCollapsed(v) })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    let cancelled = false
    readBool(NAV_COLLAPSED_KEY).then(v => { if (!cancelled) setNavCollapsed(v) })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    chatDB.setKV(CONV_COLLAPSED_KEY, String(convCollapsed)).catch(() => {})
  }, [convCollapsed])

  useEffect(() => {
    chatDB.setKV(NAV_COLLAPSED_KEY, String(navCollapsed)).catch(() => {})
  }, [navCollapsed])

  return (
    <ConvSidebarContext.Provider value={{ open, toggle, setOpen, convCollapsed, toggleConv, setConvCollapsed, navCollapsed, toggleNav, setNavCollapsed }}>
      {children}
    </ConvSidebarContext.Provider>
  )
}
