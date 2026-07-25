'use client'

import { createContext, useCallback, useContext, useState, type ReactNode } from 'react'

interface ConvSidebarContextValue {
  open: boolean
  toggle: () => void
  setOpen: (v: boolean) => void
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
  const [navCollapsed, setNavCollapsed] = useState(false)
  const toggle = useCallback(() => setOpen(v => !v), [])
  const toggleNav = useCallback(() => setNavCollapsed(v => !v), [])

  return (
    <ConvSidebarContext.Provider value={{ open, toggle, setOpen, navCollapsed, toggleNav, setNavCollapsed }}>
      {children}
    </ConvSidebarContext.Provider>
  )
}
