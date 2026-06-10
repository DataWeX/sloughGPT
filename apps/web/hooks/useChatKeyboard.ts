'use client'

import { useEffect, useRef } from 'react'

interface KeyboardDeps {
  loading: boolean
  currentError: unknown
  showSettings: boolean
  setToolPanelOpen: (fn: (prev: boolean) => boolean) => void
  setShowSettings: (fn: (prev: boolean) => boolean) => void
  setLoading: (v: boolean) => void
  setCurrentError: (v: null) => void
  loadingRef: React.MutableRefObject<AbortController | null>
  newChatRef: React.MutableRefObject<(() => void) | null>
  handleRegenerateRef: React.MutableRefObject<(() => Promise<void>) | null>
}

export function useChatKeyboard(deps: KeyboardDeps) {
  const {
    loading, currentError, showSettings,
    setToolPanelOpen, setShowSettings, setLoading, setCurrentError,
    loadingRef, newChatRef, handleRegenerateRef,
  } = deps

  const depsRef = useRef(deps)
  depsRef.current = deps

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const d = depsRef.current
      if (e.key === 'Escape') {
        if (d.loading && d.loadingRef.current) {
          d.loadingRef.current.abort()
          d.setLoading(false)
        } else if (d.currentError) {
          d.setCurrentError(null)
        } else if (d.showSettings) {
          d.setShowSettings(prev => !prev)
        }
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        d.setToolPanelOpen(prev => !prev)
      }
      if (e.key === '?' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        d.setShowSettings(prev => !prev)
      }
      if (e.key === 'n' && (e.metaKey || e.ctrlKey) && !e.shiftKey) {
        e.preventDefault()
        d.newChatRef.current?.()
      }
      if (e.key === 'r' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        d.handleRegenerateRef.current?.()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [loading, currentError, showSettings, setToolPanelOpen, setShowSettings, setLoading, setCurrentError, loadingRef, newChatRef, handleRegenerateRef])
}
