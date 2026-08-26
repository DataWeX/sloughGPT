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
  searchInputRef?: React.RefObject<HTMLInputElement | null>
  handleSearchChange?: (v: string) => void
  onRenameConversation?: () => void
  onExportMarkdown?: () => void
  onDuplicateConversation?: () => void
  onToggleBookmarks?: () => void
  onCancelStream?: () => void
  onApproveTool?: () => void
  onDenyTool?: () => void
  onToggleSidebar?: () => void
  onAddNoteToLastMessage?: () => void
  onOpenNoteSearch?: () => void
  onOpenShortcuts?: () => void
}

export function useChatKeyboard(deps: KeyboardDeps) {
  const {
    loading, currentError, showSettings,
    setToolPanelOpen, setShowSettings, setLoading, setCurrentError,
    loadingRef, newChatRef, handleRegenerateRef,
    searchInputRef, handleSearchChange,
    onRenameConversation, onExportMarkdown,
    onDuplicateConversation, onToggleBookmarks,
    onCancelStream, onApproveTool, onDenyTool, onToggleSidebar,
    onAddNoteToLastMessage, onOpenNoteSearch, onOpenShortcuts,
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
          d.onCancelStream?.()
        } else if (d.currentError) {
          d.setCurrentError(null)
        } else if (d.showSettings) {
          d.setShowSettings(prev => !prev)
        }
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'b' && e.shiftKey) {
        e.preventDefault()
        d.setToolPanelOpen(prev => !prev)
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'b' && !e.shiftKey) {
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
      if ((e.metaKey || e.ctrlKey) && e.key === 'f' && e.shiftKey) {
        e.preventDefault()
        window.dispatchEvent(new CustomEvent('search-conversations'))
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'f' && !e.shiftKey) {
        e.preventDefault()
        d.searchInputRef?.current?.focus()
      }
      if (e.key === '/' && !(e.metaKey || e.ctrlKey) && !(e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement)) {
        e.preventDefault()
        d.searchInputRef?.current?.focus()
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'r' && e.shiftKey) {
        e.preventDefault()
        d.onRenameConversation?.()
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'e' && e.shiftKey) {
        e.preventDefault()
        d.onExportMarkdown?.()
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'd' && e.shiftKey) {
        e.preventDefault()
        d.onDuplicateConversation?.()
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'b' && e.shiftKey) {
        e.preventDefault()
        d.onToggleBookmarks?.()
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'y') {
        e.preventDefault()
        d.onApproveTool?.()
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'n') {
        e.preventDefault()
        d.onDenyTool?.()
      }
      if ((e.metaKey || e.ctrlKey) && e.key === '\\') {
        e.preventDefault()
        d.onToggleSidebar?.()
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'n' && e.shiftKey) {
        e.preventDefault()
        d.onAddNoteToLastMessage?.()
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'f' && e.shiftKey) {
        e.preventDefault()
        d.onOpenNoteSearch?.()
      }
      if ((e.metaKey || e.ctrlKey) && e.key === '/' && e.shiftKey) {
        e.preventDefault()
        d.onOpenShortcuts?.()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [loading, currentError, showSettings, setToolPanelOpen, setShowSettings, setLoading, setCurrentError, loadingRef, newChatRef, handleRegenerateRef])
}
