'use client'

import { useEffect } from 'react'

/**
 * Registers "r" key as a refresh shortcut.
 * Skips when focus is inside an input or textarea.
 */
export function useRefreshShortcut(onRefresh: () => void) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'r' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault()
        onRefresh()
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onRefresh])
}
