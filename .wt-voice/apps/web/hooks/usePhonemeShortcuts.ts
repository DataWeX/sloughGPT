'use client'

import { useEffect, useCallback } from 'react'

export interface PhonemeShortcuts {
  onEncode?: () => void
  onScore?: () => void
  onCompare?: () => void
  onPractice?: () => void
  onBatch?: () => void
  onDetect?: () => void
  onQuiz?: () => void
  onSynthesize?: () => void
  onHistory?: () => void
  onRandomWord?: () => void
  onClear?: () => void
  onSubmit?: () => void
}

export function usePhonemeShortcuts(shortcuts: PhonemeShortcuts) {
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    // Ignore if user is typing in an input/textarea
    const target = e.target as HTMLElement
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
      return
    }

    const ctrl = e.ctrlKey || e.metaKey

    // Ctrl+1-9,0: Switch tabs
    if (ctrl && !e.shiftKey) {
      switch (e.key) {
        case '1': e.preventDefault(); shortcuts.onEncode?.(); break
        case '2': e.preventDefault(); shortcuts.onScore?.(); break
        case '3': e.preventDefault(); shortcuts.onCompare?.(); break
        case '4': e.preventDefault(); shortcuts.onPractice?.(); break
        case '5': e.preventDefault(); shortcuts.onBatch?.(); break
        case '6': e.preventDefault(); shortcuts.onDetect?.(); break
        case '7': e.preventDefault(); shortcuts.onQuiz?.(); break
        case '8': e.preventDefault(); shortcuts.onSynthesize?.(); break
        case '9': e.preventDefault(); shortcuts.onHistory?.(); break
        case '0': e.preventDefault(); shortcuts.onHistory?.(); break
      }
    }

    // Ctrl+Shift+R: Random word
    if (ctrl && e.shiftKey && e.key === 'R') {
      e.preventDefault()
      shortcuts.onRandomWord?.()
    }

    // Ctrl+Shift+X: Clear
    if (ctrl && e.shiftKey && e.key === 'X') {
      e.preventDefault()
      shortcuts.onClear?.()
    }

    // Ctrl+Enter: Submit current form
    if (ctrl && e.key === 'Enter') {
      e.preventDefault()
      shortcuts.onSubmit?.()
    }
  }, [shortcuts])

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])
}
