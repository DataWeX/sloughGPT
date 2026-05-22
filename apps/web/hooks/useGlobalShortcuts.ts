'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'

const NAV_SHORTCUTS: Record<string, string> = {
  '1': '/chat',
  '2': '/models',
  '3': '/datasets',
  '4': '/training',
  '5': '/settings',
}

export function useGlobalShortcuts() {
  const router = useRouter()

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      const isInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable
      const ctrl = e.ctrlKey || e.metaKey

      // If shortcuts dialog is open, close it before a non-? shortcut action
      const shortcutsOpen = () => document.querySelector('[data-shortcuts-open]')
      const closeIfOpen = () => {
        if (shortcutsOpen()) window.dispatchEvent(new CustomEvent('toggle-shortcuts'))
      }

      if (ctrl && !e.shiftKey && !e.altKey) {
        if (NAV_SHORTCUTS[e.key]) {
          e.preventDefault()
          closeIfOpen()
          router.push(NAV_SHORTCUTS[e.key])
          return
        }
        if (e.key === 'n') {
          e.preventDefault()
          closeIfOpen()
          window.dispatchEvent(new CustomEvent('new-chat'))
          return
        }
      }

      if (ctrl && e.shiftKey && e.key === 'F') {
        e.preventDefault()
        closeIfOpen()
        window.dispatchEvent(new CustomEvent('search-conversations'))
        return
      }

      if (!isInput && e.key === '?' && !ctrl) {
        e.preventDefault()
        // `?` toggles the dialog itself — no closeIfOpen needed, toggle-shortcuts handles it
        window.dispatchEvent(new CustomEvent('toggle-shortcuts'))
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [router])
}
