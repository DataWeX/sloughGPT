'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'

const NAV_SHORTCUTS: Record<string, string> = {
  '1': '/chat',
  '2': '/training',
  '3': '/datasets',
  '4': '/models',
  '5': '/agents',
  '6': '/compare',
  '7': '/monitoring',
  '8': '/knowledge',
  '9': '/multimodal',
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

      if (ctrl && e.shiftKey && e.key === 'A') {
        e.preventDefault()
        closeIfOpen()
        router.push('/settings')
        return
      }

      if (ctrl && e.shiftKey && e.key === 'F') {
        e.preventDefault()
        closeIfOpen()
        window.dispatchEvent(new CustomEvent('search-conversations'))
        return
      }

      if (ctrl && e.shiftKey && e.key === 'N') {
        e.preventDefault()
        closeIfOpen()
        window.dispatchEvent(new CustomEvent('new-chat'))
        return
      }

      // Copy last assistant response
      if (ctrl && e.shiftKey && e.key === 'C') {
        e.preventDefault()
        window.dispatchEvent(new CustomEvent('copy-last-response'))
        return
      }

      // Toggle nav sidebar collapse: Ctrl+\
      if (ctrl && !e.shiftKey && e.key === '\\') {
        e.preventDefault()
        closeIfOpen()
        window.dispatchEvent(new CustomEvent('toggle-nav-sidebar'))
        return
      }

      // Toggle conversation sidebar collapse: Ctrl+Shift+\
      if (ctrl && e.shiftKey && e.key === '\\') {
        e.preventDefault()
        closeIfOpen()
        window.dispatchEvent(new CustomEvent('toggle-conv-sidebar'))
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
