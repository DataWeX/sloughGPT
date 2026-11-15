'use client'
import { notify } from '@/lib/notify'

import { useEffect, useState, useCallback, useRef } from 'react'
import { useRouter } from '@/vite/next-compat/navigation'
import { consciousnessController } from '@/lib/consciousness-controller'
// Module scope: `addToast` must keep a stable identity — it sits in
// useCallback/useEffect dependency arrays, and a per-render arrow would
// re-fire those effects on every render.
const addToast = (message: string, type?: string) =>
  notify.toast(message, { type: type as 'success' | 'error' | 'info' })

type ActionName = 'toggle' | 'dashboard' | 'personality' | 'quickactions' | null

export function useConsciousnessShortcuts(enabled: boolean = true) {
  const router = useRouter()
  const [lastAction, setLastAction] = useState<ActionName>(null)
  const [registered, setRegistered] = useState(enabled)
  const lastActionRef = useRef<ActionName>(null)

  const registerShortcuts = useCallback((enable: boolean) => {
    setRegistered(enable)
  }, [])

  useEffect(() => {
    if (!registered) return

    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      const isInput =
        target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable
      const ctrl = e.ctrlKey || e.metaKey

      if (!ctrl || !e.shiftKey) return
      if (isInput) return

      // Ctrl+Shift+R is Chrome's hard-reload shortcut — never intercept it
      if (e.key === 'R') return

      const action: ActionName =
        e.key === 'C'
          ? 'toggle'
          : e.key === 'D'
            ? 'dashboard'
            : e.key === 'P'
              ? 'personality'
              : e.key === 'Q'
                ? 'quickactions'
                : null

      if (!action) return

      e.preventDefault()
      lastActionRef.current = action
      setLastAction(action)

      if (action === 'toggle') {
        notify.toast('Toggling consciousness...', { type: 'info' })
        consciousnessController
          .getStatus()
          .then((status) => {
            const currentLevel = (status as any)?.level ?? 0
            const newLevel = currentLevel === 0 ? 1 : 0
            return consciousnessController.updateConfig({ level: newLevel })
          })
          .then((result) => {
            const level = (result as any)?.level
            notify.toast(
              level ? `Consciousness enabled (level ${level})` : 'Consciousness disabled',
              { type: 'success' },
            )
          })
          .catch(() => {
            notify.toast('Failed to toggle consciousness', { type: 'error' })
          })
      }

      if (action === 'dashboard') {
        router.push('/consciousness/dashboard')
      }

      if (action === 'personality') {
        router.push('/consciousness/personality')
      }

      if (action === 'quickactions') {
        window.dispatchEvent(new CustomEvent('toggle-whatsnew'))
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [registered, router])

  return { registerShortcuts, lastAction } as any
}
