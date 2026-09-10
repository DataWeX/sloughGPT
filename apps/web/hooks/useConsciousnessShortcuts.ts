'use client'

import { useEffect, useState, useCallback, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { PUBLIC_API_URL } from '@/lib/config'
import { useToastStore } from '@/lib/toast-store'

type ActionName = 'toggle' | 'reflect' | 'dashboard' | 'personality' | null

export function useConsciousnessShortcuts(enabled: boolean = true) {
  const router = useRouter()
  const addToast = useToastStore(s => s.addToast)
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
      const isInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable
      const ctrl = e.ctrlKey || e.metaKey

      if (!ctrl || !e.shiftKey) return
      if (isInput) return

      const action: ActionName = e.key === 'C' ? 'toggle'
        : e.key === 'R' ? 'reflect'
        : e.key === 'D' ? 'dashboard'
        : e.key === 'P' ? 'personality'
        : null

      if (!action) return

      e.preventDefault()
      lastActionRef.current = action
      setLastAction(action)

      if (action === 'toggle') {
        addToast('Toggling consciousness...', 'info')
        fetch(`${PUBLIC_API_URL}/consciousness/config`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ level: 1 }),
        }).then(async res => {
          if (!res.ok) {
            const body = await res.json().catch(() => ({}))
            const currentLevel = body?.data?.level ?? 0
            const newLevel = currentLevel === 0 ? 1 : 0
            return fetch(`${PUBLIC_API_URL}/consciousness/config`, {
              method: 'PATCH',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ level: newLevel }),
            })
          }
          return res.json()
        }).then(res => {
          if (!res) return
          return res.json?.() ?? res
        }).then(body => {
          const level = body?.data?.level
          addToast(level ? `Consciousness enabled (level ${level})` : 'Consciousness disabled', 'success')
        }).catch(() => {
          addToast('Failed to toggle consciousness', 'error')
        })
      }

      if (action === 'reflect') {
        addToast('Triggering reflection...', 'info')
        fetch(`${PUBLIC_API_URL}/consciousness/reflect`, { method: 'POST' })
          .then(res => {
            if (!res.ok) throw new Error('reflect failed')
            addToast('Reflection triggered', 'success')
          })
          .catch(() => {
            addToast('Failed to trigger reflection', 'error')
          })
      }

      if (action === 'dashboard') {
        router.push('/consciousness/dashboard')
      }

      if (action === 'personality') {
        router.push('/personality')
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [registered, router, addToast])

  return { registerShortcuts, lastAction } as any
}
