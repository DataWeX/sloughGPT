'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { Button } from '@sloughgpt/strui'
import { useLocale } from '@/hooks/useLocale'
import { useToastStore } from '@/lib/toast-store'
import { apiGet, apiPost } from '@/lib/http-client'
import { extractErrorMessage } from '@/lib/error-utils'

export function ConsciousnessQuickActions() {
  const router = useRouter()
  const { t } = useLocale()
  const addToast = useToastStore(s => s.addToast)
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState<string | null>(null)
  const [result, setResult] = useState<{ action: string; data: any } | null>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)

  const handleToggle = useCallback(() => {
    setOpen(prev => !prev)
    setResult(null)
  }, [])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'Q') {
        e.preventDefault()
        handleToggle()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [handleToggle])

  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (
        panelRef.current &&
        !panelRef.current.contains(e.target as Node) &&
        buttonRef.current &&
        !buttonRef.current.contains(e.target as Node)
      ) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  const runAction = useCallback(async (action: string, fn: () => Promise<any>) => {
    setLoading(action)
    try {
      const data = await fn()
      setResult({ action, data })
      addToast(t(`consciousness_quick_actions.toast_${action}`), 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setLoading(null)
    }
  }, [addToast, t])

  const handleReflect = useCallback(() => {
    runAction('reflect', () => apiPost('/consciousness/reflect'))
  }, [runAction])

  const handleSeed = useCallback(() => {
    runAction('seed', () => apiPost('/consciousness/seed?count=5'))
  }, [runAction])

  const handleStatus = useCallback(() => {
    runAction('status', () => apiGet('/consciousness/status'))
  }, [runAction])

  const handleHealth = useCallback(() => {
    runAction('health', () => apiGet('/consciousness/health'))
  }, [runAction])

  return (
    <>
      <button
        ref={buttonRef}
        onClick={handleToggle}
        className="fixed bottom-6 right-6 z-[9999] h-12 w-12 rounded-full bg-violet-600 text-white shadow-lg transition-all hover:bg-violet-700 hover:shadow-xl hover:scale-110 active:scale-95 flex items-center justify-center"
        aria-label={t('consciousness_quick_actions.toggle')}
      >
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
          <path d="M12 2C6.477 2 2 6.477 2 12s4.477 10 10 10 10-4.477 10-10S17.523 2 12 2zm0 2a8 8 0 110 16 8 8 0 010-16zm-1 3v2h2V7h-2zm0 4v6h2v-6h-2z" />
        </svg>
      </button>

      {open && (
        <div
          ref={panelRef}
          className="fixed bottom-20 right-6 z-[9999] w-72 rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl overflow-hidden transition-all"
          style={{ animation: 'consciousness-quick-fadein 0.15s ease-out' }}
        >
          <div className="px-4 py-3 border-b border-zinc-700 flex items-center justify-between">
            <span className="text-sm font-semibold text-zinc-100">{t('consciousness_quick_actions.title')}</span>
            <span className="text-[10px] text-zinc-500">Ctrl+Shift+Q</span>
          </div>

          <div className="p-2 space-y-1">
            <QuickAction
              label={t('consciousness_quick_actions.reflect')}
              loading={loading === 'reflect'}
              onClick={handleReflect}
            />
            <QuickAction
              label={t('consciousness_quick_actions.seed')}
              loading={loading === 'seed'}
              onClick={handleSeed}
            />
            <QuickAction
              label={t('consciousness_quick_actions.status')}
              loading={loading === 'status'}
              onClick={handleStatus}
            />
            <QuickAction
              label={t('consciousness_quick_actions.health')}
              loading={loading === 'health'}
              onClick={handleHealth}
            />

            <div className="border-t border-zinc-700 my-1" />

            <button
              onClick={() => { router.push('/consciousness/dashboard'); setOpen(false) }}
              className="w-full px-3 py-2 text-left text-sm text-zinc-300 rounded-lg hover:bg-zinc-800 transition-colors"
            >
              {t('consciousness_quick_actions.open_dashboard')}
            </button>
            <button
              onClick={() => { router.push('/chat'); setOpen(false) }}
              className="w-full px-3 py-2 text-left text-sm text-zinc-300 rounded-lg hover:bg-zinc-800 transition-colors"
            >
              {t('consciousness_quick_actions.open_chat')}
            </button>
          </div>

          {result && (
            <div className="px-4 py-3 border-t border-zinc-700 max-h-48 overflow-auto">
              <div className="text-[10px] font-medium text-zinc-500 mb-1 uppercase tracking-wider">
                {t(`consciousness_quick_actions.label_${result.action}`)}
              </div>
              <pre className="text-xs text-zinc-400 whitespace-pre-wrap break-all font-mono">
                {JSON.stringify(result.data, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}

      <style>{`
        @keyframes consciousness-quick-fadein {
          from { opacity: 0; transform: translateY(8px) scale(0.95); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
      `}</style>
    </>
  )
}

function QuickAction({ label, loading, onClick }: { label: string; loading: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      className="w-full px-3 py-2 text-left text-sm text-zinc-300 rounded-lg hover:bg-zinc-800 disabled:opacity-50 transition-colors flex items-center gap-2"
    >
      {loading ? (
        <span className="h-3 w-3 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
      ) : null}
      <span>{label}</span>
    </button>
  )
}
