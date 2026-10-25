'use client'

import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { useRouter } from 'next/navigation'
import { useLocale } from '@/hooks/useLocale'
import { useToastStore } from '@/lib/toast-store'
import { apiGet, apiPost } from '@/lib/http-client'
import { extractErrorMessage } from '@/lib/error-utils'
import { requestOverlayDockOpen, type OverlayWidget } from '@/components/overlay/OverlayDock'

/**
 * Consciousness quick actions as a dock tenant.
 *
 * v2: the standalone bottom-right FAB is gone — the M key lives in the
 * overlay rail (the decluttered corner), this panel opens beside the rail,
 * and Ctrl+Shift+Q asks the host to open it (`requestOverlayDockOpen`).
 * The host owns open state, outside-click and Esc; the panel itself is
 * tokenized (the old zinc/violet hardcodes are gone).
 */
export function useConsciousnessDockWidget(): OverlayWidget {
  const router = useRouter()
  const { t } = useLocale()
  const addToast = useToastStore((s) => s.addToast)
  const [loading, setLoading] = useState<string | null>(null)
  const [result, setResult] = useState<{ action: string; data: unknown } | null>(null)

  // Ctrl+Shift+Q — the dock opens this widget by id.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'Q') {
        e.preventDefault()
        requestOverlayDockOpen('consciousness')
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const runAction = useCallback(
    async (action: string, fn: () => Promise<unknown>) => {
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
    },
    [addToast, t],
  )

  const handleReflect = useCallback(() => {
    void runAction('reflect', () => apiPost('/consciousness/reflect'))
  }, [runAction])

  const handleSeed = useCallback(() => {
    void runAction('seed', () => apiPost('/consciousness/seed?count=5'))
  }, [runAction])

  const handleStatus = useCallback(() => {
    void runAction('status', () => apiGet('/consciousness/status'))
  }, [runAction])

  const handleHealth = useCallback(() => {
    void runAction('health', () => apiGet('/consciousness/health'))
  }, [runAction])

  return useMemo(() => {
    const renderPanel = (close: () => void): ReactNode => (
      <div className="space-y-2" data-testid="consciousness-panel">
        <div className="flex items-center justify-between border-b border-border px-1 pb-2">
          <span className="text-sm font-semibold text-foreground">
            {t('consciousness_quick_actions.title')}
          </span>
          <span className="rounded border border-border px-1.5 py-0.5 text-[10px] text-muted-foreground">
            Ctrl+Shift+Q
          </span>
        </div>

        <div className="space-y-1">
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

          <div className="my-1 border-t border-border" />

          <button
            type="button"
            onClick={() => {
              router.push('/consciousness/dashboard')
              close()
            }}
            className="w-full cursor-pointer rounded-lg px-3 py-2 text-left text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            {t('consciousness_quick_actions.open_dashboard')}
          </button>
          <button
            type="button"
            onClick={() => {
              router.push('/chat')
              close()
            }}
            className="w-full cursor-pointer rounded-lg px-3 py-2 text-left text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            {t('consciousness_quick_actions.open_chat')}
          </button>
        </div>

        {result ? (
          <div className="max-h-48 overflow-auto border-t border-border px-1 pt-2">
            <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
              {t(`consciousness_quick_actions.label_${result.action}`)}
            </div>
            <pre className="whitespace-pre-wrap break-all font-mono text-xs text-muted-foreground">
              {JSON.stringify(result.data, null, 2)}
            </pre>
          </div>
        ) : null}
      </div>
    )

    return {
      id: 'consciousness',
      label: t('consciousness_quick_actions.toggle'),
      icon: (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="currentColor"
          className="h-5 w-5"
          aria-hidden
        >
          <path d="M12 2C6.477 2 2 6.477 2 12s4.477 10 10 10 10-4.477 10-10S17.523 2 12 2zm0 2a8 8 0 110 16 8 8 0 010-16zm-1 3v2h2V7h-2zm0 4v6h2v-6h-2z" />
        </svg>
      ),
      tone: 'primary',
      render: renderPanel,
    }
  }, [t, loading, router, handleReflect, handleSeed, handleStatus, handleHealth])
}

function QuickAction({
  label,
  loading,
  onClick,
}: {
  label: string
  loading: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={loading}
      className="flex w-full cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-left text-sm text-foreground transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
    >
      {loading ? (
        <span className="h-3 w-3 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      ) : null}
      <span>{label}</span>
    </button>
  )
}
