'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { useLocale } from '@/hooks/useLocale'
import { useConsciousnessNotifications } from '@/hooks/useConsciousnessNotifications'

const TYPE_COLORS: Record<string, string> = {
  info: 'text-blue-400',
  warning: 'text-amber-400',
  success: 'text-emerald-400',
  error: 'text-red-400',
}

const TYPE_ICONS: Record<string, JSX.Element> = {
  info: (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z" clipRule="evenodd" />
    </svg>
  ),
  warning: (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
      <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
    </svg>
  ),
  success: (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clipRule="evenodd" />
    </svg>
  ),
  error: (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.28 7.22a.75.75 0 00-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 101.06 1.06L10 11.06l1.72 1.72a.75.75 0 101.06-1.06L11.06 10l1.72-1.72a.75.75 0 00-1.06-1.06L10 8.94 8.28 7.22z" clipRule="evenodd" />
    </svg>
  ),
}

function timeAgo(ts: number): string {
  const seconds = Math.floor((Date.now() - ts) / 1000)
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h`
  const days = Math.floor(hours / 24)
  return `${days}d`
}

export function ConsciousnessNotificationsPanel() {
  const { t } = useLocale()
  const { notifications, unreadCount, markAsRead, clearAll } = useConsciousnessNotifications()
  const [open, setOpen] = useState(false)
  const panelRef = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)

  const toggle = useCallback(() => {
    setOpen(prev => !prev)
  }, [])

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

  return (
    <div className="relative">
      <button
        ref={buttonRef}
        onClick={toggle}
        className="relative p-2 text-zinc-400 hover:text-zinc-100 transition-colors"
        aria-label={t('consciousness_notifications.toggle')}
      >
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
          <path fillRule="evenodd" d="M10 2a6 6 0 00-6 6c0 1.887-.454 3.665-1.257 5.234a.75.75 0 00.515 1.076 32.91 32.91 0 003.256.508 3.5 3.5 0 006.972 0 32.903 32.903 0 003.256-.508.75.75 0 00.515-1.076A11.448 11.448 0 0116 8a6 6 0 00-6-6z" clipRule="evenodd" />
        </svg>
        {unreadCount > 0 && (
          <span className="absolute -top-0.5 -right-0.5 flex h-4 w-4 items-center justify-center rounded-full bg-violet-600 text-[10px] font-bold text-white">
            {unreadCount > 99 ? '99+' : unreadCount}
          </span>
        )}
      </button>

      {open && (
        <div
          ref={panelRef}
          className="absolute right-0 top-full mt-2 z-[9999] w-80 max-h-96 rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl overflow-hidden transition-all"
          style={{ animation: 'consciousness-notif-fadein 0.15s ease-out' }}
        >
          <div className="px-4 py-3 border-b border-zinc-700 flex items-center justify-between">
            <span className="text-sm font-semibold text-zinc-100">{t('consciousness_notifications.title')}</span>
            {notifications.length > 0 && (
              <button
                onClick={clearAll}
                className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
              >
                {t('consciousness_notifications.clear_all')}
              </button>
            )}
          </div>

          <div className="overflow-y-auto max-h-80">
            {notifications.length === 0 ? (
              <div className="px-4 py-8 text-center text-sm text-zinc-500">
                {t('consciousness_notifications.empty')}
              </div>
            ) : (
              notifications.map(notif => (
                <button
                  key={notif.id}
                  onClick={() => markAsRead(notif.id)}
                  className={`w-full text-left px-4 py-3 border-b border-zinc-800 hover:bg-zinc-800/50 transition-colors ${!notif.read ? 'bg-zinc-800/30' : ''}`}
                >
                  <div className="flex items-start gap-2.5">
                    <span className={`mt-0.5 flex-shrink-0 ${TYPE_COLORS[notif.type]}`}>
                      {TYPE_ICONS[notif.type]}
                    </span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-zinc-100 truncate">{notif.title}</span>
                        {!notif.read && (
                          <span className="flex-shrink-0 h-1.5 w-1.5 rounded-full bg-violet-500" />
                        )}
                        <span className="flex-shrink-0 text-[10px] text-zinc-500 ml-auto">{timeAgo(notif.timestamp)}</span>
                      </div>
                      <p className="text-xs text-zinc-400 mt-0.5 line-clamp-2">{notif.body}</p>
                    </div>
                  </div>
                </button>
              ))
            )}
          </div>
        </div>
      )}

      <style>{`
        @keyframes consciousness-notif-fadein {
          from { opacity: 0; transform: translateY(-4px) scale(0.95); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
      `}</style>
    </div>
  )
}
