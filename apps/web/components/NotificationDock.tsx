'use client'

/**
 * NotificationDock (card 20261102_014254) — third notification tier.
 *
 * MULTI-SLOT, process-backed, action-first — deliberately NOT the
 * single-slot GlobalBanner (journey blockers, one-banner invariant) and
 * not the toast store (transient confirmations). Each entry is a live
 * process: it renders the contract's structured action surface (the
 * process's real commands), the event's process link, and honest
 * lifecycle states — dispatched (ran, awaiting state sync) or stale
 * (server refused; the action is no longer available, so the dead button
 * is withdrawn rather than left to fail again).
 *
 * Positioned as a floating stack clear of the top banner and the mobile
 * bottom nav; tokens only (Noir Violet locked).
 */

import Link from '@/vite/next-compat/link'
import { memo } from 'react'

import { Button, IconX, StatusDot, type StatusDotTone } from '@sloughgpt/strui'
import {
  dispatchAction,
  useProcessNotifications,
  type ProcessAction,
  type ProcessNotification,
} from '@/lib/process-notifications'

const STATE_TONE: Record<string, StatusDotTone> = {
  interrupted: 'warning',
  failed: 'destructive',
  running: 'success',
  auto_resume_skipped: 'warning',
}

function actionKindClass(intent: ProcessAction['intent']): string {
  if (intent === 'recover' || intent === 'cancel') return 'text-primary'
  return 'text-muted-foreground'
}

const DockEntry = memo(function DockEntry({
  note,
}: {
  note: ProcessNotification
}) {
  const tone = STATE_TONE[note.state] ?? 'warning'
  const eventId = note.event_id
  const failed = note.stale === true
  const sent = note.dispatched === true

  return (
    <li
      className="flex w-full flex-col gap-1.5 border border-border bg-card p-3 shadow-lg"
      data-event-id={eventId}
      data-state={note.state}
    >
      <div className="flex items-center gap-1.5">
        <StatusDot tone={tone} pulse={!failed && !sent && note.state === 'running'} />
        <strong className="min-w-0 flex-1 truncate text-xs font-semibold text-foreground">
          {note.title}
        </strong>
        <Link
          href={note.process_link}
          className="shrink-0 text-[11px] text-muted-foreground underline-offset-2 hover:text-primary hover:underline"
        >
          View job
        </Link>
        <Button
          size="sm"
          variant="ghost"
          className="h-5 w-5 shrink-0"
          onClick={() => void dispatchAction(eventId, 'dismiss')}
          aria-label={`Dismiss: ${note.title}`}
        >
          <IconX className="h-3 w-3" aria-hidden="true" />
        </Button>
      </div>

      {note.name && (
        <p className="truncate text-[11px] text-muted-foreground">{note.name}</p>
      )}
      {note.reason && (
        <p className="text-[11px] text-muted-foreground">{note.reason}</p>
      )}

      {/* Lifecycle feedback — explicit, never inferred from content. */}
      {failed && (
        <p className="text-[11px] text-destructive">
          This action is no longer available — the process state changed.
        </p>
      )}
      {sent && !failed && (
        <p className="text-[11px] text-muted-foreground">
          Dispatched — waiting for the process to update.
        </p>
      )}

      {/* Action surface: the contract's REAL commands. A stale entry keeps
          only Dismiss (the one action that is always valid locally); the
          refused command is withdrawn instead of failing on click again. */}
      <div className="flex flex-wrap items-center gap-1">
        {note.actions
          .filter((a) => !failed || a.intent === 'dismiss')
          .map((a) => (
            <Button
              key={a.id}
              size="sm"
              variant="ghost"
              className={`h-6 text-[11px] ${actionKindClass(a.intent)}`}
              disabled={sent && (a.intent === 'recover' || a.intent === 'cancel')}
              onClick={() => void dispatchAction(eventId, a.id)}
            >
              {a.label}
            </Button>
          ))}
      </div>
    </li>
  )
})

export const NotificationDock = memo(function NotificationDock() {
  const notifications = useProcessNotifications((s) => s.notifications)
  if (notifications.length === 0) return null

  return (
    <div
      className="fixed bottom-20 right-4 z-[150] flex w-96 max-w-[calc(100vw-2rem)] flex-col gap-2 md:bottom-4"
      aria-live="polite"
      aria-label="Process notifications"
    >
      <ul className="flex flex-col gap-2">
        {notifications.map((n) => (
          <DockEntry key={n.event_id} note={n} />
        ))}
      </ul>
    </div>
  )
})
