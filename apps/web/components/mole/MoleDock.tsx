'use client'

import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { useLocale } from '@/hooks/useLocale'
import { apiGet } from '@/lib/http-client'
import { logger } from '@/lib/dev-log'
import { IconHeart } from '@/components/icons/NavIcons'
import type { OverlayWidget } from '@/components/overlay/OverlayDock'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { ComponentHealthStrip } from './ComponentHealthStrip'
import { FindingsList } from './FindingsList'
import { MoleSummary } from './MoleSummary'
import { RunMoleButton } from './RunMoleButton'
import { SEVERITY_TONE } from './severity'
import type { MoleReport, MoleReportResponse } from './types'

/**
 * Site doctor (Mole) as a floating dock tenant.
 *
 * Mole is a tool, not a journey: read-only checks need to be *available*,
 * not navigated to — so the widget lives in the app-wide OverlayDock
 * instead of a /mole route (AGENTS.md Page Rule, "not a view" clause; the
 * operator tier of the external tooling pair, "Site Doctor").
 *
 * The panel reuses the page's summary/findings/run components unchanged —
 * this is composition, not a rewrite. One GET /mole/report per app shell
 * mount (light, read-only), refreshed after every successful run.
 */
export function useMoleDockWidget(): OverlayWidget {
  const { t } = useLocale()
  const { health, connectionStatus } = useLiveStatus()
  const [report, setReport] = useState<MoleReport | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const res = await apiGet<MoleReportResponse>('/mole/report')
      setReport(res?.report ?? null)
      setError(null)
    } catch (e) {
      logger.warning('mole report load failed', { error: String(e) })
      setError(e instanceof Error ? e.message : 'Could not load the mole report')
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const overall = report?.overall ?? 'ok'
  const counts = report?.summary?.by_severity
  // Badge = worst actionable count: critical first, then warnings; silent
  // when everything is fine (the ring stays green — status-LED behavior).
  const badge = counts?.critical
    ? String(counts.critical)
    : counts?.warn
      ? String(counts.warn)
      : null

  return useMemo(() => {
    const renderPanel = (close: () => void): ReactNode => (
      <div className="space-y-3" data-testid="mole-dock-panel">
        <header className="flex flex-wrap items-start justify-between gap-2">
          <div className="min-w-0">
            <h2 className="truncate text-sm font-semibold">{t('nav.mole')}</h2>
            <p className="text-xs text-muted-foreground">
              Read-only checks across the API, streams and journeys
            </p>
          </div>
          <RunMoleButton onCompleted={() => void load()} />
        </header>

        {error ? (
          <p role="alert" data-testid="mole-dock-error" className="text-xs text-destructive">
            {error}
          </p>
        ) : null}

        <ComponentHealthStrip health={health} connectionStatus={connectionStatus} dense />

        {report ? (
          <>
            <MoleSummary report={report} />
            <FindingsList findings={report.findings ?? []} />
          </>
        ) : error ? null : (
          <p
            data-testid="mole-dock-empty"
            className="py-4 text-center text-sm text-muted-foreground"
          >
            No report yet — run a check.
          </p>
        )}

        <button
          type="button"
          onClick={close}
          data-testid="mole-dock-close"
          className="w-full cursor-pointer rounded-md border border-border/60 px-2 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          Close
        </button>
      </div>
    )

    return {
      id: 'mole',
      label: t('nav.mole'),
      icon: <IconHeart className="h-5 w-5" aria-hidden />,
      badge,
      tone: report ? SEVERITY_TONE[overall] : 'default',
      role: 'operator',
      render: renderPanel,
    }
  }, [t, badge, overall, report, error, load, health, connectionStatus])
}
