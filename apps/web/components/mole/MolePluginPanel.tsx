'use client'

/**
 * MolePluginPanel — the Mole's UI, rendered on the Plugins page.
 *
 * This is the same surface the old /mole page shipped, rehomed as a
 * plugin-owned panel: it renders the live report only while the Mole
 * plugin is loaded AND enabled (POST /plugins/{name}/enable|disable),
 * and shows a disabled empty-state otherwise. Data path is unchanged —
 * GET /mole/report on mount and after every successful run.
 */

import { useCallback, useEffect, useState } from 'react'
import { Button, EmptyCard, IconHeart } from '@sloughgpt/strui'
import { apiGet } from '@/lib/http-client'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { logger } from '@/lib/dev-log'
import { ComponentHealthStrip } from './ComponentHealthStrip'
import { MoleSummary } from './MoleSummary'
import { FindingsList } from './FindingsList'
import { RunMoleButton } from './RunMoleButton'
import type { MoleReport, MoleReportResponse } from './types'

interface MolePluginPanelProps {
  /** The Mole plugin entry from GET /plugins; null while the list loads. */
  plugin: { enabled: boolean } | null | undefined
}

export function MolePluginPanel({ plugin }: MolePluginPanelProps) {
  const [report, setReport] = useState<MoleReport | null>(null)
  const [ageS, setAgeS] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { health, connectionStatus } = useLiveStatus()

  const enabled = plugin?.enabled ?? false
  const present = plugin != null

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const res = await apiGet<MoleReportResponse>('/mole/report')
      setReport(res?.report ?? null)
      setAgeS(res?.age_s ?? null)
      setError(null)
    } catch (e) {
      logger.warning('mole report load failed', { error: String(e) })
      setError(e instanceof Error ? e.message : 'Could not load the mole report')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (enabled) void load()
  }, [enabled, load])

  if (!present) return null

  if (!enabled) {
    return (
      <EmptyCard
        message="Mole plugin disabled"
        description="Enable the Mole plugin to run read-only checks across the API, streams and journeys."
        icon={<IconHeart className="h-5 w-5" aria-hidden />}
      />
    )
  }

  if (error) {
    return (
      <div role="alert" className="text-sm text-destructive">
        {error}
        <Button variant="outline" size="sm" className="ml-2" onClick={() => void load()}>
          Retry
        </Button>
      </div>
    )
  }

  return (
    <div className="space-y-4" data-testid="mole-plugin-panel">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold">Mole</h2>
        <RunMoleButton onCompleted={() => void load()} />
      </div>

      <ComponentHealthStrip health={health} connectionStatus={connectionStatus} />

      {report ? (
        <>
          <MoleSummary report={report} ageS={ageS} />
          <FindingsList findings={report.findings ?? []} />
        </>
      ) : (
        <EmptyCard
          message={loading ? 'Loading report…' : 'No report yet — run a check'}
          description="The mole probes the live stack read-only, then writes a JSON report you can inspect here."
          icon={<IconHeart className="h-5 w-5" aria-hidden />}
        />
      )}
    </div>
  )
}
