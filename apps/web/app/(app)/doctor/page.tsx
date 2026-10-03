'use client'

import { useCallback, useEffect, useState } from 'react'
import { EmptyCard, IconHeart } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { apiGet } from '@/lib/http-client'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { logger } from '@/lib/dev-log'
import { ComponentHealthStrip } from '@/components/doctor/ComponentHealthStrip'
import { DoctorSummary } from '@/components/doctor/DoctorSummary'
import { FindingsList } from '@/components/doctor/FindingsList'
import { RunDoctorButton } from '@/components/doctor/RunDoctorButton'
import type { DoctorReport, DoctorReportResponse } from '@/components/doctor/types'

/**
 * Site Doctor page — surfaces the read-only doctor report.
 *
 * GET /doctor/report on mount (and after every successful run), the live
 * component strip comes from the shared health stream.
 */
export default function DoctorPage() {
  const [report, setReport] = useState<DoctorReport | null>(null)
  const [ageS, setAgeS] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const { health, connectionStatus } = useLiveStatus()

  const load = useCallback(async (initial = false) => {
    if (initial) setLoading(true)
    try {
      const res = await apiGet<DoctorReportResponse>('/doctor/report')
      setReport(res?.report ?? null)
      setAgeS(res?.age_s ?? null)
      setError(null)
    } catch (e) {
      logger.warning('doctor report load failed', { error: String(e) })
      setError(e instanceof Error ? e.message : 'Could not load the doctor report')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load(true)
  }, [load])

  return (
    <PageContainer
      title="Site Doctor"
      subtitle="Read-only checks across the API, streams and journeys"
      headerRight={<RunDoctorButton onCompleted={() => void load()} />}
      loading={loading}
      loadingCards={2}
      error={error}
      onRetry={() => void load(true)}
    >
      <ComponentHealthStrip health={health} connectionStatus={connectionStatus} />

      {report ? (
        <>
          <DoctorSummary report={report} ageS={ageS} />
          <FindingsList findings={report.findings ?? []} />
        </>
      ) : (
        <EmptyCard
          message="No report yet — run a check"
          description="The doctor probes the live stack read-only, then writes a JSON report you can inspect here."
          icon={<IconHeart className="h-5 w-5" aria-hidden />}
        />
      )}
    </PageContainer>
  )
}
