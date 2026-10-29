'use client'

import { useState } from 'react'
import { Button, cn } from '@sloughgpt/strui'
import { apiPost } from '@/lib/http-client'
import { useToastStore } from '@/lib/toast-store'
import type { DoctorReport } from './types'

interface RunDoctorButtonProps {
  /** Fired after a successful run so the page can refresh its report. */
  onCompleted?: (report: DoctorReport) => void
  className?: string
}

/**
 * Kicks off the LIGHT doctor run (http + sse + journey-from-disk) and
 * reports the result: transient success toast, inline error on failure.
 */
export function RunDoctorButton({ onCompleted, className }: RunDoctorButtonProps) {
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const addToast = useToastStore((s) => s.addToast)

  const run = async () => {
    if (running) return
    setRunning(true)
    setError(null)
    try {
      const res = await apiPost<{ report: DoctorReport }>(
        '/doctor/run',
        undefined,
        // No retries — a timed-out run must never fire a second sweep.
        { skipCircuitBreaker: true },
      )
      addToast('Doctor check complete', 'success')
      if (res?.report) onCompleted?.(res.report)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Doctor check failed')
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className={cn('flex flex-wrap items-center gap-2', className)}>
      <Button
        onClick={() => void run()}
        loading={running}
        loadingText="Running…"
        data-testid="run-doctor-button"
      >
        Run check
      </Button>
      {error ? (
        <span
          role="alert"
          data-testid="run-doctor-error"
          className="text-xs text-destructive"
        >
          {error}
        </span>
      ) : null}
    </div>
  )
}
