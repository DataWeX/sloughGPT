'use client'

import { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'

interface StartupRecord {
  timestamp: number
  total_duration: number
  stage_durations: Record<string, number>
  hook_durations: Record<string, number>
  success: boolean
}

interface StartupHistoryChartProps {
  className?: string
}

export function StartupHistoryChart({ className }: StartupHistoryChartProps) {
  const [records, setRecords] = useState<StartupRecord[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await fetch('/health/startup-history')
        const json = await res.json()
        setRecords(json.data?.records ?? [])
      } catch {
        // ignore
      } finally {
        setLoading(false)
      }
    }
    fetchHistory()
  }, [])

  if (loading) {
    return (
      <div className={cn('rounded-xl border border-white/[0.06] bg-[#0a0a0a] p-4', className)}>
        <div className="h-9 -mx-4 -mt-4 mb-4 px-4 bg-[#1c1c1e] border-b border-white/[0.06] flex items-center">
          <span className="text-[11px] font-medium text-[#8e8e93]">Startup History</span>
        </div>
        <div className="h-32 flex items-center justify-center">
          <span className="text-[11px] text-[#636366]">Loading...</span>
        </div>
      </div>
    )
  }

  if (records.length === 0) {
    return (
      <div className={cn('rounded-xl border border-white/[0.06] bg-[#0a0a0a] p-4', className)}>
        <div className="h-9 -mx-4 -mt-4 mb-4 px-4 bg-[#1c1c1e] border-b border-white/[0.06] flex items-center">
          <span className="text-[11px] font-medium text-[#8e8e93]">Startup History</span>
        </div>
        <div className="h-32 flex items-center justify-center">
          <span className="text-[11px] text-[#636366]">No startup records yet</span>
        </div>
      </div>
    )
  }

  const maxDuration = Math.max(...records.map(r => r.total_duration))
  const avgDuration = records.reduce((sum, r) => sum + r.total_duration, 0) / records.length

  return (
    <div className={cn('rounded-xl border border-white/[0.06] bg-[#0a0a0a] overflow-hidden', className)}>
      <div className="h-9 px-4 bg-[#1c1c1e] border-b border-white/[0.06] flex items-center justify-between">
        <span className="text-[11px] font-medium text-[#8e8e93]">Startup History</span>
        <span className="text-[10px] text-[#636366] font-mono">{records.length} runs</span>
      </div>
      <div className="p-4">
        {/* Chart */}
        <div className="h-32 flex items-end gap-1">
          {records.slice(-20).map((record, i) => {
            const height = maxDuration > 0 ? (record.total_duration / maxDuration) * 100 : 0
            const isAboveAvg = record.total_duration > avgDuration
            return (
              <div
                key={`${record.timestamp}-${i}`}
                className="flex-1 flex flex-col items-center gap-1 group"
              >
                {/* Tooltip */}
                <div className="opacity-0 group-hover:opacity-100 transition-opacity absolute -mt-8 bg-[#1c1c1e] border border-white/[0.06] rounded px-2 py-1 text-[9px] font-mono text-[#c7c7cc] pointer-events-none z-10">
                  {record.total_duration.toFixed(1)}s
                </div>
                {/* Bar */}
                <div
                  className={cn(
                    'w-full rounded-t transition-all duration-200',
                    record.success
                      ? isAboveAvg
                        ? 'bg-[#febc2e]'
                        : 'bg-[#28c840]'
                      : 'bg-[#ff5f57]',
                  )}
                  style={{ height: `${Math.max(4, height)}%` }}
                />
              </div>
            )
          })}
        </div>

        {/* Average line */}
        <div className="relative h-px bg-[#636366]/30 mt-2">
          <div
            className="absolute left-0 right-0 h-px border-t border-dashed border-[#0a7aff]"
            style={{ top: 0 }}
          />
          <span className="absolute -top-3 right-0 text-[9px] text-[#0a7aff] font-mono">
            avg {avgDuration.toFixed(1)}s
          </span>
        </div>

        {/* Stats */}
        <div className="flex items-center justify-between mt-4 text-[10px]">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-sm bg-[#28c840]" />
              <span className="text-[#636366]">Below avg</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-sm bg-[#febc2e]" />
              <span className="text-[#636366]">Above avg</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-sm bg-[#ff5f57]" />
              <span className="text-[#636366]">Failed</span>
            </div>
          </div>
          <span className="text-[#636366] font-mono">
            Last: {records[records.length - 1]?.total_duration.toFixed(1)}s
          </span>
        </div>
      </div>
    </div>
  )
}
