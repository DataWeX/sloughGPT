'use client'

import { useCallback, useEffect, useState } from 'react'
import { cn, Button, Skeleton, EmptyCard } from '@sloughgpt/strui'
import {
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle,
} from '@sloughgpt/strui'
import { IconFolder, IconDownload, IconRefresh, IconChevronDown } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { memoryController, type MemoryArchiveStats, type MemoryArchiveRecord } from '@/lib/memory-controller'
import { todayDateString } from '@/lib/format-bytes'
import { downloadJson } from '@/lib/download-utils'
import { SectionErrorBoundary } from '@/components/SectionErrorBoundary'
import { archiveTypeLabel, archiveBadgeClass, archiveSummary } from './memory-card-utils'

interface ArchiveDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  archiveStats: MemoryArchiveStats | null
}

export function ArchiveDialog({ open, onOpenChange, archiveStats }: ArchiveDialogProps) {
  const addToast = useToastStore(s => s.addToast)
  const [records, setRecords] = useState<MemoryArchiveRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [expandedRecordId, setExpandedRecordId] = useState<string | null>(null)
  const [exporting, setExporting] = useState(false)

  const loadArchive = useCallback(async () => {
    setLoading(true)
    try {
      const result = await memoryController.archive(20)
      setRecords(result.records || [])
    } catch {
      addToast('Could not load archive records', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    if (open) loadArchive()
  }, [open, loadArchive])

  const handleOpen = useCallback((nextOpen: boolean) => {
    onOpenChange(nextOpen)
    if (nextOpen) loadArchive()
  }, [onOpenChange, loadArchive])

  const handleExport = useCallback(async () => {
    setExporting(true)
    try {
      const result = await memoryController.archive(1000)
      downloadJson(result.records || [], `memory-archive-${todayDateString()}.json`)
      addToast(`Exported ${result.records?.length ?? 0} archive record(s)`, 'success')
    } catch {
      addToast('Could not export archive', 'error')
    } finally {
      setExporting(false)
    }
  }, [addToast])

  return (
    <SectionErrorBoundary sectionName="Import dialog">
    <Dialog open={open} onOpenChange={handleOpen}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="text-base flex items-center gap-2">
            <IconFolder className="h-4 w-4 text-primary" />
            Provenance archive
          </DialogTitle>
          <DialogDescription>
            {archiveStats
              ? `${archiveStats.records} record(s) — ${(archiveStats.bytes / 1024).toFixed(1)} KB`
              : 'Task-backed memory records'}
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-1.5 max-h-[55vh] overflow-y-auto overscroll-contain pr-1">
          {loading && records.length === 0 ? (
            <div className="space-y-2 py-2">
              <Skeleton className="h-12 rounded-lg" />
              <Skeleton className="h-12 rounded-lg" />
              <Skeleton className="h-12 rounded-lg" />
            </div>
          ) : records.length === 0 ? (
            <EmptyCard
              message="No archive records yet"
              description="Task-backed memory records appear here as background tasks store facts."
              icon={<IconFolder className="h-5 w-5" />}
              action={null}
            />
          ) : (
            records.map(record => {
              const recordId = record.task_id || `${record.task_type}-${record.ts}`
              const summary = archiveSummary(record)
              const expanded = expandedRecordId === recordId
              return (
                <div key={recordId} className="rounded-lg border border-border/60">
                  <button
                    type="button"
                    onClick={() => setExpandedRecordId(expanded ? null : recordId)}
                    className="w-full text-left px-3 py-2 hover:bg-muted/40 transition-colors rounded-lg"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className={cn('text-[9px] px-1.5 py-0.5 rounded-full font-medium', archiveBadgeClass(record.task_type))}>
                        {archiveTypeLabel(record.task_type)}
                      </span>
                      <div className="flex items-center gap-2">
                        {record.ts > 0 && (
                          <span className="text-[9px] text-muted-foreground font-mono">
                            {new Date(record.ts * 1000).toLocaleString()}
                          </span>
                        )}
                        <IconChevronDown className={cn('h-3.5 w-3.5 text-muted-foreground transition-transform', expanded ? 'rotate-180' : '')} />
                      </div>
                    </div>
                    <p className="text-sm mt-1 line-clamp-2 break-words">{summary.text || '—'}</p>
                    {summary.detail && <p className="text-[10px] text-muted-foreground mt-0.5">{summary.detail}</p>}
                  </button>
                  {expanded && (
                    <pre className="text-[10px] font-mono leading-relaxed text-muted-foreground bg-muted/50 rounded-lg mx-3 mb-3 px-3 py-2 overflow-x-auto whitespace-pre-wrap break-words">
                      {JSON.stringify(record, null, 2)}
                    </pre>
                  )}
                </div>
              )
            })
          )}
        </div>
        <DialogFooter className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" className="h-8 text-xs" onClick={handleExport} disabled={exporting || records.length === 0}>
              <IconDownload className="h-3 w-3 mr-1" />
              {exporting ? 'Exporting…' : 'Export'}
            </Button>
            <Button size="sm" variant="outline" className="h-8 text-xs" onClick={loadArchive} disabled={loading}>
              <IconRefresh className={loading ? 'animate-spin h-3 w-3 mr-1' : 'h-3 w-3 mr-1'} />
              Refresh
            </Button>
          </div>
          <Button size="sm" className="h-8 text-xs" onClick={() => onOpenChange(false)}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
    </SectionErrorBoundary>
  )
}
