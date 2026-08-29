'use client'

import { useCallback, useEffect, useState } from 'react'
import { Button, Input } from '@sloughgpt/strui'
import { FoldSection } from '@sloughgpt/strui'
import { IconFilter, IconFolder, IconClock, IconDownload } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { memoryController, type MemoryArchiveStats } from '@/lib/memory-controller'
import { SectionErrorBoundary } from '@/components/SectionErrorBoundary'
import { todayDateString } from '@/lib/format-bytes'
import { downloadJson, importFile } from '@/lib/download-utils'
import { parseMemoryImport } from './memory-card-utils'

interface MemoryMaintenancePanelProps {
  itemCount: number
  archiveStats: MemoryArchiveStats | null
  loading: boolean
  fetchData: () => Promise<void>
  openArchive: () => void
}

export function MemoryMaintenancePanel({
  itemCount, archiveStats, loading, fetchData, openArchive,
}: MemoryMaintenancePanelProps) {
  const addToast = useToastStore(s => s.addToast)
  const [consolidating, setConsolidating] = useState(false)
  const [pruning, setPruning] = useState(false)
  const [retentionDays, setRetentionDays] = useState<number | null>(null)
  const [retentionLoading, setRetentionLoading] = useState(true)
  const [savingRetention, setSavingRetention] = useState(false)
  const [importing, setImporting] = useState(false)
  const [importProgress, setImportProgress] = useState<{ current: number; total: number } | null>(null)

  useEffect(() => {
    let active = true
    const load = async () => {
      setRetentionLoading(true)
      try {
        const config = await memoryController.getConfig()
        if (active) setRetentionDays(config.archive_retention_days ?? 30)
      } catch {
        if (active) setRetentionDays(null)
      } finally {
        if (active) setRetentionLoading(false)
      }
    }
    void load()
    return () => { active = false }
  }, [])

  const handleConsolidate = useCallback(async () => {
    setConsolidating(true)
    try {
      const result = await memoryController.consolidate()
      if (result.removed > 0) {
        addToast(`Consolidated ${result.removed} duplicate fact(s), kept ${result.kept}`, 'success')
      } else {
        addToast('No near-duplicate facts found', 'info')
      }
      await fetchData()
    } catch {
      addToast('Could not consolidate memory', 'error')
    } finally {
      setConsolidating(false)
    }
  }, [addToast, fetchData])

  const handlePruneArchive = useCallback(async () => {
    setPruning(true)
    try {
      const result = await memoryController.archivePrune(retentionDays ?? undefined)
      addToast(result.pruned > 0 ? `Pruned ${result.pruned} archive record(s)` : 'Archive already within retention', 'success')
      await fetchData()
    } catch {
      addToast('Could not prune archive', 'error')
    } finally {
      setPruning(false)
    }
  }, [addToast, fetchData, retentionDays])

  const handleSaveRetention = useCallback(async () => {
    if (retentionDays == null || Number.isNaN(retentionDays)) {
      addToast('Enter a retention window in days', 'error')
      return
    }
    const days = Math.round(Math.max(0, Math.min(retentionDays, 3650)))
    setSavingRetention(true)
    try {
      const config = await memoryController.updateConfig({ archive_retention_days: days })
      setRetentionDays(config.archive_retention_days)
      addToast(`Archive retention set to ${config.archive_retention_days} day(s)`, 'success')
    } catch {
      addToast('Could not save retention', 'error')
    } finally {
      setSavingRetention(false)
    }
  }, [retentionDays, addToast])

  const handleExportMemory = useCallback(async () => {
    try {
      const result = await memoryController.list(1000)
      const data = (result.items || []).map(i => ({ content: i.content, topic: i.topic || 'manual', source: i.source || 'api' }))
      downloadJson(data, `memory-export-${todayDateString()}.json`)
      addToast(`Exported ${data.length} memory item(s)`, 'success')
    } catch {
      addToast('Could not export memory', 'error')
    }
  }, [addToast])

  const handleImportMemory = useCallback(async () => {
    const file = await importFile('.json,.csv')
    if (!file) return
    setImporting(true)
    setImportProgress({ current: 0, total: 0 })
    try {
      const text = await file.text()
      const entries = parseMemoryImport(text, file.name)
      if (entries.length === 0) {
        addToast('No memory items found in file', 'error')
        return
      }
      setImportProgress({ current: 0, total: entries.length })
      let stored = 0
      for (let i = 0; i < entries.length; i++) {
        try {
          const result = await memoryController.store(entries[i].content, entries[i].topic)
          if (result.stored) stored++
        } catch { /* skip unimportable entry */ }
        setImportProgress({ current: i + 1, total: entries.length })
      }
      addToast(`Imported ${stored} of ${entries.length} memory item(s)`, 'success')
      await fetchData()
    } catch {
      addToast('Could not import memory', 'error')
    } finally {
      setImporting(false)
      setImportProgress(null)
    }
  }, [addToast, fetchData])

  return (
    <SectionErrorBoundary sectionName="Archive panel">
    <FoldSection heading={
      <span className="flex items-center gap-2">
        <IconFilter className="h-4 w-4 text-muted-foreground" />
        Maintenance
      </span>
    } className="mt-4">
      <div className="space-y-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="text-sm font-medium text-foreground flex items-center gap-1.5">
              <IconFilter className="h-3.5 w-3.5 text-muted-foreground" />
              Consolidate duplicates
            </p>
            <p className="text-xs mt-0.5">Merge near-identical facts, keeping the longest copy.</p>
          </div>
          <Button
            size="sm"
            variant="outline"
            className="h-8 text-xs shrink-0"
            onClick={handleConsolidate}
            disabled={consolidating || itemCount === 0}
          >
            {consolidating ? 'Merging…' : 'Consolidate'}
          </Button>
        </div>
        <div className="flex items-start justify-between gap-3 border-t border-border pt-3">
          <div className="min-w-0">
            <p className="text-sm font-medium text-foreground flex items-center gap-1.5">
              <IconFolder className="h-3.5 w-3.5 text-muted-foreground" />
              Provenance archive
            </p>
            <p className="text-xs mt-0.5">
              {loading ? 'Loading…' : (
                <>
                  {archiveStats?.records ?? 0} record(s), {archiveStats?.bytes != null ? `${(archiveStats.bytes / 1024).toFixed(1)} KB` : '—'}
                </>
              )}
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Button
              size="sm"
              variant="outline"
              className="h-8 text-xs shrink-0"
              onClick={openArchive}
            >
              View records
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="h-8 text-xs shrink-0"
              onClick={handlePruneArchive}
              disabled={pruning || (archiveStats?.records ?? 0) === 0}
            >
              {pruning ? 'Pruning…' : 'Prune old'}
            </Button>
          </div>
        </div>
        <div className="flex items-start justify-between gap-3 border-t border-border pt-3">
          <div className="min-w-0">
            <p className="text-sm font-medium text-foreground flex items-center gap-1.5">
              <IconClock className="h-3.5 w-3.5 text-muted-foreground" />
              Archive retention
            </p>
            <p className="text-xs mt-0.5">Pruning removes records older than this window (days).</p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Input
              type="number"
              min={0}
              max={3650}
              value={retentionDays ?? ''}
              onChange={e => setRetentionDays(e.target.value === '' ? null : Number(e.target.value))}
              placeholder={retentionLoading ? 'Loading…' : '30'}
              className="h-8 w-20 text-xs"
              aria-label="Archive retention days"
            />
            <Button
              size="sm"
              variant="outline"
              className="h-8 text-xs shrink-0"
              onClick={handleSaveRetention}
              disabled={savingRetention || retentionLoading}
            >
              {savingRetention ? 'Saving…' : 'Save'}
            </Button>
          </div>
        </div>
        <div className="flex items-start justify-between gap-3 border-t border-border pt-3">
          <div className="min-w-0">
            <p className="text-sm font-medium text-foreground flex items-center gap-1.5">
              <IconDownload className="h-3.5 w-3.5 text-muted-foreground" />
              Backup memory
            </p>
            <p className="text-xs mt-0.5">
              {importProgress ? `Importing ${importProgress.current}/${importProgress.total}…` : 'Export all facts as JSON, or import from a JSON/CSV backup.'}
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Button
              size="sm"
              variant="outline"
              className="h-8 text-xs shrink-0"
              onClick={handleExportMemory}
            >
              Export
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="h-8 text-xs shrink-0"
              onClick={handleImportMemory}
              disabled={importing}
            >
              {importing ? 'Importing…' : 'Import'}
            </Button>
          </div>
        </div>
      </div>
    </FoldSection>
    </SectionErrorBoundary>
  )
}
