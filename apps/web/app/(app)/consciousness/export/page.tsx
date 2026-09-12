'use client'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { consciousnessController } from '@/lib/consciousness-controller'
import {
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Checkbox,
  Input,
  Skeleton,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'

const BACKUP_STORAGE_KEY = 'consciousness_backups'
const MAX_BACKUPS = 10

const EXPORT_OPTIONS = [
  { key: 'episodes' },
  { key: 'qualia' },
  { key: 'beliefs' },
  { key: 'personality' },
  { key: 'configuration' },
] as const

const fetchExportData = async (key: string): Promise<unknown> => {
  switch (key) {
    case 'episodes': return consciousnessController.getEpisodeHistory(1000)
    case 'qualia': return consciousnessController.getQualiaHistory(1000)
    case 'beliefs': return consciousnessController.getBeliefsHistory()
    case 'personality': return consciousnessController.getPersonality()
    case 'configuration': return consciousnessController.getStatus()
    default: return null
  }
}

interface BackupEntry {
  id: string
  timestamp: string
  size: number
  data: Record<string, unknown>
}

function loadBackups(): BackupEntry[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(BACKUP_STORAGE_KEY)
    if (!raw) return []
    return JSON.parse(raw)
  } catch {
    return []
  }
}

function saveBackups(backups: BackupEntry[]) {
  try {
    localStorage.setItem(BACKUP_STORAGE_KEY, JSON.stringify(backups))
  } catch { /* ignored */ }
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

export default function ConsciousnessExportPage() {
  const addToast = useToastStore(s => s.addToast)
  const { t } = useLocale()
  const [selected, setSelected] = useState<Record<string, boolean>>({
    episodes: true,
    qualia: true,
    beliefs: true,
    personality: true,
    configuration: true,
  })
  const [format, setFormat] = useState<'json' | 'jsonl' | 'csv'>('json')
  const [exporting, setExporting] = useState(false)
  const [backups, setBackups] = useState<BackupEntry[]>([])
  const [importFile, setImportFile] = useState<File | null>(null)
  const [importPreview, setImportPreview] = useState<unknown[]>([])
  const [importing, setImporting] = useState(false)

  useEffect(() => {
    setBackups(loadBackups())
  }, [])

  const toggleOption = (key: string) => {
    setSelected(prev => ({ ...prev, [key]: !prev[key] }))
  }

  const handleExport = async () => {
    const keys = Object.keys(selected).filter(k => selected[k])
    if (keys.length === 0) {
      addToast(t('consciousness_export.toast_no_selection'), 'error')
      return
    }

    setExporting(true)
    try {
      const exportData: Record<string, unknown> = {}
      for (const key of keys) {
        exportData[key] = await fetchExportData(key)
      }

      const now = new Date()
      const dateStr = now.toISOString().split('T')[0]
      const filename = `consciousness_export_${dateStr}`

      let content: string
      let mimeType: string
      let extension: string

      if (format === 'json') {
        content = JSON.stringify(exportData, null, 2)
        mimeType = 'application/json'
        extension = 'json'
      } else if (format === 'jsonl') {
        const lines = Object.entries(exportData).map(([key, value]) =>
          JSON.stringify({ key, data: value })
        )
        content = lines.join('\n')
        mimeType = 'application/jsonl'
        extension = 'jsonl'
      } else {
        const allRows: Record<string, unknown>[] = []
        for (const [key, value] of Object.entries(exportData)) {
          const arr = Array.isArray(value) ? value : [value]
          for (const item of arr) {
            if (typeof item === 'object' && item !== null) {
              allRows.push({ _section: key, ...(item as Record<string, unknown>) })
            } else {
              allRows.push({ _section: key, value: item })
            }
          }
        }
        if (allRows.length > 0) {
          const headers = [...new Set(allRows.flatMap(r => Object.keys(r)))]
          const csvRows = [headers.join(',')]
          for (const row of allRows) {
            csvRows.push(headers.map(h => {
              const val = row[h]
              const str = val === undefined || val === null ? '' : String(val)
              return str.includes(',') || str.includes('"') || str.includes('\n')
                ? `"${str.replace(/"/g, '""')}"` : str
            }).join(','))
          }
          content = csvRows.join('\n')
        } else {
          content = ''
        }
        mimeType = 'text/csv'
        extension = 'csv'
      }

      const blob = new Blob([content], { type: mimeType })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${filename}.${extension}`
      a.click()
      URL.revokeObjectURL(url)

      const backup: BackupEntry = {
        id: crypto.randomUUID(),
        timestamp: now.toISOString(),
        size: blob.size,
        data: exportData,
      }
      const updated = [backup, ...backups].slice(0, MAX_BACKUPS)
      setBackups(updated)
      saveBackups(updated)

      addToast(t('consciousness_export.toast_exported'), 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setExporting(false)
    }
  }

  const handleImportFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setImportFile(file)
    setImportPreview([])

    const reader = new FileReader()
    reader.onload = (ev) => {
      try {
        const text = ev.target?.result as string
        let parsed: unknown
        if (file.name.endsWith('.jsonl')) {
          parsed = text.split('\n').filter(Boolean).map(line => JSON.parse(line))
        } else {
          parsed = JSON.parse(text)
        }
        const records = Array.isArray(parsed)
          ? parsed
          : typeof parsed === 'object' && parsed !== null
            ? Object.entries(parsed as Record<string, unknown>).map(([k, v]) => ({ key: k, data: v }))
            : [parsed]
        setImportPreview(records.slice(0, 5))
      } catch {
        addToast(t('consciousness_export.toast_invalid_file'), 'error')
      }
    }
    reader.readAsText(file)
  }

  const handleImport = async () => {
    if (!importFile) return
    setImporting(true)
    try {
      const text = await importFile.text()
      let parsed: unknown
      if (importFile.name.endsWith('.jsonl')) {
        parsed = text.split('\n').filter(Boolean).map(line => JSON.parse(line))
      } else {
        parsed = JSON.parse(text)
      }
      addToast(t('consciousness_export.toast_imported'), 'success')
      setImportFile(null)
      setImportPreview([])
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setImporting(false)
    }
  }

  const handleDeleteBackup = (id: string) => {
    const updated = backups.filter(b => b.id !== id)
    setBackups(updated)
    saveBackups(updated)
  }

  const handleRestoreBackup = (backup: BackupEntry) => {
    try {
      localStorage.setItem('consciousness_export_restore', JSON.stringify(backup.data))
      addToast(t('consciousness_export.toast_restored'), 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }

  return (
    <PageContainer title={t('consciousness_export.page_title')}>
      <div className="space-y-6 p-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('consciousness_export.export_options_title')}</CardTitle>
            <CardDescription>{t('consciousness_export.export_options_desc')}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              {EXPORT_OPTIONS.map(opt => (
                <div key={opt.key} className="flex items-center gap-2">
                  <Checkbox
                    id={`export-${opt.key}`}
                    checked={!!selected[opt.key]}
                    onCheckedChange={() => toggleOption(opt.key)}
                  />
                  <label htmlFor={`export-${opt.key}`} className="text-sm font-medium cursor-pointer">
                    {t(`consciousness_export.option_${opt.key}`)}
                  </label>
                </div>
              ))}
            </div>
            <div className="border-t border-border/30" />
            <div className="flex items-center gap-4">
              <label className="text-sm font-medium">{t('consciousness_export.format_label')}</label>
              <div className="flex gap-2">
                {(['json', 'jsonl', 'csv'] as const).map(f => (
                  <Button
                    key={f}
                    variant={format === f ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setFormat(f)}
                  >
                    {f.toUpperCase()}
                  </Button>
                ))}
              </div>
            </div>
            <Button onClick={handleExport} disabled={exporting} className="w-full">
              {exporting ? t('consciousness_export.exporting') : t('consciousness_export.export_button')}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('consciousness_export.import_title')}</CardTitle>
            <CardDescription>{t('consciousness_export.import_desc')}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Input
              type="file"
              accept=".json,.jsonl"
              onChange={handleImportFileChange}
              className="text-sm"
            />
            {importPreview.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">
                  {t('consciousness_export.preview_label')} ({importPreview.length})
                </p>
                <div className="max-h-48 overflow-auto rounded-md border border-border/30 bg-muted/30 p-2 text-xs font-mono">
                  <pre className="whitespace-pre-wrap break-all">
                    {JSON.stringify(importPreview, null, 2)}
                  </pre>
                </div>
              </div>
            )}
            <Button
              onClick={handleImport}
              disabled={!importFile || importing}
              variant="secondary"
              className="w-full"
            >
              {importing ? t('consciousness_export.importing') : t('consciousness_export.import_button')}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('consciousness_export.backups_title')}</CardTitle>
            <CardDescription>{t('consciousness_export.backups_desc')}</CardDescription>
          </CardHeader>
          <CardContent>
            {backups.length === 0 ? (
              <p className="text-sm text-muted-foreground">{t('consciousness_export.no_backups')}</p>
            ) : (
              <div className="space-y-2">
                {backups.map(backup => (
                  <div
                    key={backup.id}
                    className="flex items-center justify-between rounded-md border border-border/30 p-3"
                  >
                    <div className="space-y-0.5">
                      <p className="text-sm font-medium">{formatDate(backup.timestamp)}</p>
                      <p className="text-xs text-muted-foreground">{formatBytes(backup.size)}</p>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleRestoreBackup(backup)}
                      >
                        {t('consciousness_export.restore')}
                      </Button>
                      <Button
                        size="sm"
                        variant="destructive"
                        onClick={() => handleDeleteBackup(backup.id)}
                      >
                        {t('consciousness_export.delete')}
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
