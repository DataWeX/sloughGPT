'use client'

import { useCallback, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Badge } from '@sloughgpt/strui'
import { IconTrash, IconDownload, IconUpload } from '@sloughgpt/strui'
import { usePhonemeStore, type HistoryEntry } from '@/lib/phoneme-store'
import { useToastStore } from '@/lib/toast-store'

function HistoryRow({ entry }: { entry: HistoryEntry }) {
  const time = new Date(entry.timestamp).toLocaleTimeString()
  const pct = (entry.score * 100).toFixed(0)

  return (
    <div className="flex items-center gap-3 p-2 rounded-lg hover:bg-muted/50 text-sm">
      <span className="text-muted-foreground w-16 shrink-0">{time}</span>
      <span className="font-medium">{entry.target}</span>
      <span className="text-muted-foreground">→</span>
      <span>{entry.spoken}</span>
      <Badge
        variant={entry.score >= 0.8 ? 'default' : entry.score >= 0.5 ? 'secondary' : 'destructive'}
        className="ml-auto"
      >
        {pct}%
      </Badge>
      <Badge variant="outline" className="w-8 justify-center">{entry.language.toUpperCase()}</Badge>
    </div>
  )
}

export default function HistoryCard() {
  const history = usePhonemeStore(s => s.history)
  const clearHistory = usePhonemeStore(s => s.clearHistory)
  const exportHistory = usePhonemeStore(s => s.exportHistory)
  const importHistory = usePhonemeStore(s => s.importHistory)
  const addToast = useToastStore(s => s.addToast)
  const fileRef = useRef<HTMLInputElement>(null)

  const handleExport = useCallback(() => {
    const json = exportHistory()
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'phoneme-history.json'
    a.click()
    URL.revokeObjectURL(url)
    addToast('History exported', 'success')
  }, [exportHistory, addToast])

  const handleImport = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      const json = reader.result as string
      importHistory(json)
      addToast('History imported', 'success')
    }
    reader.readAsText(file)
    e.target.value = ''
  }, [importHistory, addToast])

  const handleClear = useCallback(() => {
    clearHistory()
    addToast('History cleared', 'info')
  }, [clearHistory, addToast])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Pronunciation History</span>
          <span className="text-sm font-normal text-muted-foreground">{history.length} entries</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex gap-2">
          <Button variant="secondary" size="sm" onClick={handleExport}>
            <IconDownload className="mr-2 h-4 w-4" />
            Export
          </Button>
          <Button variant="secondary" size="sm" onClick={() => fileRef.current?.click()}>
            <IconUpload className="mr-2 h-4 w-4" />
            Import
          </Button>
          <Button variant="destructive" size="sm" onClick={handleClear}>
            <IconTrash className="mr-2 h-4 w-4" />
            Clear
          </Button>
          <input ref={fileRef} type="file" accept=".json" className="hidden" onChange={handleImport} />
        </div>

        <div className="space-y-1 max-h-[400px] overflow-y-auto">
          {history.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-8">No history yet. Score some pronunciation to get started.</p>
          ) : (
            history.map(entry => (
              <HistoryRow key={entry.id} entry={entry} />
            ))
          )}
        </div>
      </CardContent>
    </Card>
  )
}
