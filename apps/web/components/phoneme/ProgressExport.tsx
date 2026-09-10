'use client'

import { useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'
import { useToastStore } from '@/lib/toast-store'

export default function ProgressExport() {
  const history = usePhonemeStore(s => s.history)
  const addToast = useToastStore(s => s.addToast)

  const exportData = useCallback(() => {
    const data = {
      version: 1,
      exportedAt: new Date().toISOString(),
      history,
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `phoneme-progress-${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
    addToast({ type: 'success', message: `Exported ${history.length} entries` })
  }, [history, addToast])

  const importData = useCallback(() => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return

      try {
        const text = await file.text()
        const data = JSON.parse(text)

        if (!data.history || !Array.isArray(data.history)) {
          addToast({ type: 'error', message: 'Invalid file format' })
          return
        }

        const store = usePhonemeStore.getState()
        const existing = store.history
        const existingWords = new Set(existing.map(h => `${h.targetWord}-${h.language}-${h.spokenWord}`))

        let imported = 0
        for (const entry of data.history) {
          const key = `${entry.targetWord}-${entry.language}-${entry.spokenWord}`
          if (!existingWords.has(key)) {
            store.addToHistory(entry)
            imported++
          }
        }

        addToast({ type: 'success', message: `Imported ${imported} new entries (${data.history.length - imported} duplicates skipped)` })
      } catch {
        addToast({ type: 'error', message: 'Failed to parse file' })
      }
    }
    input.click()
  }, [addToast])

  return (
    <Card>
      <CardHeader>
        <CardTitle>Export / Import</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex gap-3">
          <Button variant="outline" onClick={exportData} disabled={history.length === 0}>
            Export ({history.length} entries)
          </Button>
          <Button variant="outline" onClick={importData}>
            <IconRefresh className="w-4 h-4 mr-2" />
            Import
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-2">
          Export your practice history to back it up or import from another device.
        </p>
      </CardContent>
    </Card>
  )
}
