'use client'
import { notify } from '@/lib/notify'

import { useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'
// Module scope: `addToast` must keep a stable identity — it sits in
// useCallback/useEffect dependency arrays, and a per-render arrow would
// re-fire those effects on every render.
const addToast = (message: string, type?: string) =>
  notify.toast(message, { type: type as 'success' | 'error' | 'info' })

export default function ProgressExport() {
  const history = usePhonemeStore((s) => s.history)
  const addToHistory = usePhonemeStore((s) => s.addToHistory)

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
    notify.toast(`Exported ${history.length} entries`, { type: 'success' })
  }, [history])

  const importData = useCallback(() => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return

      try {
        const text = await file.text()
        const data: {
          history: Array<{
            target: string
            spoken: string
            language: string
            targetPhonemes: string[]
            spokenPhonemes: string[]
            score: number
          }>
        } = JSON.parse(text)

        if (!data.history || !Array.isArray(data.history)) {
          notify.toast('Invalid file format', { type: 'error' })
          return
        }

        const existingWords = new Set(history.map((h) => `${h.target}-${h.language}-${h.spoken}`))

        let imported = 0
        for (const entry of data.history) {
          const key = `${entry.target}-${entry.language}-${entry.spoken}`
          if (!existingWords.has(key)) {
            addToHistory({
              target: entry.target,
              spoken: entry.spoken,
              targetPhonemes: entry.targetPhonemes,
              spokenPhonemes: entry.spokenPhonemes,
              score: entry.score,
              language: entry.language,
            })
            imported++
          }
        }

        notify.toast(
          `Imported ${imported} new entries (${data.history.length - imported} duplicates skipped)`,
          { type: 'success' },
        )
      } catch {
        notify.toast('Failed to parse file', { type: 'error' })
      }
    }
    input.click()
  }, [history, addToHistory])

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
