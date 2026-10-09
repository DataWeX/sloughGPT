'use client'

import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle, Button } from '@sloughgpt/strui'
import { downloadJson, importFile } from '@/lib/download-utils'
import { useToastStore } from '@/lib/toast-store'

interface SettingsBackupRestoreCardProps {
  settings: Record<string, unknown>
  onImport: (valid: Record<string, unknown>) => void
  version?: string
}

export function SettingsBackupRestoreCard({ settings, onImport, version }: SettingsBackupRestoreCardProps) {
  const addToast = useToastStore(s => s.addToast)

  const handleExport = () => {
    downloadJson(settings, 'sloughgpt-settings.json')
    addToast('Settings exported', 'success')
  }

  const handleImport = async () => {
    const file = await importFile('.json')
    if (!file) return
    try {
      const text = await file.text()
      const raw = JSON.parse(text)
      const valid: Record<string, unknown> = {}
      if (typeof raw.apiUrl === 'string') valid.apiUrl = raw.apiUrl
      if (typeof raw.hfToken === 'string') valid.hfToken = raw.hfToken
      if (typeof raw.defaultTemp === 'number') valid.defaultTemp = raw.defaultTemp
      if (typeof raw.defaultMaxTokens === 'number') valid.defaultMaxTokens = raw.defaultMaxTokens
      if (typeof raw.defaultTopP === 'number') valid.defaultTopP = raw.defaultTopP
      if (typeof raw.defaultTopK === 'number') valid.defaultTopK = raw.defaultTopK
      if (['dark', 'light', 'system'].includes(raw.theme)) valid.theme = raw.theme
      if (typeof raw.streaming === 'boolean') valid.streaming = raw.streaming
      if (typeof raw.customContext === 'string') valid.customContext = raw.customContext
      if (typeof raw.collapsibleMessageLength === 'number') valid.collapsibleMessageLength = raw.collapsibleMessageLength
      if (Object.keys(valid).length === 0) throw new Error('No valid settings found')
      onImport(valid)
      addToast('Settings imported', 'success')
    } catch {
      addToast('Invalid settings file', 'error')
    }
  }

  return (
    <Card>
      <CardHeader>
        <div>
          <CardTitle className="text-base">Backup & restore</CardTitle>
          <CardDescription>Export your settings to a file, or import from a backup</CardDescription>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={handleExport}>Export settings</Button>
          <Button size="sm" variant="outline" onClick={handleImport}>Import settings</Button>
        </div>
      </CardContent>
      <CardFooter className="justify-end">
        {version && <span className="text-[10px] font-mono px-1.5 py-0 h-5 text-muted-foreground border border-border/50 rounded text-xs">v{version}</span>}
      </CardFooter>
    </Card>
  )
}
