'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Textarea } from '@sloughgpt/strui'
import { Slider } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import { ToggleGroup as ToggleGroupRadix, ToggleGroupItem } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { useSettings, useUpdateSettings } from '@/lib/store'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { systemController, type DetailedHealth, type SystemMetrics, type DiskUsage, type SystemInfo } from '@/lib/system-controller'
import { formatUptime } from '@/lib/chat-utils'

function SettingsSlider({
  label, value, onChange, min, max, step, formatValue,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  formatValue?: (v: number) => string
}) {
  const display = formatValue ? formatValue(value) : value.toString()
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">{label}</label>
        <span className="text-sm text-muted-foreground">{display}</span>
      </div>
      <Slider value={[value]} onValueChange={([v]: number[]) => onChange(v)} min={min} max={max} step={step} />
    </div>
  )
}

export default function SettingsPage() {
  const settings = useSettings()
  const updateSettings = useUpdateSettings()
  const addToast = useToastStore(s => s.addToast)
  const { healthLegacy: apiHealth } = useLiveStatus()
  const [detailed, setDetailed] = useState<DetailedHealth | null>(null)
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
  const [disk, setDisk] = useState<DiskUsage | null>(null)
  const [info, setInfo] = useState<SystemInfo | null>(null)

  const fetchHealth = useCallback(async () => {
    const [d, m, dk, inf] = await Promise.allSettled([
      systemController.getDetailedHealth(),
      systemController.getMetrics(),
      systemController.getDisk(),
      systemController.getInfo(),
    ])
    if (d.status === 'fulfilled') setDetailed(d.value)
    if (m.status === 'fulfilled') setMetrics(m.value)
    if (dk.status === 'fulfilled') setDisk(dk.value)
    if (inf.status === 'fulfilled') setInfo(inf.value)
  }, [])

  useEffect(() => { fetchHealth() }, [fetchHealth])

  const isOnline = apiHealth !== null && apiHealth !== 'offline'
  const apiOk = isOnline && (apiHealth.status === 'healthy' || detailed?.status === 'healthy')
  const modelLoaded = isOnline && (apiHealth.model_loaded || detailed?.model_loaded)
  const modelType = isOnline ? (apiHealth.model_type || detailed?.model_type) : null

  const clearChat = () => {
    localStorage.removeItem('man_current_conversation')
    addToast('Chat history cleared', 'success')
  }

  const resetAllSettings = () => {
    updateSettings({
      apiUrl: 'http://localhost:8000',
      hfToken: '',
      defaultModel: 'gpt2',
      defaultTemp: 0.8,
      defaultMaxTokens: 200,
      theme: 'light',
      streaming: true,
      customContext: '',
      collapsibleMessageLength: 500,
    })
    addToast('Settings reset to defaults', 'success')
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Settings" />} />

      <div className="space-y-4">
        {/* Appearance */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Appearance</CardTitle>
            <CardDescription>Theme preference</CardDescription>
          </CardHeader>
          <CardContent>
            <ToggleGroupRadix type="single" value={settings.theme} onValueChange={(v) => v && updateSettings({ theme: v as 'light' | 'dark' | 'system' })}>
              <ToggleGroupItem value="light">Light</ToggleGroupItem>
              <ToggleGroupItem value="dark">Dark</ToggleGroupItem>
              <ToggleGroupItem value="system">System</ToggleGroupItem>
            </ToggleGroupRadix>
          </CardContent>
        </Card>

        {/* Chat defaults */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Chat defaults</CardTitle>
            <CardDescription>Default model and generation settings</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <SettingsSlider
                label="Temperature"
                value={settings.defaultTemp}
                onChange={(v) => updateSettings({ defaultTemp: v })}
                min={0}
                max={2}
                step={0.1}
              />
              <SettingsSlider
                label="Max tokens"
                value={settings.defaultMaxTokens}
                onChange={(v) => updateSettings({ defaultMaxTokens: v })}
                min={50}
                max={1000}
                step={50}
              />
            </div>
            <div className="pt-2">
              <SettingsSlider
                label="Auto-collapse messages longer than"
                value={settings.collapsibleMessageLength}
                onChange={(v) => updateSettings({ collapsibleMessageLength: v })}
                min={0}
                max={2000}
                step={50}
                formatValue={(v) => v === 0 ? 'Disabled' : `${v} chars`}
              />
            </div>
          </CardContent>
        </Card>

        {/* Memory */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Memory</CardTitle>
            <CardDescription>Custom instructions included with every prompt</CardDescription>
          </CardHeader>
          <CardContent>
            <Textarea
              className="min-h-[120px]"
              placeholder="e.g., You are a helpful coding assistant. Keep responses concise..."
              value={settings.customContext}
              onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => updateSettings({ customContext: e.target.value })}
              aria-label="Custom instructions"
            />
          </CardContent>
        </Card>

        {/* Chat commands reference */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Chat commands</CardTitle>
            <CardDescription>Type <kbd className="inline-flex items-center rounded border border-border/40 bg-muted px-1.5 py-0.5 text-[11px] font-mono">/</kbd> in chat to open the command palette</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
              {[
                ['/help', 'Show all commands'],
                ['/clear', 'Clear the chat'],
                ['/temp <n>', 'Set temperature (0–2)'],
                ['/model <name>', 'Switch model'],
                ['/soul <name>', 'Switch soul'],
                ['/export', 'Export chat as MD'],
                ['/file', 'Attach a file'],
                ['/knowledge <q>', 'Search knowledge'],
                ['/goto <path>', 'Navigate to page'],
                ['/summarize', 'Summarise chat'],
                ['/feedback +/- [r]', 'Rate response'],
                ['/translate <lang>', 'Translate reply'],
                ['/search <q>', 'Search conversations'],
                ['/archive', 'Archive & start fresh'],
                ['/rename <n>', 'Rename conversation'],
              ].map(([cmd, desc]) => (
                <div key={cmd} className="flex items-center gap-2">
                  <code className="text-xs text-primary font-mono shrink-0">{cmd}</code>
                  <span className="text-xs text-muted-foreground truncate">{desc}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* System health */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">System health</CardTitle>
            <CardDescription>Backend status and resource usage</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Top row: API + Model + Uptime + Responses */}
            <KpiGrid columns={4}>
              <StatCard
                label="API"
                value={<span className="font-mono">{apiOk ? 'Healthy' : 'Error'}</span>}
                icon={<span className={`inline-block w-2 h-2 rounded-full ${apiOk ? 'bg-success' : 'bg-destructive'}`} />}
              />
              <StatCard
                label="Model"
                value={<span className="font-mono text-xs">{modelLoaded ? (modelType || 'Loaded') : 'None'}</span>}
                icon={<span className={`inline-block w-2 h-2 rounded-full ${modelLoaded ? 'bg-success' : 'bg-muted-foreground/50'}`} />}
              />
              <StatCard
                label="Uptime"
                value={<span className="font-mono">{formatUptime(detailed?.uptime_seconds ?? 0)}</span>}
              />
              <StatCard
                label="Responses"
                value={<span className="font-mono">{String(detailed?.inference?.inference_count ?? 0)}</span>}
              />
            </KpiGrid>

            {/* Resource rows */}
            <div className="grid grid-cols-2 gap-4">
              <StatCard
                label="CPU"
                value={<span className="font-mono">{metrics ? `${metrics.cpu_percent}%` : '...'}</span>}
                icon={<span className={`inline-block w-2 h-2 rounded-full ${(metrics?.cpu_percent ?? 0) > 80 ? 'bg-warning' : 'bg-success'}`} />}
              />
              <StatCard
                label="Memory"
                value={<span className="font-mono">{metrics ? `${metrics.memory_used_gb.toFixed(1)} / ${metrics.memory_total_gb.toFixed(0)} GB` : '...'}</span>}
                icon={<span className={`inline-block w-2 h-2 rounded-full ${(metrics?.memory_percent ?? 0) > 80 ? 'bg-warning' : 'bg-success'}`} />}
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <StatCard
                label="Disk"
                value={<span className="font-mono">{disk ? `${disk.used_gb.toFixed(0)} / ${disk.total_gb.toFixed(0)} GB` : '...'}</span>}
                icon={<span className={`inline-block w-2 h-2 rounded-full ${(disk?.percent ?? 0) > 80 ? 'bg-warning' : 'bg-success'}`} />}
              />
              <StatCard
                label="GPU"
                value={<span className="font-mono text-xs">{detailed?.gpu ? `${detailed.gpu.backend.toUpperCase()} · ${detailed.gpu.tier}` : 'None'}</span>}
                icon={<span className={`inline-block w-2 h-2 rounded-full ${detailed?.gpu ? 'bg-success' : 'bg-muted-foreground/50'}`} />}
              />
            </div>

            {/* Platform info */}
            {info && (
              <div className="rounded-md bg-muted/50 px-3 py-2 text-xs text-muted-foreground font-mono flex flex-wrap gap-x-4 gap-y-1">
                <span>{info.platform} {info.platform_release}</span>
                <span>{info.architecture}</span>
                <span>{info.processor}</span>
                <span>{info.cpu_count} cores</span>
              </div>
            )}

            <div className="flex items-center justify-between">
              <Button variant="ghost" size="sm" className="text-xs" onClick={fetchHealth}>
                Refresh health
              </Button>
              <span className="text-[10px] text-muted-foreground font-mono">v1.0.0</span>
            </div>
          </CardContent>
        </Card>

        {/* Export / Import settings */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Backup & restore</CardTitle>
            <CardDescription>Export your settings to a file, or import from a backup</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Button size="sm" variant="outline" onClick={() => {
                const data = JSON.stringify(settings, null, 2)
                const blob = new Blob([data], { type: 'application/json' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url; a.download = 'sloughgpt-settings.json'; a.click()
                URL.revokeObjectURL(url)
                addToast('Settings exported', 'success')
              }}>Export settings</Button>
              <Button size="sm" variant="outline" onClick={() => {
                const input = document.createElement('input')
                input.type = 'file'; input.accept = '.json'
                input.onchange = async (e) => {
                  const file = (e.target as HTMLInputElement).files?.[0]
                  if (!file) return
                  try {
                    const text = await file.text()
                    const raw = JSON.parse(text)
                    const valid: Record<string, unknown> = {}
                    if (typeof raw.apiUrl === 'string') valid.apiUrl = raw.apiUrl
                    if (typeof raw.hfToken === 'string') valid.hfToken = raw.hfToken
                    if (typeof raw.defaultModel === 'string') valid.defaultModel = raw.defaultModel
                    if (typeof raw.defaultTemp === 'number') valid.defaultTemp = raw.defaultTemp
                    if (typeof raw.defaultMaxTokens === 'number') valid.defaultMaxTokens = raw.defaultMaxTokens
                    if (['dark', 'light', 'system'].includes(raw.theme)) valid.theme = raw.theme
                    if (typeof raw.streaming === 'boolean') valid.streaming = raw.streaming
                    if (typeof raw.customContext === 'string') valid.customContext = raw.customContext
                    if (typeof raw.collapsibleMessageLength === 'number') valid.collapsibleMessageLength = raw.collapsibleMessageLength
                    if (Object.keys(valid).length === 0) throw new Error('No valid settings found')
                    updateSettings(valid)
                    addToast('Settings imported', 'success')
                  } catch {
                    addToast('Invalid settings file', 'error')
                  }
                }
                input.click()
              }}>Import settings</Button>
            </div>
          </CardContent>
        </Card>

        {/* Danger zone */}
        <Card className="border-destructive/30">
          <CardHeader>
            <CardTitle className="text-base text-destructive">Danger zone</CardTitle>
            <CardDescription>Irreversible actions</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Clear chat history</p>
                <p className="text-xs text-muted-foreground">Removes all saved conversations from this browser</p>
              </div>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button type="button" variant="destructive" size="sm">Clear</Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Clear all chat history?</AlertDialogTitle>
                    <AlertDialogDescription>This cannot be undone.</AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={clearChat} className="bg-destructive text-destructive-foreground">Clear</AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
            <div className="border-t border-border/30" />
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Reset all settings</p>
                <p className="text-xs text-muted-foreground">Restore theme, model defaults, and custom instructions to defaults</p>
              </div>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button type="button" variant="destructive" size="sm">Reset</Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Reset all settings?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This will clear your theme preference, default model settings, and custom instructions. Chat history is not affected.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={resetAllSettings} className="bg-destructive text-destructive-foreground">Reset</AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
