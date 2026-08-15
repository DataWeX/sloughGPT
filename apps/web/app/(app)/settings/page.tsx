'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { PUBLIC_API_URL } from '@/lib/config'
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
  IconRefresh,
  Skeleton,
} from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { Textarea } from '@sloughgpt/strui'
import { Slider } from '@sloughgpt/strui'
import { Switch } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import { ToggleGroup as ToggleGroupRadix, ToggleGroupItem } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { useSettings, useUpdateSettings, DEFAULT_SETTINGS } from '@/lib/store'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { useLocale, LOCALES } from '@/hooks/useLocale'
import { useTheme } from '@/components/ThemeProvider'
import { PALETTE_IDS, PALETTE_LABELS, type StoredPaletteId } from '@/lib/theme-storage'
import { systemController, type DetailedHealth, type SystemMetrics, type DiskUsage, type SystemInfo } from '@/lib/system-controller'
import { modelController } from '@/lib/model-controller'
import { formatUptime } from '@/lib/chat-utils'
import { downloadJson, importFile } from '@/lib/download-utils'
import { extractErrorMessage } from '@/lib/error-utils'
import { settingsSchema } from '@/lib/validation-schemas'

function SettingsSlider({
  label, value, onChange, min, max, step, formatValue,
}: {
  label: string
  value: number | undefined
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  formatValue?: (v: number) => string
}) {
  const safeValue = value ?? 0
  const display = formatValue ? formatValue(safeValue) : `${safeValue}`
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">{label}</label>
        <span className="text-sm text-muted-foreground">{display}</span>
      </div>
      <Slider value={[safeValue]} onValueChange={([v]: number[]) => onChange(v)} min={min} max={max} step={step} />
    </div>
  )
}

export default function SettingsPage() {
  const settings = useSettings()
  const updateSettings = useUpdateSettings()
  const addToast = useToastStore(s => s.addToast)
  const { healthLegacy: apiHealth } = useLiveStatus()
  const { locale, setLocale } = useLocale()
  const { palette, setPalette } = useTheme()
  const [detailed, setDetailed] = useState<DetailedHealth | null>(null)
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
  const [disk, setDisk] = useState<DiskUsage | null>(null)
  const [info, setInfo] = useState<SystemInfo | null>(null)
  const [connectionTest, setConnectionTest] = useState<{ status: 'idle' | 'testing' | 'ok' | 'error'; latency?: number; error?: string }>({ status: 'idle' })
  const [settingsErrors, setSettingsErrors] = useState<{ apiUrl?: string; hfToken?: string }>({})
  const [processGuard, setProcessGuard] = useState<import('@/lib/system-controller').ProcessGuardStatus | null>(null)
  const [healthError, setHealthError] = useState(false)

  const fetchHealth = useCallback(async () => {
    setHealthError(false)
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
    if (d.status === 'rejected' && m.status === 'rejected' && dk.status === 'rejected' && inf.status === 'rejected') {
      setHealthError(true)
    }
  }, [])

  useEffect(() => { fetchHealth() }, [fetchHealth])

  const fetchProcessGuard = useCallback(async () => {
    try {
      const status = await systemController.getProcessGuardStatus()
      setProcessGuard(status)
    } catch { /* unavailable */ }
  }, [])

  useEffect(() => { fetchProcessGuard() }, [fetchProcessGuard])

  const handleToggleProcessGuard = async (enabled: boolean) => {
    try {
      const status = await systemController.setProcessGuardEnabled(enabled)
      setProcessGuard(status)
      addToast(enabled ? 'Process isolation enabled' : 'Process isolation disabled', 'success')
    } catch (e: unknown) {
      addToast(extractErrorMessage(e, 'Failed to toggle process guard'), 'error')
    }
  }

  const isOnline = apiHealth !== null && apiHealth !== 'offline'
  const apiOk = isOnline && (apiHealth.status === 'healthy' || detailed?.status === 'healthy')
  const modelLoaded = isOnline && (apiHealth.model_loaded || detailed?.model_loaded)
  const modelType = isOnline ? (apiHealth.model_type || detailed?.model_type) : null

  const handleTestConnection = async () => {
    setConnectionTest({ status: 'testing' })
    const start = Date.now()
    try {
      const h = await modelController.getHealth()
      const latency = Date.now() - start
      if (h && (h.status === 'healthy' || h.model_loaded !== undefined)) {
        setConnectionTest({ status: 'ok', latency })
      } else {
        setConnectionTest({ status: 'error', error: 'Unexpected response' })
      }
    } catch (e: unknown) {
      setConnectionTest({ status: 'error', error: extractErrorMessage(e, 'Connection failed') })
    }
  }

  const clearChat = () => {
    localStorage.removeItem('man_current_conversation')
    addToast('Chat history cleared', 'success')
  }

  const resetAllSettings = () => {
    updateSettings(DEFAULT_SETTINGS)
    addToast('Settings reset to defaults', 'success')
  }

  return (
    <PageContainer title="Settings">
      {/* Appearance */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base">Appearance</CardTitle>
                <CardDescription>Theme preference</CardDescription>
              </div>
              {settings.theme !== 'light' && (
                <Button size="sm" variant="ghost" className="h-8 text-xs text-muted-foreground" onClick={() => updateSettings({ theme: 'light' })}>
                  Reset
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <ToggleGroupRadix type="single" value={settings.theme} onValueChange={(v) => v && updateSettings({ theme: v as 'light' | 'dark' | 'system' })}>
              <ToggleGroupItem value="light">Light</ToggleGroupItem>
              <ToggleGroupItem value="dark">Dark</ToggleGroupItem>
              <ToggleGroupItem value="system">System</ToggleGroupItem>
            </ToggleGroupRadix>
            <div className="mt-3 p-3 rounded-lg border border-border/50 bg-card">
              <div className="flex items-center gap-2 mb-2">
                <div className="h-3 w-3 rounded-full bg-primary" />
                <div className="h-2 w-16 rounded bg-muted" />
              </div>
              <div className="space-y-1.5">
                <div className="h-2 w-full rounded bg-muted/50" />
                <div className="h-2 w-3/4 rounded bg-muted/50" />
              </div>
              <div className="flex gap-1.5 mt-2">
                <div className="h-5 px-2 rounded bg-primary text-[8px] text-primary-foreground flex items-center justify-center">Button</div>
                <div className="h-5 px-2 rounded bg-muted text-[8px] text-muted-foreground flex items-center justify-center">Secondary</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Palette */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Palette</CardTitle>
            <CardDescription>Full-spectrum color palette</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {PALETTE_IDS.map((id) => (
                <Button
                  key={id}
                  size="sm"
                  variant={palette === id ? 'default' : 'outline'}
                  className="h-9 text-xs"
                  onClick={() => setPalette(id)}
                >
                  {PALETTE_LABELS[id]}
                </Button>
              ))}
            </div>
            <div className="mt-3 p-3 rounded-lg border border-border/50 bg-card">
              <div className="flex items-center gap-2 mb-2">
                <div className="h-3 w-3 rounded-full bg-primary" />
                <div className="h-2 w-16 rounded bg-muted" />
              </div>
              <div className="space-y-1.5">
                <div className="h-2 w-full rounded bg-muted/50" />
                <div className="h-2 w-3/4 rounded bg-muted/50" />
              </div>
              <div className="flex gap-1.5 mt-2">
                <div className="h-5 px-2 rounded bg-primary text-[8px] text-primary-foreground flex items-center justify-center">Primary</div>
                <div className="h-5 px-2 rounded bg-accent text-[8px] text-accent-foreground flex items-center justify-center">Accent</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Language */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Language</CardTitle>
            <CardDescription>Interface language</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {LOCALES.map(l => (
                <Button
                  key={l.code}
                  size="sm"
                  variant={locale === l.code ? 'default' : 'outline'}
                  className="h-9 text-xs gap-1.5"
                  onClick={() => setLocale(l.code)}
                >
                  <span>{l.flag}</span>
                  <span>{l.name}</span>
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Connection */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Connection</CardTitle>
            <CardDescription>API server and authentication</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">API URL</label>
              <Input
                value={settings.apiUrl}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                  updateSettings({ apiUrl: e.target.value })
                  if (settingsErrors.apiUrl) setSettingsErrors(prev => ({ ...prev, apiUrl: undefined }))
                }}
                onBlur={() => {
                  if (settings.apiUrl) {
                    const result = settingsSchema.shape.apiUrl.safeParse(settings.apiUrl)
                    if (!result.success) {
                      setSettingsErrors(prev => ({ ...prev, apiUrl: result.error.issues[0]?.message }))
                    }
                  }
                }}
                placeholder={PUBLIC_API_URL}
                className={`font-mono text-xs ${settingsErrors.apiUrl ? 'border-destructive ring-destructive/20' : ''}`}
                aria-label="API server URL"
                aria-invalid={!!settingsErrors.apiUrl}
                aria-describedby={settingsErrors.apiUrl ? 'apiurl-error' : undefined}
              />
              {settingsErrors.apiUrl && (
                <p id="apiurl-error" className="text-[10px] text-destructive" role="alert">{settingsErrors.apiUrl}</p>
              )}
              <p className="text-[11px] text-muted-foreground">Backend server address. Changes take effect on next request.</p>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">HuggingFace Token</label>
              <Input
                type="password"
                value={settings.hfToken}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateSettings({ hfToken: e.target.value })}
                placeholder="hf_..."
                className="font-mono text-xs"
                aria-label="HuggingFace API token"
              />
              <p className="text-xs text-muted-foreground">Required for loading private HuggingFace models. Stored locally in your browser.</p>
            </div>
            <div className="flex items-center gap-2 pt-2">
              <Button size="sm" variant="outline" className="h-9 text-xs" onClick={handleTestConnection} disabled={connectionTest.status === 'testing'}>
                {connectionTest.status === 'testing' ? 'Testing...' : 'Test connection'}
              </Button>
              {connectionTest.status === 'ok' && (
                <span className="text-xs text-success font-medium">Connected ({connectionTest.latency}ms)</span>
              )}
              {connectionTest.status === 'error' && (
                <span className="text-xs text-destructive font-medium">{connectionTest.error}</span>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Chat defaults */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base">Chat defaults</CardTitle>
                <CardDescription>Default model and generation settings</CardDescription>
              </div>
              {(settings.defaultTemp !== DEFAULT_SETTINGS.defaultTemp || settings.defaultMaxTokens !== DEFAULT_SETTINGS.defaultMaxTokens || settings.defaultTopP !== DEFAULT_SETTINGS.defaultTopP || settings.defaultTopK !== DEFAULT_SETTINGS.defaultTopK) && (
                <Button size="sm" variant="ghost" className="h-8 text-xs text-muted-foreground" onClick={() => updateSettings({ defaultTemp: DEFAULT_SETTINGS.defaultTemp, defaultMaxTokens: DEFAULT_SETTINGS.defaultMaxTokens, defaultTopP: DEFAULT_SETTINGS.defaultTopP, defaultTopK: DEFAULT_SETTINGS.defaultTopK })}>
                  Reset
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
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
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <SettingsSlider
                label="Top-P"
                value={settings.defaultTopP}
                onChange={(v) => updateSettings({ defaultTopP: v })}
                min={0}
                max={1}
                step={0.05}
              />
              <SettingsSlider
                label="Top-K"
                value={settings.defaultTopK}
                onChange={(v) => updateSettings({ defaultTopK: v })}
                min={0}
                max={100}
                step={5}
              />
            </div>
            <div className="flex items-center justify-between pt-2">
              <div>
                <p className="text-sm font-medium">Streaming</p>
                <p className="text-xs text-muted-foreground">Show tokens as they are generated</p>
              </div>
              <Switch
                checked={settings.streaming}
                onCheckedChange={(checked) => updateSettings({ streaming: checked })}
                aria-label="Toggle streaming"
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

        {/* Keyboard shortcuts */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Keyboard shortcuts</CardTitle>
            <CardDescription>Global shortcuts available on any page</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
              {[
                ['Ctrl+1–9', 'Navigate pages (Chat, Models, etc.)'],
                ['Ctrl+N', 'New conversation'],
                ['Ctrl+\\', 'Toggle sidebar'],
                ['Ctrl+K', 'Command palette'],
                ['Ctrl+Shift+F', 'Search conversations'],
                ['Ctrl+Shift+A', 'Open settings'],
                ['?', 'Show keyboard shortcuts'],
                ['Esc', 'Close dialog / Cancel'],
              ].map(([key, desc]) => (
                <div key={key} className="flex items-center gap-2">
                  <kbd className="text-xs text-primary font-mono shrink-0 min-w-[80px]">{key}</kbd>
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
            {healthError && !detailed && !metrics ? (
              <div className="text-center py-6">
                <p className="text-sm text-destructive mb-3">Could not connect to server</p>
                <Button size="sm" variant="outline" onClick={fetchHealth}>
                  <IconRefresh className="h-3.5 w-3.5 mr-1.5" />
                  Retry
                </Button>
              </div>
            ) : (
              <>
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
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <StatCard
                label="CPU"
                value={<span className="font-mono">{metrics ? `${metrics.cpu_percent}%` : '...'}</span>}
                icon={<span className={`inline-block w-2 h-2 rounded-full ${(metrics?.cpu_percent ?? 0) > 80 ? 'bg-warning' : 'bg-success'}`} />}
              />
              <StatCard
                label="Memory"
                value={<span className="font-mono">{metrics ? `${(metrics.memory_used_gb ?? 0).toFixed(1)} / ${(metrics.memory_total_gb ?? 0).toFixed(0)} GB` : '...'}</span>}
                icon={<span className={`inline-block w-2 h-2 rounded-full ${(metrics?.memory_percent ?? 0) > 80 ? 'bg-warning' : 'bg-success'}`} />}
              />
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <StatCard
                label="Disk"
                value={<span className="font-mono">{disk ? `${(disk.used_gb ?? 0).toFixed(0)} / ${(disk.total_gb ?? 0).toFixed(0)} GB` : '...'}</span>}
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
              <span className="text-xs text-muted-foreground font-mono">v1.0.0</span>
            </div>
            </>
            )}
          </CardContent>
        </Card>

        {/* System info */}
        {info && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">System information</CardTitle>
              <CardDescription>Detailed platform and environment details</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">Platform</p>
                  <p className="text-sm font-medium mt-0.5">{info.platform} {info.platform_release}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">Architecture</p>
                  <p className="text-sm font-medium mt-0.5">{info.architecture}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">Processor</p>
                  <p className="text-sm font-medium mt-0.5 truncate">{info.processor}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">CPU cores</p>
                  <p className="text-sm font-medium mt-0.5">{info.cpu_count}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">Platform version</p>
                  <p className="text-sm font-medium mt-0.5 font-mono">{info.platform_version}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Export / Import settings */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Backup & restore</CardTitle>
            <CardDescription>Export your settings to a file, or import from a backup</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Button size="sm" variant="outline" onClick={() => {
                downloadJson(settings, 'sloughgpt-settings.json')
                addToast('Settings exported', 'success')
              }}>Export settings</Button>
              <Button size="sm" variant="outline" onClick={async () => {
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
                  updateSettings(valid)
                  addToast('Settings imported', 'success')
                } catch {
                  addToast('Invalid settings file', 'error')
                }
              }}>Import settings</Button>
            </div>
          </CardContent>
        </Card>

        {/* Process isolation */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base">Process isolation</CardTitle>
                <CardDescription>Run model inference in a separate process for crash safety</CardDescription>
              </div>
              {processGuard && (
                <Switch
                  checked={processGuard.enabled}
                  onCheckedChange={handleToggleProcessGuard}
                  aria-label="Toggle process isolation"
                />
              )}
            </div>
          </CardHeader>
          <CardContent>
            {processGuard ? (
              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Status</span>
                  <span className={processGuard.active ? 'text-success' : 'text-muted-foreground'}>
                    {processGuard.active ? 'Active' : processGuard.enabled ? 'Enabled (not running)' : 'Disabled'}
                  </span>
                </div>
                {processGuard.model_id && (
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Model</span>
                    <span className="font-mono text-xs">{processGuard.model_id}</span>
                  </div>
                )}
                <p className="text-xs text-muted-foreground pt-1">
                  Process isolation runs the model in a subprocess. If it crashes, the server survives and restarts it automatically. Uses ~2 GB more RAM.
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                <Skeleton className="h-4 w-40" />
                <Skeleton className="h-4 w-64" />
                <Skeleton className="h-3 w-full" />
              </div>
            )}
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
    </PageContainer>
  )
}
