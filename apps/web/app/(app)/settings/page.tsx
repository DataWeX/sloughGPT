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
  Badge,
  IconRefresh,
  Skeleton,
  STRUI_VERSION,
  VersionInspector,
} from '@sloughgpt/strui'
import { Button, cn } from '@sloughgpt/strui'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Input } from '@sloughgpt/strui'
import { Textarea } from '@sloughgpt/strui'
import { Slider } from '@sloughgpt/strui'
import { Switch } from '@sloughgpt/strui'
import { StatCard, KpiGrid } from '@sloughgpt/strui'
import { SettingsChatDefaultsCard } from '@/components/settings/SettingsChatDefaultsCard'
import { SettingsMemoryCard } from '@/components/settings/SettingsMemoryCard'
import { SettingsSystemHealthCard } from '@/components/settings/SettingsSystemHealthCard'
import { SettingsBackupRestoreCard } from '@/components/settings/SettingsBackupRestoreCard'
import { ToggleGroup as ToggleGroupRadix, ToggleGroupItem } from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { useSettings, useUpdateSettings, DEFAULT_SETTINGS } from '@/lib/store'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { useLocale, LOCALES } from '@/hooks/useLocale'
import { useTheme } from '@/components/ThemeProvider'
import { PALETTE_IDS, PALETTE_LABELS, type StoredPaletteId } from '@/lib/theme-storage'
import { systemController, type DetailedHealth, type SystemMetrics, type DiskUsage, type SystemInfo } from '@/lib/system-controller'
import { chatDB } from '@/lib/db'
import { CURRENT_SESSION_KEY } from '@/lib/chat-utils'
import { modelController } from '@/lib/model-controller'
import { formatUptime } from '@/lib/chat-utils'
import { ServingProfilesCard } from '@/components/ServingProfilesCard'
import { downloadJson, importFile } from '@/lib/download-utils'
import { extractErrorMessage } from '@/lib/error-utils'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'
import { settingsSchema } from '@/lib/validation-schemas'

const WEB_VERSION = '3.0.0'

function VersionBadge({ label, version }: { label: string; version?: string | null }) {
  if (!version) return null
  return (
    <Badge variant="outline" className="text-[10px] font-mono px-1.5 py-0 h-5 text-muted-foreground border-border/50">
      {label} {version}
    </Badge>
  )
}

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

  useRefreshShortcut(fetchHealth)

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
      addToast(extractErrorMessage(e, 'Could not toggle process guard'), 'error')
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
      setConnectionTest({ status: 'error', error: extractErrorMessage(e, 'Could not connection') })
    }
  }

  const clearChat = () => {
    chatDB.deleteKV(CURRENT_SESSION_KEY).catch(() => {})
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
          <CardFooter className="justify-end">
            <VersionBadge label="v" version={WEB_VERSION} />
          </CardFooter>
        </Card>

        {/* Palette */}
        <Card>
          <CardHeader>
            <div>
              <CardTitle className="text-base">Palette</CardTitle>
              <CardDescription>Full-spectrum color palette</CardDescription>
            </div>
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
          <CardFooter className="justify-end">
            <VersionBadge label="v" version={WEB_VERSION} />
          </CardFooter>
        </Card>

        {/* Language */}
        <Card>
          <CardHeader>
            <div>
              <CardTitle className="text-base">Language</CardTitle>
              <CardDescription>Interface language</CardDescription>
            </div>
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
          <CardFooter className="justify-end">
            <VersionBadge label="v" version={WEB_VERSION} />
          </CardFooter>
        </Card>

        {/* Connection */}
        <Card>
          <CardHeader>
            <div>
              <CardTitle className="text-base">Connection</CardTitle>
              <CardDescription>Service connection and authentication</CardDescription>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="settings-api-url" className="text-sm font-medium">API URL</label>
              <Input
                id="settings-api-url"
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
                className={cn('font-mono text-xs', settingsErrors.apiUrl && 'border-destructive ring-destructive/20')}
                aria-label="Service URL"
                aria-invalid={!!settingsErrors.apiUrl}
                aria-describedby={settingsErrors.apiUrl ? 'apiurl-error' : undefined}
              />
              {settingsErrors.apiUrl && (
                <p id="apiurl-error" className="text-xs text-destructive" role="alert">{settingsErrors.apiUrl}</p>
              )}
              <p className="text-[11px] text-muted-foreground">Service address. Changes take effect on next request.</p>
            </div>
            <div className="space-y-2">
              <label htmlFor="settings-hf-token" className="text-sm font-medium">HuggingFace Token</label>
              <Input
                id="settings-hf-token"
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
          <CardFooter className="justify-end">
            <VersionBadge label="v" version={detailed?.versions?.api} />
          </CardFooter>
        </Card>

        {/* Chat defaults */}
        <SettingsChatDefaultsCard
          temperature={settings.defaultTemp ?? 0.7}
          maxTokens={settings.defaultMaxTokens ?? 512}
          topP={settings.defaultTopP ?? 0.9}
          topK={settings.defaultTopK ?? 50}
          streaming={settings.streaming ?? true}
          collapsibleMessageLength={settings.collapsibleMessageLength ?? 0}
          onTemperatureChange={(v) => updateSettings({ defaultTemp: v })}
          onMaxTokensChange={(v) => updateSettings({ defaultMaxTokens: v })}
          onTopPChange={(v) => updateSettings({ defaultTopP: v })}
          onTopKChange={(v) => updateSettings({ defaultTopK: v })}
          onStreamingChange={(v) => updateSettings({ streaming: v })}
          onCollapsibleMessageLengthChange={(v) => updateSettings({ collapsibleMessageLength: v })}
          onReset={() => updateSettings({ defaultTemp: DEFAULT_SETTINGS.defaultTemp, defaultMaxTokens: DEFAULT_SETTINGS.defaultMaxTokens, defaultTopP: DEFAULT_SETTINGS.defaultTopP, defaultTopK: DEFAULT_SETTINGS.defaultTopK })}
          version={WEB_VERSION}
        />

        {/* Memory */}
        <SettingsMemoryCard
          customContext={settings.customContext ?? ''}
          onChange={(v) => updateSettings({ customContext: v })}
          version={WEB_VERSION}
        />

        {/* Chat commands reference */}
        <Card>
          <CardHeader>
            <div>
              <CardTitle className="text-base">Chat commands</CardTitle>
              <CardDescription>Type <kbd className="inline-flex items-center rounded border border-border/40 bg-muted px-1.5 py-0.5 text-[11px] font-mono">/</kbd> in chat to open the command palette</CardDescription>
            </div>
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
          <CardFooter className="justify-end">
            <VersionBadge label="v" version={WEB_VERSION} />
          </CardFooter>
        </Card>

        {/* Keyboard shortcuts */}
        <Card>
          <CardHeader>
            <div>
              <CardTitle className="text-base">Keyboard shortcuts</CardTitle>
              <CardDescription>Global shortcuts available on any page</CardDescription>
            </div>
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
                ['Ctrl+Shift+M', 'Toggle dark mode'],
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
          <CardFooter className="justify-end">
            <VersionBadge label="v" version={WEB_VERSION} />
          </CardFooter>
        </Card>

        {/* System health */}
        <SettingsSystemHealthCard
          apiOk={apiOk}
          modelLoaded={modelLoaded}
          modelType={modelType}
          detailed={detailed}
          metrics={metrics}
          disk={disk}
          info={info}
          healthError={healthError}
          onRefresh={fetchHealth}
        />

        {/* System info */}
        {info && (
          <Card>
            <CardHeader>
              <div>
                <CardTitle className="text-base">System information</CardTitle>
                <CardDescription>Detailed platform and environment details</CardDescription>
              </div>
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
            <CardFooter className="justify-end">
              <VersionBadge label="torch" version={detailed?.versions?.torch} />
            </CardFooter>
          </Card>
        )}

        {/* Version Inspector */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Version inspector</CardTitle>
            <CardDescription>Strui component ↔ backend feature version alignment</CardDescription>
          </CardHeader>
          <CardContent>
            <VersionInspector backendFeatures={detailed?.versions?.features} />
            <div className="flex items-center justify-between mt-3 pt-3 border-t border-border/40">
              <p className="text-xs text-muted-foreground">Component ↔ feature version sync</p>
              <span className="text-xs text-muted-foreground font-mono">strui {STRUI_VERSION}</span>
            </div>
          </CardContent>
        </Card>

        {/* Backup & restore */}
        <SettingsBackupRestoreCard
          settings={settings as unknown as Record<string, unknown>}
          onImport={(valid) => updateSettings(valid)}
          version={WEB_VERSION}
        />

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
                {processGuard.health && (
                  <>
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Alive</span>
                      <span className={processGuard.health.alive ? 'text-success' : 'text-destructive'}>
                        {processGuard.health.alive ? 'Yes' : 'No'}
                      </span>
                    </div>
                    {processGuard.health.memory_mb !== undefined && (
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Memory</span>
                        <span className="font-mono text-xs">{processGuard.health.memory_mb} MB</span>
                      </div>
                    )}
                    {processGuard.health.restarts !== undefined && (
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Restarts</span>
                        <span className="font-mono text-xs">{processGuard.health.restarts}</span>
                      </div>
                    )}
                  </>
                )}
                <p className="text-xs text-muted-foreground pt-1">
                  Process isolation runs the model in a subprocess. If it crashes, the app survives and restarts it automatically. Uses ~2 GB more RAM.
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
          <CardFooter className="justify-end">
            <VersionBadge label="v" version={detailed?.versions?.package} />
          </CardFooter>
        </Card>

        {/* Serving profiles */}
        <ServingProfilesCard />

        {/* Training config */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Training</CardTitle>
            <CardDescription>Default training parameters</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Preferred model</label>
                <Input
                  value={settings.trainingPreferredModel || ''}
                  onChange={(e) => updateSettings({ trainingPreferredModel: e.target.value })}
                  placeholder="e.g. slo-1.6b"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Preferred method</label>
                <Input
                  value={settings.trainingPreferredMethod || ''}
                  onChange={(e) => updateSettings({ trainingPreferredMethod: e.target.value })}
                  placeholder="e.g. finetune"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Max checkpoints</label>
                <Input
                  type="number"
                  min={1}
                  max={50}
                  value={settings.trainingMaxCheckpoints ?? 10}
                  onChange={(e) => updateSettings({ trainingMaxCheckpoints: parseInt(e.target.value) || 10 })}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Auto-train threshold</label>
                <Input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={settings.trainingAutoTrainThreshold ?? 0.8}
                  onChange={(e) => updateSettings({ trainingAutoTrainThreshold: parseFloat(e.target.value) || 0.8 })}
                />
              </div>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Auto-train</p>
                <p className="text-xs text-muted-foreground">Automatically train when data quality exceeds threshold</p>
              </div>
              <Switch
                checked={settings.trainingAutoTrain ?? false}
                onCheckedChange={(checked) => updateSettings({ trainingAutoTrain: checked })}
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Enable outcome tracking</p>
                <p className="text-xs text-muted-foreground">Record training runs for adaptive learning</p>
              </div>
              <Switch
                checked={settings.trainingEnableTracking ?? true}
                onCheckedChange={(checked) => updateSettings({ trainingEnableTracking: checked })}
              />
            </div>
          </CardContent>
        </Card>

        {/* Danger zone */}
        <Card className="border-destructive/30">
          <CardHeader>
            <div>
              <CardTitle className="text-base text-destructive">Danger zone</CardTitle>
              <CardDescription>Irreversible actions</CardDescription>
            </div>
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
          <CardFooter className="justify-end">
            <VersionBadge label="v" version={WEB_VERSION} />
          </CardFooter>
        </Card>
    </PageContainer>
  )
}
