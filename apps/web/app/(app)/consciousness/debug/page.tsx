'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { PUBLIC_API_URL } from '@/lib/config'
import { consciousnessController } from '@/lib/consciousness-controller'
import {
  Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle,
  Input, Skeleton, Switch,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'
import { IconCode, IconPlus, IconTrash, IconClock } from '@/components/icons/NavIcons'

interface BusEvent {
  id: string
  type: string
  data: unknown
  timestamp: number
}

const EVENT_TYPES = [
  'consciousness.reflect',
  'consciousness.status',
  'consciousness.qualia',
  'consciousness.belief_update',
  'consciousness.episode',
  'consciousness.health',
  'consciousness.error',
  'consciousness.config_change',
  'consciousness.training',
  'consciousness.personality',
  'consciousness.self_model',
]

const COMMON_EXPRESSIONS = [
  { labelKey: 'consciousness_debug.shortcut_status', endpoint: '/consciousness/status' },
  { labelKey: 'consciousness_debug.shortcut_qualia', endpoint: '/consciousness/qualia' },
  { labelKey: 'consciousness_debug.shortcut_beliefs', endpoint: '/consciousness/self-model' },
  { labelKey: 'consciousness_debug.shortcut_episodes', endpoint: '/consciousness/history/episodes?limit=5' },
]

const STATE_ENDPOINTS = [
  { key: 'status', labelKey: 'consciousness_debug.state_status', endpoint: '/consciousness/status' },
  { key: 'self_model', labelKey: 'consciousness_debug.state_self_model', endpoint: '/consciousness/self-model' },
  { key: 'personality', labelKey: 'consciousness_debug.state_personality', endpoint: '/consciousness/personality' },
  { key: 'config', labelKey: 'consciousness_debug.state_config', endpoint: '/consciousness/config' },
  { key: 'training', labelKey: 'consciousness_debug.state_training', endpoint: '/consciousness/train/status' },
] as const

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
}

function formatTimestamp(ts: number): string {
  return new Date(ts).toLocaleTimeString()
}

function formatJson(data: unknown): string {
  try {
    return JSON.stringify(data, null, 2)
  } catch {
    return String(data)
  }
}

export default function ConsciousnessDebugPage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()

  const [autoRefresh, setAutoRefresh] = useState(false)
  const [autoRefreshInterval, setAutoRefreshInterval] = useState(10)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const [stateData, setStateData] = useState<Record<string, unknown>>({})
  const [stateLoading, setStateLoading] = useState<Record<string, boolean>>({})
  const [expandedState, setExpandedState] = useState<Record<string, boolean>>({})

  const [events, setEvents] = useState<BusEvent[]>([])
  const [eventFilter, setEventFilter] = useState<string>('all')
  const [eventInput, setEventInput] = useState('')

  const [consoleInput, setConsoleInput] = useState('')
  const [consoleOutput, setConsoleOutput] = useState<string[]>([])
  const [consoleLoading, setConsoleLoading] = useState(false)

  const [healthData, setHealthData] = useState<unknown>(null)
  const [healthLoading, setHealthLoading] = useState(false)
  const [evalData, setEvalData] = useState<unknown>(null)
  const [evalLoading, setEvalLoading] = useState(false)
  const [connected, setConnected] = useState<boolean | null>(null)
  const [lastError, setLastError] = useState<string | null>(null)

  const fetchState = useCallback(async (key: string, endpoint: string) => {
    setStateLoading(prev => ({ ...prev, [key]: true }))
    try {
      let data: unknown
      if (endpoint === '/consciousness/status') {
        data = await consciousnessController.getStatus()
      } else if (endpoint === '/consciousness/self-model') {
        data = await consciousnessController.getSelfModel()
      } else if (endpoint === '/consciousness/train/status') {
        data = await consciousnessController.getTrainingStatus()
      } else {
        const res = await fetch(endpoint)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const json = await res.json()
        data = json.data ?? json
      }
      setStateData(prev => ({ ...prev, [key]: data }))
      setConnected(true)
    } catch (e) {
      setConnected(false)
      setLastError(extractErrorMessage(e))
      setStateData(prev => ({ ...prev, [key]: { error: extractErrorMessage(e) } }))
    } finally {
      setStateLoading(prev => ({ ...prev, [key]: false }))
    }
  }, [])

  const fetchAllStates = useCallback(() => {
    for (const ep of STATE_ENDPOINTS) {
      fetchState(ep.key, ep.endpoint)
    }
  }, [fetchState])

  const fetchHealth = useCallback(async () => {
    setHealthLoading(true)
    try {
      const data = await consciousnessController.healthCheck()
      setHealthData(data)
      setConnected(true)
    } catch (e) {
      setConnected(false)
      setLastError(extractErrorMessage(e))
      setHealthData({ error: extractErrorMessage(e) })
    } finally {
      setHealthLoading(false)
    }
  }, [])

  const fetchEval = useCallback(async () => {
    setEvalLoading(true)
    try {
      const data = await consciousnessController.evaluate()
      setEvalData(data)
      setConnected(true)
    } catch (e) {
      setConnected(false)
      setLastError(extractErrorMessage(e))
      setEvalData({ error: extractErrorMessage(e) })
    } finally {
      setEvalLoading(false)
    }
  }, [])

  const handleSubscribe = useCallback(() => {
    const filter = eventInput.trim()
    setEventFilter(filter || 'all')

    const newEvent: BusEvent = {
      id: generateId(),
      type: filter || 'consciousness.debug',
      data: { subscribed: true, filter: filter || null },
      timestamp: Date.now(),
    }
    setEvents(prev => [newEvent, ...prev].slice(0, 50))
    addToast(t('consciousness_debug.toast_subscribed'), 'success')
  }, [eventInput, addToast, t])

  const clearEvents = useCallback(() => {
    setEvents([])
    addToast(t('consciousness_debug.toast_events_cleared'), 'success')
  }, [addToast, t])

  const handleConsoleExecute = useCallback(async () => {
    if (!consoleInput.trim()) return
    setConsoleLoading(true)

    const input = consoleInput.trim()
    const output: string[] = [`> ${input}`]

    try {
      if (input.startsWith('GET ') || input.startsWith('POST ') || input.startsWith('PATCH ') || input.startsWith('DELETE ')) {
        const parts = input.split(' ')
        const method = parts[0]
        const path = parts.slice(1).join(' ')
        const res = await fetch(`${PUBLIC_API_URL}${path}`, {
          method,
          headers: { 'Content-Type': 'application/json' },
          ...(method !== 'GET' && parts.length > 2 ? { body: parts.slice(2).join(' ') } : {}),
        })
        const json = await res.json()
        output.push(JSON.stringify(json.data ?? json, null, 2))
        setConnected(true)
      } else if (input.startsWith('fetch ') || input.startsWith('http')) {
        const url = input.startsWith('fetch ') ? input.slice(6) : input
        const res = await fetch(url)
        const json = await res.json()
        output.push(JSON.stringify(json, null, 2))
        setConnected(true)
      } else {
        const res = await fetch(`${PUBLIC_API_URL}/consciousness/process`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: input }),
        })
        const json = await res.json()
        output.push(JSON.stringify(json.data ?? json, null, 2))
        setConnected(true)
      }
    } catch (e) {
      output.push(`Error: ${extractErrorMessage(e)}`)
      setConnected(false)
      setLastError(extractErrorMessage(e))
    }

    setConsoleOutput(prev => [...output, ...prev].slice(0, 100))
    setConsoleLoading(false)
  }, [consoleInput])

  const handleShortcut = useCallback(async (endpoint: string) => {
    setConsoleLoading(true)
    const output: string[] = [`> GET ${endpoint}`]
    try {
      const res = await fetch(`${PUBLIC_API_URL}${endpoint}`)
      const json = await res.json()
      output.push(JSON.stringify(json.data ?? json, null, 2))
      setConnected(true)
    } catch (e) {
      output.push(`Error: ${extractErrorMessage(e)}`)
      setConnected(false)
      setLastError(extractErrorMessage(e))
    }
    setConsoleOutput(prev => [...output, ...prev].slice(0, 100))
    setConsoleLoading(false)
  }, [])

  const handleReset = useCallback(async () => {
    try {
      const res = await fetch(`${PUBLIC_API_URL}/consciousness/reset`, { method: 'POST' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      addToast(t('consciousness_debug.toast_reset'), 'success')
      fetchAllStates()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }, [fetchAllStates, addToast, t])

  const handleSeed = useCallback(async () => {
    try {
      await consciousnessController.seedData({ count: 5 })
      addToast(t('consciousness_debug.toast_seeded'), 'success')
      fetchAllStates()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }, [fetchAllStates, addToast, t])

  const handleExportLog = useCallback(() => {
    const log = {
      events,
      consoleOutput,
      states: stateData,
      health: healthData,
      evaluation: evalData,
      lastError,
      exportedAt: new Date().toISOString(),
    }
    const blob = new Blob([JSON.stringify(log, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `consciousness-debug-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
    addToast(t('consciousness_debug.toast_exported'), 'success')
  }, [events, consoleOutput, stateData, healthData, evalData, lastError, addToast, t])

  const handleClearAll = useCallback(() => {
    setStateData({})
    setEvents([])
    setConsoleOutput([])
    setHealthData(null)
    setEvalData(null)
    setLastError(null)
    addToast(t('consciousness_debug.toast_cleared'), 'success')
  }, [addToast, t])

  useEffect(() => {
    fetchAllStates()
    fetchHealth()
    fetchEval()
  }, [fetchAllStates, fetchHealth, fetchEval])

  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(() => {
        fetchAllStates()
      }, autoRefreshInterval * 1000)
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [autoRefresh, autoRefreshInterval, fetchAllStates])

  const filteredEvents = eventFilter === 'all'
    ? events
    : events.filter(e => e.type === eventFilter)

  return (
    <PageContainer title={t('consciousness_debug.page_title')}>
      <div className="space-y-4 p-4 sm:p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <Switch checked={autoRefresh} onCheckedChange={setAutoRefresh} />
            <span className="text-xs text-muted-foreground">{t('consciousness_debug.auto_refresh')}</span>
            {autoRefresh && (
              <div className="flex items-center gap-1.5">
                <Input
                  type="number"
                  value={autoRefreshInterval}
                  onChange={(e) => setAutoRefreshInterval(Math.max(1, parseInt((e.target as HTMLInputElement).value) || 10))}
                  className="h-7 w-16 text-xs"
                  min={1}
                  max={300}
                />
                <span className="text-[10px] text-muted-foreground">s</span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${connected === true ? 'bg-green-500' : connected === false ? 'bg-red-500' : 'bg-yellow-500'}`} />
            <span className="text-xs text-muted-foreground">
              {connected === true ? t('consciousness_debug.connected') : connected === false ? t('consciousness_debug.disconnected') : t('consciousness_debug.checking')}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card className="lg:col-span-2">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-sm">{t('consciousness_debug.state_inspector')}</CardTitle>
                  <CardDescription>{t('consciousness_debug.state_inspector_desc')}</CardDescription>
                </div>
                <Button variant="outline" size="sm" onClick={fetchAllStates} className="h-7 text-xs">
                  {t('consciousness_debug.refresh_all')}
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {STATE_ENDPOINTS.map(ep => (
                <div key={ep.key} className="rounded-md border border-border/50 bg-muted/10">
                  <button
                    type="button"
                    onClick={() => setExpandedState(prev => ({ ...prev, [ep.key]: !prev[ep.key] }))}
                    className="w-full flex items-center justify-between px-3 py-2 text-left"
                  >
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="font-mono text-[9px] border px-1.5 py-0 shrink-0 bg-emerald-500/15 text-emerald-400 border-emerald-500/30">
                        GET
                      </Badge>
                      <span className="text-xs font-medium">{t(ep.labelKey)}</span>
                      <span className="text-[10px] text-muted-foreground font-mono">{ep.endpoint}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {stateLoading[ep.key] && <Skeleton className="h-3 w-3" />}
                      <span className="text-muted-foreground text-[10px]">{expandedState[ep.key] ? '▼' : '▶'}</span>
                    </div>
                  </button>
                  {expandedState[ep.key] && (
                    <div className="px-3 pb-3">
                      <pre className="max-h-60 overflow-auto rounded border border-border/30 bg-background/50 p-3 text-[11px] font-mono whitespace-pre-wrap">
                        {stateData[ep.key] ? formatJson(stateData[ep.key]) : '{ }'}
                      </pre>
                    </div>
                  )}
                </div>
              ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-sm">{t('consciousness_debug.event_log')}</CardTitle>
                  <CardDescription>{t('consciousness_debug.event_log_desc')}</CardDescription>
                </div>
                <Button variant="outline" size="sm" onClick={clearEvents} className="h-7 text-xs">
                  <IconTrash className="h-3 w-3 mr-1" />
                  {t('consciousness_debug.clear')}
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex gap-2">
                <Input
                  value={eventInput}
                  onChange={(e) => setEventInput((e.target as HTMLInputElement).value)}
                  placeholder={t('consciousness_debug.event_filter_placeholder')}
                  className="h-7 text-xs"
                />
                <Button size="sm" onClick={handleSubscribe} className="h-7 text-xs shrink-0">
                  <IconPlus className="h-3 w-3 mr-1" />
                  {t('consciousness_debug.subscribe')}
                </Button>
              </div>
              <select
                value={eventFilter}
                onChange={(e) => setEventFilter((e.target as HTMLSelectElement).value)}
                className="w-full h-7 rounded-md border border-border/50 bg-muted/20 px-2 text-xs"
              >
                <option value="all">{t('consciousness_debug.all_events')}</option>
                {EVENT_TYPES.map(type => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </select>
              <div className="max-h-80 overflow-y-auto space-y-1">
                {filteredEvents.length === 0 && (
                  <p className="text-[10px] text-muted-foreground text-center py-4">{t('consciousness_debug.no_events')}</p>
                )}
                {filteredEvents.map(ev => (
                  <div
                    key={ev.id}
                    className="rounded border border-border/30 bg-muted/10 px-2 py-1.5 text-[10px]"
                  >
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="text-muted-foreground font-mono">{formatTimestamp(ev.timestamp)}</span>
                      <Badge variant="outline" className="text-[8px] font-mono border px-1 py-0">
                        {ev.type}
                      </Badge>
                    </div>
                    <pre className="whitespace-pre-wrap break-all text-[9px] font-mono text-muted-foreground">
                      {formatJson(ev.data)}
                    </pre>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">{t('consciousness_debug.console')}</CardTitle>
              <CardDescription>{t('consciousness_debug.console_desc')}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex flex-wrap gap-1.5">
                {COMMON_EXPRESSIONS.map(expr => (
                  <Button
                    key={expr.endpoint}
                    variant="outline"
                    size="sm"
                    className="h-6 text-[10px] px-2"
                    onClick={() => handleShortcut(expr.endpoint)}
                    disabled={consoleLoading}
                  >
                    {t(expr.labelKey)}
                  </Button>
                ))}
              </div>
              <div className="flex gap-2">
                <Input
                  value={consoleInput}
                  onChange={(e) => setConsoleInput((e.target as HTMLInputElement).value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleConsoleExecute() }}
                  placeholder={t('consciousness_debug.console_placeholder')}
                  className="h-7 text-xs font-mono"
                  disabled={consoleLoading}
                />
                <Button size="sm" onClick={handleConsoleExecute} disabled={consoleLoading} className="h-7 text-xs shrink-0">
                  {consoleLoading ? '...' : t('consciousness_debug.execute')}
                </Button>
              </div>
              <div className="max-h-60 overflow-y-auto rounded border border-border/30 bg-background/50 p-2 text-[11px] font-mono">
                {consoleOutput.length === 0 ? (
                  <p className="text-muted-foreground text-center py-2">{t('consciousness_debug.console_empty')}</p>
                ) : (
                  consoleOutput.map((line, i) => (
                    <div key={i} className={`whitespace-pre-wrap break-all py-0.5 ${line.startsWith('>') ? 'text-primary' : 'text-muted-foreground'}`}>
                      {line}
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">{t('consciousness_debug.diagnostics')}</CardTitle>
              <CardDescription>{t('consciousness_debug.diagnostics_desc')}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <Button variant="outline" onClick={fetchHealth} disabled={healthLoading} className="h-8 text-xs">
                  {healthLoading ? '...' : t('consciousness_debug.check_health')}
                </Button>
                <Button variant="outline" onClick={fetchEval} disabled={evalLoading} className="h-8 text-xs">
                  {evalLoading ? '...' : t('consciousness_debug.run_eval')}
                </Button>
              </div>
              {healthData != null && (
                <div>
                  <p className="text-[10px] font-medium text-muted-foreground mb-1">{t('consciousness_debug.health_report')}</p>
                  <pre className="max-h-40 overflow-auto rounded border border-border/30 bg-background/50 p-2 text-[10px] font-mono whitespace-pre-wrap">
                    {formatJson(healthData as Record<string, unknown>)}
                  </pre>
                </div>
              )}
              {evalData != null && (
                <div>
                  <p className="text-[10px] font-medium text-muted-foreground mb-1">{t('consciousness_debug.eval_report')}</p>
                  <pre className="max-h-40 overflow-auto rounded border border-border/30 bg-background/50 p-2 text-[10px] font-mono whitespace-pre-wrap">
                    {formatJson(evalData as Record<string, unknown>)}
                  </pre>
                </div>
              )}
              {lastError && (
                <div>
                  <p className="text-[10px] font-medium text-red-400 mb-1">{t('consciousness_debug.last_error')}</p>
                  <pre className="max-h-24 overflow-auto rounded border border-red-500/20 bg-red-500/5 p-2 text-[10px] font-mono text-red-400 whitespace-pre-wrap">
                    {lastError}
                  </pre>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">{t('consciousness_debug.actions')}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              <Button variant="outline" onClick={handleReset} className="h-8 text-xs">
                {t('consciousness_debug.reset_defaults')}
              </Button>
              <Button variant="outline" onClick={handleSeed} className="h-8 text-xs">
                {t('consciousness_debug.seed_data')}
              </Button>
              <Button variant="outline" onClick={handleExportLog} className="h-8 text-xs">
                {t('consciousness_debug.export_log')}
              </Button>
              <Button variant="destructive" onClick={handleClearAll} className="h-8 text-xs">
                {t('consciousness_debug.clear_all')}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
