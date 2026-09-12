'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { consciousnessController } from '@/lib/consciousness-controller'
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Input,
  Skeleton,
  Switch,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'

interface HealthData {
  status: string
  health_score: number
  enabled: boolean
  level: number
  episodes: number
  avg_growth: number
  positive_ratio: number
  qualia: Record<string, number>
  last_reflection: string
}

interface Episode {
  timestamp: number
  growth_delta: number
  rating: number
  qualia: Record<string, number>
  [key: string]: unknown
}

interface EpisodesData {
  episodes: Episode[]
  total: number
}

type AlertSeverity = 'critical' | 'warning' | 'info'

interface Alert {
  id: string
  severity: AlertSeverity
  message: string
  timestamp: number
  dismissed: boolean
}

interface AlertRule {
  id: string
  label: string
  enabled: boolean
  threshold: number
}

interface AlertSettings {
  notifications: boolean
  sound: boolean
  rules: AlertRule[]
}

const ALERTS_HISTORY_KEY = 'consciousness_alerts_history'
const ALERTS_SETTINGS_KEY = 'consciousness_alerts_settings'
const MAX_HISTORY = 50

const DEFAULT_RULES: AlertRule[] = [
  { id: 'health_score', label: 'Health score below threshold', enabled: true, threshold: 50 },
  { id: 'avg_growth', label: 'Avg growth negative', enabled: true, threshold: -0.05 },
  { id: 'positive_ratio', label: 'Positive ratio low', enabled: true, threshold: 0.3 },
  { id: 'no_episodes', label: 'No episodes for N hours', enabled: true, threshold: 24 },
  { id: 'beliefs_unstable', label: 'Beliefs unstable', enabled: true, threshold: 0.2 },
]

const DEFAULT_SETTINGS: AlertSettings = {
  notifications: false,
  sound: false,
  rules: DEFAULT_RULES,
}

function loadHistory(): Alert[] {
  try {
    const raw = localStorage.getItem(ALERTS_HISTORY_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveHistory(history: Alert[]) {
  try {
    localStorage.setItem(ALERTS_HISTORY_KEY, JSON.stringify(history.slice(-MAX_HISTORY)))
  } catch (e) {
    console.error('Failed to save alert history', e)
  }
}

function loadSettings(): AlertSettings {
  try {
    const raw = localStorage.getItem(ALERTS_SETTINGS_KEY)
    if (!raw) return DEFAULT_SETTINGS
    const parsed = JSON.parse(raw)
    return { ...DEFAULT_SETTINGS, ...parsed, rules: parsed.rules ?? DEFAULT_RULES }
  } catch {
    return DEFAULT_SETTINGS
  }
}

function saveSettings(settings: AlertSettings) {
  try {
    localStorage.setItem(ALERTS_SETTINGS_KEY, JSON.stringify(settings))
  } catch (e) {
    console.error('Failed to save alert settings', e)
  }
}

function formatTimeAgo(ts: number): string {
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

function severityColor(sev: AlertSeverity): string {
  if (sev === 'critical') return 'bg-red-500/10 text-red-500'
  if (sev === 'warning') return 'bg-amber-500/10 text-amber-500'
  return 'bg-blue-500/10 text-blue-500'
}

function severityBadgeColor(sev: AlertSeverity): string {
  if (sev === 'critical') return 'bg-red-500'
  if (sev === 'warning') return 'bg-amber-500'
  return 'bg-blue-500'
}

function playBeep() {
  try {
    const ctx = new AudioContext()
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.connect(gain)
    gain.connect(ctx.destination)
    osc.frequency.value = 800
    gain.gain.value = 0.3
    osc.start()
    osc.stop(ctx.currentTime + 0.2)
  } catch (e) {
    console.error('Failed to play alert beep', e)
  }
}

function sendNotification(title: string, body: string) {
  if (typeof window !== 'undefined' && 'Notification' in window && Notification.permission === 'granted') {
    new Notification(title, { body })
  }
}

export default function ConsciousnessAlertsPage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()
  const [loading, setLoading] = useState(true)
  const [health, setHealth] = useState<HealthData | null>(null)
  const [episodesData, setEpisodesData] = useState<EpisodesData | null>(null)
  const [activeAlerts, setActiveAlerts] = useState<Alert[]>([])
  const [alertHistory, setAlertHistory] = useState<Alert[]>([])
  const [settings, setSettings] = useState<AlertSettings>(DEFAULT_SETTINGS)
  const [historyFilter, setHistoryFilter] = useState<AlertSeverity | 'all'>('all')
  const [lastCheck, setLastCheck] = useState<number>(Date.now() / 1000)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchHealth = useCallback(async () => {
    try {
      const [healthResult, episodesResult] = await Promise.allSettled([
        consciousnessController.healthCheck(),
        consciousnessController.getEpisodeHistory(50),
      ])

      if (healthResult.status === 'fulfilled') {
        setHealth(healthResult.value as unknown as HealthData)
      }
      if (episodesResult.status === 'fulfilled') {
        setEpisodesData(episodesResult.value as unknown as EpisodesData)
      }
      setLastCheck(Date.now() / 1000)
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    setAlertHistory(loadHistory())
    setSettings(loadSettings())
    fetchHealth()
  }, [fetchHealth])

  useEffect(() => {
    if (settings.notifications && typeof window !== 'undefined' && 'Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission()
    }
  }, [settings.notifications])

  const generateAlerts = useCallback(() => {
    const newAlerts: Alert[] = []
    const now = Date.now() / 1000

    const findRule = (id: string) => settings.rules.find(r => r.id === id)

    if (health) {
      const healthRule = findRule('health_score')
      if (healthRule?.enabled && health.health_score < healthRule.threshold) {
        newAlerts.push({
          id: `health_score_${now}`,
          severity: health.health_score < healthRule.threshold / 2 ? 'critical' : 'warning',
          message: `Health score ${health.health_score.toFixed(0)} is below threshold ${healthRule.threshold}`,
          timestamp: now,
          dismissed: false,
        })
      }

      const growthRule = findRule('avg_growth')
      if (growthRule?.enabled && health.avg_growth < growthRule.threshold) {
        newAlerts.push({
          id: `avg_growth_${now}`,
          severity: 'critical',
          message: `Avg growth ${(health.avg_growth * 100).toFixed(1)}% is below threshold ${(growthRule.threshold * 100).toFixed(1)}%`,
          timestamp: now,
          dismissed: false,
        })
      }

      const ratioRule = findRule('positive_ratio')
      if (ratioRule?.enabled && health.positive_ratio < ratioRule.threshold) {
        newAlerts.push({
          id: `positive_ratio_${now}`,
          severity: 'warning',
          message: `Positive ratio ${(health.positive_ratio * 100).toFixed(0)}% is below threshold ${(ratioRule.threshold * 100).toFixed(0)}%`,
          timestamp: now,
          dismissed: false,
        })
      }
    }

    if (episodesData?.episodes && episodesData.episodes.length > 0) {
      const epRule = findRule('no_episodes')
      if (epRule?.enabled) {
        const latestTs = Math.max(...episodesData.episodes.map(e => typeof e.timestamp === 'number' ? e.timestamp : 0))
        const hoursSinceLast = (now - latestTs) / 3600
        if (hoursSinceLast > epRule.threshold) {
          newAlerts.push({
            id: `no_episodes_${now}`,
            severity: 'warning',
            message: `No episodes for ${Math.floor(hoursSinceLast)}h (threshold: ${epRule.threshold}h)`,
            timestamp: now,
            dismissed: false,
          })
        }
      }
    }

    const beliefsRule = findRule('beliefs_unstable')
    if (beliefsRule?.enabled && health?.qualia) {
      const beliefs = health.qualia
      const unstable = Object.entries(beliefs).filter(([, v]) => v < beliefsRule.threshold || v > (1 - beliefsRule.threshold + 0.05))
      if (unstable.length > 0) {
        newAlerts.push({
          id: `beliefs_unstable_${now}`,
          severity: 'warning',
          message: `Beliefs unstable: ${unstable.map(([k]) => k).join(', ')}`,
          timestamp: now,
          dismissed: false,
        })
      }
    }

    if (newAlerts.length > 0) {
      setActiveAlerts(prev => {
        const existing = new Set(prev.filter(a => !a.dismissed).map(a => a.message))
        const fresh = newAlerts.filter(a => !existing.has(a.message))
        if (fresh.length > 0) {
          setAlertHistory(prevH => {
            const next = [...prevH, ...fresh].slice(-MAX_HISTORY)
            saveHistory(next)
            return next
          })
          if (settings.sound) playBeep()
          if (settings.notifications) {
            fresh.forEach(a => sendNotification(`Consciousness Alert [${a.severity}]`, a.message))
          }
          return [...prev, ...fresh]
        }
        return prev
      })
    }
  }, [health, episodesData, settings, addToast])

  useEffect(() => {
    generateAlerts()
  }, [health, episodesData, generateAlerts])

  useEffect(() => {
    intervalRef.current = setInterval(fetchHealth, 60000)
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [fetchHealth])

  const dismissAlert = (id: string) => {
    setActiveAlerts(prev => prev.map(a => a.id === id ? { ...a, dismissed: true } : a))
  }

  const clearAllAlerts = () => {
    setActiveAlerts(prev => prev.map(a => ({ ...a, dismissed: true })))
  }

  const updateSettings = (partial: Partial<AlertSettings>) => {
    setSettings(prev => {
      const next = { ...prev, ...partial }
      saveSettings(next)
      return next
    })
  }

  const updateRule = (ruleId: string, partial: Partial<AlertRule>) => {
    setSettings(prev => {
      const next = {
        ...prev,
        rules: prev.rules.map(r => r.id === ruleId ? { ...r, ...partial } : r),
      }
      saveSettings(next)
      return next
    })
  }

  const clearHistory = () => {
    setAlertHistory([])
    saveHistory([])
  }

  const unfilteredActive = activeAlerts.filter(a => !a.dismissed)
  const activeCount = unfilteredActive.length
  const filteredHistory = historyFilter === 'all'
    ? alertHistory
    : alertHistory.filter(a => a.severity === historyFilter)

  if (loading) {
    return (
      <PageContainer title={t('consciousness_alerts.page_title')}>
        <div className="space-y-6 p-6">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
            <Skeleton className="h-20" />
            <Skeleton className="h-20" />
            <Skeleton className="h-20" />
          </div>
          <Skeleton className="h-48" />
          <Skeleton className="h-64" />
          <Skeleton className="h-64" />
        </div>
      </PageContainer>
    )
  }

  return (
    <PageContainer title={t('consciousness_alerts.page_title')}>
      <div className="space-y-4 sm:space-y-6 p-4 sm:p-6">
        <div className="grid grid-cols-2 gap-2 sm:gap-3 md:grid-cols-3">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground">{t('consciousness_alerts.health_score')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-xl font-bold ${health && health.health_score > 70 ? 'text-green-500' : health && health.health_score > 40 ? 'text-amber-500' : 'text-red-500'}`}>
                {health?.health_score?.toFixed(0) ?? '—'}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground">{t('consciousness_alerts.active_alerts')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-xl font-bold ${activeCount > 0 ? 'text-red-500' : 'text-green-500'}`}>
                {activeCount}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs text-muted-foreground">{t('consciousness_alerts.last_check')}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xs text-muted-foreground font-mono">
                {formatTimeAgo(lastCheck)}
              </div>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">{t('consciousness_alerts.active_alerts_title')}</CardTitle>
              {unfilteredActive.length > 0 && (
                <Button variant="ghost" size="sm" onClick={clearAllAlerts}>
                  {t('consciousness_alerts.clear_all')}
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {unfilteredActive.length === 0 ? (
              <p className="text-sm text-muted-foreground py-4 text-center">{t('consciousness_alerts.no_active')}</p>
            ) : (
              <div className="space-y-2">
                {unfilteredActive.map(alert => (
                  <div key={alert.id} className={`flex items-start gap-3 rounded-md p-3 ${severityColor(alert.severity)}`}>
                    <span className={`inline-block w-2 h-2 rounded-full mt-1 shrink-0 ${severityBadgeColor(alert.severity)}`} />
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium">{alert.message}</p>
                      <p className="text-xs opacity-70 mt-0.5">{formatTimeAgo(alert.timestamp)}</p>
                    </div>
                    <Badge variant="outline" className="shrink-0 text-[10px] capitalize">{alert.severity}</Badge>
                    <button
                      onClick={() => dismissAlert(alert.id)}
                      className="shrink-0 p-1 rounded hover:bg-foreground/5 transition-colors"
                      aria-label="Dismiss alert"
                    >
                      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">{t('consciousness_alerts.alert_history')}</CardTitle>
              <div className="flex items-center gap-2">
                <select
                  value={historyFilter}
                  onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setHistoryFilter(e.target.value as AlertSeverity | 'all')}
                  className="text-xs bg-transparent border border-border rounded px-2 py-1"
                >
                  <option value="all">{t('consciousness_alerts.filter_all')}</option>
                  <option value="critical">{t('consciousness_alerts.filter_critical')}</option>
                  <option value="warning">{t('consciousness_alerts.filter_warning')}</option>
                  <option value="info">{t('consciousness_alerts.filter_info')}</option>
                </select>
                <Button variant="ghost" size="sm" onClick={clearHistory}>
                  {t('consciousness_alerts.clear_history')}
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {filteredHistory.length === 0 ? (
              <p className="text-sm text-muted-foreground py-4 text-center">{t('consciousness_alerts.no_history')}</p>
            ) : (
              <div className="space-y-1 max-h-96 overflow-y-auto">
                {filteredHistory.slice().reverse().map(alert => (
                  <div key={alert.id} className="flex items-center gap-3 py-2 px-2 rounded text-xs border border-border/30">
                    <span className={`inline-block w-2 h-2 rounded-full shrink-0 ${severityBadgeColor(alert.severity)}`} />
                    <span className="flex-1 truncate">{alert.message}</span>
                    <span className="text-muted-foreground shrink-0">{formatTimeAgo(alert.timestamp)}</span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">{t('consciousness_alerts.settings_title')}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{t('consciousness_alerts.notifications_label')}</p>
                <p className="text-xs text-muted-foreground">{t('consciousness_alerts.notifications_desc')}</p>
              </div>
              <Switch
                checked={settings.notifications}
                onCheckedChange={(checked) => updateSettings({ notifications: checked })}
                aria-label="Toggle browser notifications"
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{t('consciousness_alerts.sound_label')}</p>
                <p className="text-xs text-muted-foreground">{t('consciousness_alerts.sound_desc')}</p>
              </div>
              <Switch
                checked={settings.sound}
                onCheckedChange={(checked) => updateSettings({ sound: checked })}
                aria-label="Toggle alert sound"
              />
            </div>
            <div className="border-t border-border/30" />
            <div>
              <p className="text-sm font-medium mb-3">{t('consciousness_alerts.rules_title')}</p>
              <div className="space-y-3">
                {settings.rules.map(rule => (
                  <div key={rule.id} className="flex items-center gap-3">
                    <Switch
                      checked={rule.enabled}
                      onCheckedChange={(checked) => updateRule(rule.id, { enabled: checked })}
                      aria-label={`Toggle ${rule.label}`}
                    />
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium truncate">{rule.label}</p>
                    </div>
                    <Input
                      type="number"
                      step={rule.id === 'avg_growth' ? 0.01 : rule.id === 'positive_ratio' ? 0.1 : 1}
                      value={rule.threshold}
                      onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateRule(rule.id, { threshold: parseFloat(e.target.value) || 0 })}
                      className="w-20 h-7 text-xs text-right"
                      aria-label={`Threshold for ${rule.label}`}
                    />
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
