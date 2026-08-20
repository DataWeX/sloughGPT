'use client'

export const dynamic = 'force-dynamic'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'

import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import {
  IconChat,
  IconModels,
} from '@/components/icons/NavIcons'
import { IconChevronRight, IconMessage, IconSearch, IconBolt, IconChart, LossCurve } from '@sloughgpt/strui'

import { apiGet } from '@/lib/http-client'
import { extractErrorMessage } from '@/lib/error-utils'
import { chatController } from '@/lib/chat-controller'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { useLocale } from '@/hooks/useLocale'
import { knowledgeController } from '@/lib/knowledge-controller'
import { useToastStore } from '@/lib/toast-store'
import { sessionController } from '@/lib/session-controller'
import { datasetController } from '@/lib/dataset-controller'
import { PUBLIC_API_URL } from '@/lib/config'
import { chatDB } from '@/lib/db'
import { useHomePageData } from '@/hooks/useHomePageData'
import { formatUptime } from '@/lib/chat-utils'
import { MS_PER_MINUTE, formatBytes } from '@/lib/format-bytes'

function Greeting() {
  const [greeting, setGreeting] = useState('Hello')
  useEffect(() => {
    const h = new Date().getHours()
    if (h < 5) setGreeting('Burning the midnight oil')
    else if (h < 12) setGreeting('Good morning')
    else if (h < 18) setGreeting('Good afternoon')
    else if (h < 22) setGreeting('Good evening')
    else setGreeting('Late night vibes')
  }, [])
  return <span>{greeting}</span>
}

export default function HomePage() {
  const router = useRouter()
  const { t } = useLocale()
  const { healthLegacy: health, health: liveHealth } = useLiveStatus()
  const addToast = useToastStore(s => s.addToast)
  const { modelCount, currentSoul, modelStatus, inferenceCount, runningTraining, knowledgeCount, recentSessions, recentJobs, recentDatasets, healthSummary, feedbackStats, ...data } = useHomePageData(health)

  const apiStatus = health === null ? 'loading' : health === 'offline' ? 'offline' : 'online'

  const [startup, setStartup] = useState<{phase: string; step: number; total: number; message: string} | null>(null)
  const [onboardingDismissed, setOnboardingDismissed] = useState<boolean>(false)

  useEffect(() => {
    let cancelled = false
    chatDB.getKV<string>('onboarding_dismissed').then(v => {
      if (!cancelled) setOnboardingDismissed(v === '1')
    })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if (onboardingDismissed) {
      chatDB.setKV('onboarding_dismissed', '1').catch(() => {})
    } else {
      chatDB.deleteKV('onboarding_dismissed').catch(() => {})
    }
  }, [onboardingDismissed])

  const [convStats, setConvStats] = useState<{ totalConversations: number; totalMessages: number; totalWords: number; activeDays: number; mostActiveHour: number | null } | null>(null)

  const [datasetStats, setDatasetStats] = useState<{ totalDatasets: number; totalSize: number; totalSamples: number } | null>(null)

  useEffect(() => {
    if (apiStatus !== 'online') return
    let cancelled = false
    sessionController.list().then(sessions => {
      if (cancelled) return
      let totalMessages = 0
      let totalWords = 0
      const days = new Set<string>()
      const hourCounts = new Array(24).fill(0)
      for (const s of sessions) {
        totalMessages += s.messages?.length || 0
        for (const m of s.messages || []) {
          totalWords += m.content ? m.content.split(/\s+/).length : 0
          if (m.timestamp) {
            const d = new Date(m.timestamp)
            days.add(d.toISOString().slice(0, 10))
            hourCounts[d.getHours()]++
          }
        }
        if (s.updated_at) days.add(new Date(s.updated_at).toISOString().slice(0, 10))
      }
      const maxHour = hourCounts.indexOf(Math.max(...hourCounts))
      setConvStats({
        totalConversations: sessions.length,
        totalMessages,
        totalWords,
        activeDays: days.size,
        mostActiveHour: days.size > 0 ? maxHour : null,
      })
    }).catch(() => { /* session stats non-critical */ })
    return () => { cancelled = true }
  }, [apiStatus])

  useEffect(() => {
    if (apiStatus !== 'online') return
    let cancelled = false
    datasetController.list().then(list => {
      if (cancelled) return
      setDatasetStats({
        totalDatasets: list.length,
        totalSize: list.reduce((sum, ds) => sum + (ds.size || 0), 0),
        totalSamples: list.reduce((sum, ds) => sum + (ds.samples || 0), 0),
      })
    }).catch(() => { /* dataset stats non-critical */ })
    return () => { cancelled = true }
  }, [apiStatus])

  useEffect(() => {
    if (apiStatus !== 'offline') return
    let cancelled = false
    const poll = async () => {
      try {
        const result = await apiGet<{phase: string; step: number; total: number; message: string}>('/health/startup-progress')
        if (!cancelled) setStartup(result)
      } catch {
        if (!cancelled) setStartup(null)
      }
    }
    poll()
    const id = setInterval(poll, 3000)
    return () => { cancelled = true; clearInterval(id) }
  }, [apiStatus])

  function subtitleText(): string {
    if (apiStatus === 'loading') return 'Connecting...'
    if (apiStatus === 'offline') return t('home.subtitle.offline')
    if (healthSummary) {
      const shortName = healthSummary.split('/').pop() || healthSummary
      const convs = inferenceCount !== null ? `${inferenceCount} conversation${inferenceCount === 1 ? '' : 's'}` : null
      return convs ? `${shortName} loaded · ${convs}` : `${shortName} loaded`
    }
    return 'Your AI, your way'
  }

  return (
    <PageContainer
      title={<Greeting />}
      subtitle={subtitleText()}
      className="relative"
      contentClassName="space-y-4"
    >
      {/* Decorative training loss curve — signature element */}
      <div className="absolute top-0 right-0 w-48 h-32 opacity-[0.06] pointer-events-none overflow-hidden hidden sm:block" aria-hidden="true">
        <LossCurve className="w-full h-full" />
      </div>

      {apiStatus === 'offline' ? (
        startup ? (
          <Card className="border-warning/35 bg-warning/5">
            <CardHeader>
              <CardTitle className="text-base">Starting up… ({startup.step}/{startup.total})</CardTitle>
            </CardHeader>
            <CardContent className="text-sm space-y-2">
              <p>{startup.message}</p>
              <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
                <div className="h-full bg-warning transition-all duration-500" style={{width: `${(startup.step / startup.total) * 100}%`}} />
              </div>
              <p className="text-xs text-muted-foreground">Server may take 90s on first cold start (PyTorch import).</p>
            </CardContent>
          </Card>
        ) : (
          <Card className="border-warning/35 bg-warning/5">
            <CardHeader>
              <CardTitle className="text-base">{t('home.apiOffline.title')}</CardTitle>
            </CardHeader>
            <CardContent className="text-sm">
              {t('home.apiOffline.body', { url: PUBLIC_API_URL })}
            </CardContent>
          </Card>
        )
      ) : null}

      {apiStatus === 'offline' ? null : (
        <>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Card className="flex flex-col justify-between">
              <CardHeader className="pb-1 px-4 pt-4">
                <CardDescription className="text-xs font-medium text-muted-foreground">{t('home.stats.status')}</CardDescription>
              </CardHeader>
              <CardContent className="pb-4 px-4">
                {apiStatus === 'loading' ? (
                  <div className="h-6 w-20 animate-pulse rounded bg-muted" />
                ) : (
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-2 w-2">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-success/40" />
                      <span className="relative inline-flex h-2 w-2 rounded-full bg-success" />
                    </span>
                    <p className="text-sm font-semibold">Online</p>
                  </div>
                )}
              </CardContent>
            </Card>
            <Card className="flex flex-col justify-between">
              <CardHeader className="pb-1 px-4 pt-4">
                <CardDescription className="text-xs font-medium text-muted-foreground">{t('home.stats.models')}</CardDescription>
              </CardHeader>
              <CardContent className="pb-4 px-4">
                {apiStatus === 'loading' ? (
                  <div className="h-6 w-12 animate-pulse rounded bg-muted" />
                ) : (
                  <p className="text-sm font-semibold tabular-nums">{modelCount !== null ? modelCount : '\u2014'}</p>
                )}
              </CardContent>
            </Card>
            <Card className="flex flex-col justify-between">
              <CardHeader className="pb-1 px-4 pt-4">
                <CardDescription className="text-xs font-medium text-muted-foreground">{t('home.stats.personality')}</CardDescription>
              </CardHeader>
              <CardContent className="pb-4 px-4">
                <p className="text-sm font-semibold truncate">{currentSoul?.name || '\u2014'}</p>
              </CardContent>
            </Card>
            <Card className="bg-gradient-to-br from-accent/5 to-transparent border-accent/20 flex flex-col justify-between">
              <CardHeader className="pb-1 px-4 pt-4">
                <CardDescription className="text-xs font-medium text-muted-foreground">Active</CardDescription>
              </CardHeader>
              <CardContent className="pb-4 px-4">
                <p className="text-sm font-semibold truncate">{modelStatus.loaded ? `${modelStatus.model} + ${currentSoul?.name || 'default'}` : 'Not loaded'}</p>
                {inferenceCount !== null && inferenceCount !== undefined && (
                  <p className="text-[10px] text-muted-foreground mt-0.5 tabular-nums">{inferenceCount} conversations</p>
                )}
              </CardContent>
            </Card>
          </div>

          {feedbackStats?.db_stats && feedbackStats.db_stats.feedback_total > 0 && (
            <Card>
              <CardContent className="py-3">
                <div className="flex items-center gap-3 flex-wrap">
                  <div className="flex items-center gap-1.5">
                    <span className="text-sm text-muted-foreground">Feedback</span>
                    <span className="text-sm font-medium tabular-nums">{feedbackStats.db_stats.feedback_total}</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-sm text-success">👍 {feedbackStats.db_stats.thumbs_up}</span>
                    <span className="text-sm text-destructive">👎 {feedbackStats.db_stats.thumbs_down}</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-sm text-muted-foreground">Ratio</span>
                    <span className={`text-sm font-medium ${feedbackStats.db_stats.ratio >= 0.5 ? 'text-success' : 'text-warning'}`}>
                      {Math.round(feedbackStats.db_stats.ratio * 100)}% positive
                    </span>
                  </div>
                  <Link href="/training" className="ml-auto text-sm text-primary hover:text-primary/80 shrink-0">
                    Train from feedback →
                  </Link>
                </div>
              </CardContent>
            </Card>
          )}

          {runningTraining && (
            <Link href="/training" className="block">
              <Card className="border-primary/30 bg-primary/5 cursor-pointer hover:border-primary/50 transition-colors">
                <CardContent className="py-3">
                  <div className="flex items-center gap-3">
                    <span className="relative flex h-2 w-2 shrink-0">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
                      <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium truncate">Training: {runningTraining.name}</p>
                      <p className="text-xs text-muted-foreground truncate">{runningTraining.status_message}</p>
                    </div>
                    <span className="text-sm text-primary shrink-0">View →</span>
                  </div>
                </CardContent>
              </Card>
            </Link>
          )}
        </>
      )}

      {apiStatus === 'online' && modelStatus.loaded && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <Card>
            <CardContent className="py-3">
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium">Quick test</p>
                  <p className="text-xs text-muted-foreground">Send &quot;Hello!&quot; to verify the model works</p>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8 text-xs shrink-0"
                  disabled={data.testRunning}
                  onClick={async () => {
                    data.setTestRunning(true)
                    data.setTestResponse(null)
                    try {
                      const result = await chatController.send('Hello!')
                      data.setTestResponse(result.message || 'No response')
                    } catch (e: unknown) {
                      data.setTestResponse(extractErrorMessage(e, 'Failed to connect'))
                    } finally {
                      data.setTestRunning(false)
                    }
                  }}
                >
                  {data.testRunning ? 'Testing...' : 'Test model'}
                </Button>
              </div>
              {data.testResponse && (
                <div className="mt-2 rounded bg-muted/50 p-2 text-xs text-muted-foreground font-mono leading-relaxed">
                  {data.testResponse}
                </div>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardContent className="py-3">
              <div className="flex items-center gap-2 mb-2">
                <p className="text-sm font-medium">Quick note</p>
                <p className="text-xs text-muted-foreground">Add a fact the AI can remember</p>
              </div>
              <form onSubmit={async (e) => {
                e.preventDefault()
                const input = e.currentTarget.querySelector('input') as HTMLInputElement
                const text = input.value.trim()
                if (!text) return
                try {
                  await knowledgeController.add(text, 'general')
                  input.value = ''
                  data.setKnowledgeCount(k => k + 1)
                  addToast('Fact saved', 'success')
                } catch { addToast('Failed to save', 'error') }
              }} className="flex gap-2">
                <input
                  type="text"
                  placeholder="e.g., I prefer Python over JavaScript"
                  className="flex-1 h-9 rounded-md border border-border/60 bg-background px-2.5 text-sm placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-primary/30"
                />
                <Button size="sm" type="submit" className="h-8 text-xs shrink-0">Save</Button>
              </form>
            </CardContent>
          </Card>
        </div>
      )}

      {apiStatus === 'online' && !modelStatus.loaded && (
        <Card className="border-dashed border-border/60 bg-muted/20">
          <CardContent className="py-3 flex items-center justify-between gap-3">
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium">No model loaded</p>
              <p className="text-xs text-muted-foreground">Load a model in Personalities to start chatting</p>
            </div>
            <Button size="sm" variant="outline" className="h-8 text-xs shrink-0" onClick={() => router.push('/models')}>
              Open Models
            </Button>
          </CardContent>
        </Card>
      )}

      {apiStatus === 'online' && modelStatus.loaded && (
        <Card>
          <CardContent className="py-3">
            <p className="text-sm font-medium mb-2">Recent activity</p>
            <div className="space-y-1.5">
              {recentSessions.slice(0, 3).map(s => (
                <button
                  key={s.id}
                  onClick={() => router.push(`/chat?session=${s.id}`)}
                  className="w-full flex items-center gap-2 text-left hover:bg-muted/30 rounded px-1.5 py-1 transition-colors"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-primary/60 shrink-0" />
                  <span className="text-xs truncate flex-1">{s.name}</span>
                  {s.starred && <span className="text-xs shrink-0">★</span>}
                  {s.pinned && <span className="text-xs text-primary shrink-0">📌</span>}
                  <span className="text-xs text-muted-foreground shrink-0">
                    {s.message_count != null && <span>{s.message_count}m · </span>}
                    {(() => {
                      const d = Date.now() - new Date(s.updated_at).getTime()
                      const m = Math.floor(d / MS_PER_MINUTE)
                      if (m < 1) return 'now'
                      if (m < 60) return `${m}m`
                      const h = Math.floor(m / 60)
                      return h < 24 ? `${h}h` : `${Math.floor(h / 24)}d`
                    })()}
                  </span>
                </button>
              ))}
              {recentJobs.slice(0, 2).map(j => (
                <div key={j.id} className="flex items-center gap-2 px-1.5 py-1">
                  <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${j.status === 'running' ? 'bg-success animate-pulse' : j.status === 'completed' ? 'bg-success' : j.status === 'failed' ? 'bg-destructive' : 'bg-muted-foreground/40'}`} />
                  <span className="text-xs truncate flex-1">{j.name || j.id}</span>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${j.status === 'running' ? 'bg-warning/15 text-warning' : j.status === 'completed' ? 'bg-success/15 text-success' : j.status === 'failed' ? 'bg-destructive/15 text-destructive' : 'bg-muted text-muted-foreground'}`}>{j.status}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {apiStatus === 'online' && recentDatasets.length > 0 && (
        <Card>
          <CardContent className="py-3">
            <div className="flex items-center gap-2 mb-2">
              <p className="text-sm font-medium">Recent datasets</p>
              <Link href="/datasets" className="text-[10px] text-primary hover:text-primary/80 ml-auto">View all →</Link>
            </div>
            <div className="space-y-1.5">
              {recentDatasets.map(ds => (
                <button
                  key={ds.id}
                  onClick={() => router.push(`/training?dataset=${encodeURIComponent(ds.id)}`)}
                  className="w-full flex items-center gap-2 text-left hover:bg-muted/30 rounded px-1.5 py-1 transition-colors"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-accent/60 shrink-0" />
                  <span className="text-xs truncate flex-1">{ds.name}</span>
                  {ds.samples != null && <span className="text-xs text-muted-foreground shrink-0">{ds.samples.toLocaleString()} samples</span>}
                  <span className="text-[10px] text-primary shrink-0">Train →</span>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {!onboardingDismissed && (
        <Card className="border-dashed border-primary/20 bg-primary/[0.02]">
          <CardContent className="py-3 flex items-center justify-between gap-3">
            <p className="text-sm text-muted-foreground">
              <span className="font-medium text-foreground">New here?</span>{' '}
              Start chatting, then train the model on your conversations. The more you chat, the better it gets.
            </p>
            <button
              type="button"
              className="text-xs text-muted-foreground hover:text-foreground shrink-0 transition-colors"
              onClick={() => setOnboardingDismissed(true)}
            >
              Got it
            </button>
          </CardContent>
        </Card>
      )}

      {recentSessions.length > 0 && (
        <button
          onClick={() => router.push(`/chat?session=${recentSessions[0].id}`)}
          className="w-full text-left rounded-lg border border-border/30 bg-card p-3 flex items-center gap-3 hover:border-border/60 hover:bg-muted/30 transition-colors group"
        >
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-muted/60 text-muted-foreground group-hover:text-foreground transition-colors">
            <IconMessage className="h-4 w-4" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <p className="text-xs font-medium truncate">{recentSessions[0].name}</p>
              {recentSessions[0].starred && <span className="text-[10px]">★</span>}
              {recentSessions[0].pinned && <span className="text-[10px] text-primary">📌</span>}
            </div>
            <p className="text-[10px] text-muted-foreground">
              {recentSessions[0].message_count != null && <span>{recentSessions[0].message_count} messages · </span>}
              {new Date(recentSessions[0].updated_at).toLocaleDateString()}
            </p>
          </div>
          <IconChevronRight className="h-3.5 w-3.5 text-muted-foreground/40 group-hover:text-primary/60 transition-colors" />
        </button>
      )}

      {apiStatus === 'online' && (convStats || datasetStats) && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {convStats && convStats.totalConversations > 0 && (
            <Card>
              <CardContent className="py-3">
                <div className="flex items-center gap-2 mb-2">
                  <p className="text-xs font-medium">Your stats</p>
                  <p className="text-[10px] text-muted-foreground">Usage overview</p>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <p className="text-lg font-semibold tabular-nums">{convStats.totalConversations}</p>
                    <p className="text-[10px] text-muted-foreground">Conversations</p>
                  </div>
                  <div>
                    <p className="text-lg font-semibold tabular-nums">{convStats.totalMessages.toLocaleString()}</p>
                    <p className="text-[10px] text-muted-foreground">Messages</p>
                  </div>
                  <div>
                    <p className="text-lg font-semibold tabular-nums">{convStats.totalWords.toLocaleString()}</p>
                    <p className="text-[10px] text-muted-foreground">Words</p>
                  </div>
                  <div>
                    <p className="text-lg font-semibold tabular-nums">{convStats.activeDays}</p>
                    <p className="text-[10px] text-muted-foreground">Active days</p>
                  </div>
                </div>
                {convStats.mostActiveHour !== null && (
                  <p className="text-[10px] text-muted-foreground mt-2">
                    Most active at {convStats.mostActiveHour}:00
                  </p>
                )}
              </CardContent>
            </Card>
          )}
          {datasetStats && datasetStats.totalDatasets > 0 && (
            <Card>
              <CardContent className="py-3">
                <div className="flex items-center gap-2 mb-2">
                  <p className="text-xs font-medium">Datasets</p>
                  <Link href="/datasets" className="text-[10px] text-primary hover:text-primary/80 ml-auto">View all →</Link>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div>
                    <p className="text-lg font-semibold tabular-nums">{datasetStats.totalDatasets}</p>
                    <p className="text-[10px] text-muted-foreground">Datasets</p>
                  </div>
                  <div>
                    <p className="text-lg font-semibold tabular-nums">
                      {formatBytes(datasetStats.totalSize)}
                    </p>
                    <p className="text-[10px] text-muted-foreground">Total size</p>
                  </div>
                  <div>
                    <p className="text-lg font-semibold tabular-nums">{datasetStats.totalSamples.toLocaleString()}</p>
                    <p className="text-[10px] text-muted-foreground">Samples</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {apiStatus === 'online' && liveHealth && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <Card>
            <CardContent className="py-3">
              <div className="flex items-center gap-2 mb-2">
                <p className="text-xs font-medium">System</p>
                <Link href="/monitoring" className="text-[10px] text-primary hover:text-primary/80 ml-auto">Details →</Link>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div>
                  <p className="text-sm font-semibold tabular-nums">
                    {liveHealth.cpu_percent !== null ? `${Math.round(liveHealth.cpu_percent)}%` : '—'}
                  </p>
                  <p className="text-[10px] text-muted-foreground">CPU</p>
                </div>
                <div>
                  <p className="text-sm font-semibold tabular-nums">
                    {liveHealth.memory_percent !== null ? `${Math.round(liveHealth.memory_percent)}%` : '—'}
                  </p>
                  <p className="text-[10px] text-muted-foreground">Memory</p>
                </div>
                <div>
                  <p className="text-sm font-semibold tabular-nums">{(liveHealth.request_count ?? 0).toLocaleString()}</p>
                  <p className="text-[10px] text-muted-foreground">Requests</p>
                </div>
                <div>
                  <p className="text-sm font-semibold tabular-nums">
                    {liveHealth.uptime_seconds > 0 ? formatUptime(liveHealth.uptime_seconds) : '—'}
                  </p>
                  <p className="text-[10px] text-muted-foreground">Uptime</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <div className="rounded-lg border border-border/60 p-3 sm:p-4 flex flex-col justify-between">
            <div>
              <div className="text-xs font-medium text-muted-foreground mb-2">How it works</div>
              <div className="space-y-2">
                <p className="text-[11px] text-muted-foreground/70">
                  Mix and match AI models with personalities. Each one has its own voice and style.
                </p>
                <p className="text-[11px] text-muted-foreground/70">
                  Import text, files, or conversations. The AI learns from your data and gets better over time.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {apiStatus === 'online' && datasetStats && datasetStats.totalDatasets > 0 && modelStatus.loaded && (
        <Card className="border-accent/25 bg-accent/[0.03]">
          <CardContent className="py-3 flex items-center justify-between gap-3">
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium">Ready to train</p>
              <p className="text-xs text-muted-foreground">
                {datasetStats.totalDatasets} dataset{datasetStats.totalDatasets === 1 ? '' : 's'} available · {formatBytes(datasetStats.totalSize)}
              </p>
            </div>
            <Button size="sm" className="h-8 text-xs shrink-0" onClick={() => router.push('/training')}>
              Start training
            </Button>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 sm:gap-3">
        <Link
          href="/chat"
          className="group relative overflow-hidden rounded-lg border border-accent/25 bg-gradient-to-br from-accent/8 via-accent/3 to-transparent p-3 sm:p-4 transition-all hover:shadow-lg hover:shadow-accent/8 hover:border-accent/40"
        >
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="flex h-8 w-8 sm:h-9 sm:w-9 items-center justify-center rounded-lg bg-accent/15 text-accent">
              <IconChat className="h-4 w-4 sm:h-5 sm:w-5" />
            </div>
            <div className="min-w-0">
              <p className="text-xs sm:text-sm font-semibold">Start chatting</p>
              <p className="hidden sm:block text-[11px] text-muted-foreground mt-0.5">Ask anything, get answers</p>
            </div>
          </div>
          <div className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-accent/50 transition-colors">
            <IconChevronRight className="h-4 w-4 sm:h-5 sm:w-5" />
          </div>
        </Link>
        <Link
          href="/models"
          className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/40 to-transparent p-3 sm:p-4 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20"
        >
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="flex h-8 w-8 sm:h-9 sm:w-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <IconModels className="h-4 w-4 sm:h-5 sm:w-5" />
            </div>
            <div className="min-w-0">
              <p className="text-xs sm:text-sm font-semibold">Personalities</p>
              <p className="hidden sm:block text-[11px] text-muted-foreground mt-0.5">Switch your agent&apos;s personality</p>
            </div>
          </div>
          <div className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
            <IconChevronRight className="h-4 w-4 sm:h-5 sm:w-5" />
          </div>
        </Link>
        <Link
          href="/datasets"
          className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/40 to-transparent p-3 sm:p-4 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20"
        >
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="flex h-8 w-8 sm:h-9 sm:w-9 items-center justify-center rounded-lg bg-muted-foreground/10 text-muted-foreground">
              <IconSearch className="h-4 w-4 sm:h-5 sm:w-5" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <p className="text-xs sm:text-sm font-semibold">Datasets</p>
                {datasetStats && datasetStats.totalDatasets > 0 && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium">{datasetStats.totalDatasets}</span>
                )}
              </div>
              <p className="hidden sm:block text-[11px] text-muted-foreground mt-0.5">Manage training data</p>
            </div>
          </div>
          <div className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
            <IconChevronRight className="h-4 w-4 sm:h-5 sm:w-5" />
          </div>
        </Link>
        <Link
          href="/training"
          className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/40 to-transparent p-3 sm:p-4 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20"
        >
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="flex h-8 w-8 sm:h-9 sm:w-9 items-center justify-center rounded-lg bg-accent/10 text-accent">
              <IconBolt className="h-4 w-4 sm:h-5 sm:w-5" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <p className="text-xs sm:text-sm font-semibold">Teach me</p>
                {datasetStats && datasetStats.totalDatasets > 0 && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent/15 text-accent font-medium">{datasetStats.totalDatasets}</span>
                )}
              </div>
              <p className="hidden sm:block text-[11px] text-muted-foreground mt-0.5">Train from your writing</p>
            </div>
          </div>
          <div className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
            <IconChevronRight className="h-4 w-4 sm:h-5 sm:w-5" />
          </div>
        </Link>
        <Link
          href="/monitoring"
          className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/40 to-transparent p-3 sm:p-4 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20"
        >
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="flex h-8 w-8 sm:h-9 sm:w-9 items-center justify-center rounded-lg bg-success/10 text-success">
              <IconChart className="h-4 w-4 sm:h-5 sm:w-5" />
            </div>
            <div className="min-w-0">
              <p className="text-xs sm:text-sm font-semibold">System Health</p>
              <p className="hidden sm:block text-[11px] text-muted-foreground mt-0.5">Monitor API and resources</p>
            </div>
          </div>
          <div className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
            <IconChevronRight className="h-4 w-4 sm:h-5 sm:w-5" />
          </div>
        </Link>
        <Link
          href="/knowledge"
          className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/40 to-transparent p-3 sm:p-4 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20"
        >
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="flex h-8 w-8 sm:h-9 sm:w-9 items-center justify-center rounded-lg bg-primary/8 text-primary">
              <IconSearch className="h-4 w-4 sm:h-5 sm:w-5" />
            </div>
            <div className="min-w-0">
              <p className="text-xs sm:text-sm font-semibold">Knowledge</p>
              <p className="hidden sm:block text-[11px] text-muted-foreground mt-0.5">Facts the AI remembers</p>
            </div>
          </div>
          <div className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
            <IconChevronRight className="h-4 w-4 sm:h-5 sm:w-5" />
          </div>
        </Link>
      </div>

    </PageContainer>
  )
}
