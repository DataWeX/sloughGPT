'use client'

export const dynamic = 'force-dynamic'

import React, { Suspense, useEffect, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'

import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { IconChat, IconModels } from '@/components/icons/NavIcons'
import {
  IconChevronRight,
  IconMessage,
  LossCurve,
  SectionHeader,
  StatusDot,
} from '@sloughgpt/strui'

import { chatController } from '@/lib/chat-controller'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { useLocale } from '@/hooks/useLocale'
import { useToastStore } from '@/lib/toast-store'
import { OnboardingCard } from '@/components/onboarding/OnboardingCard'
import { chatDB } from '@/lib/db'
import { useHomePageData } from '@/hooks/useHomePageData'
import { useBootstrap } from '@/contexts/BootstrapContext'
import type { BootstrapMembers } from '@/lib/bootstrap-controller'
import { formatBytes } from '@/lib/format-bytes'
import { useModelReadiness } from '@/lib/store'
import { formatUptime } from '@/lib/chat-utils'

import {
  ActiveModelBannerSkeleton,
  StatsGridSkeleton,
  FeedbackBarSkeleton,
  TrainingStatusSkeleton,
  QuickActionsSkeleton,
  RecentActivitySkeleton,
  UsageStatsSkeleton,
  SystemHealthSkeleton,
  NavigationGridSkeleton,
} from '@/components/ui/HomePageSkeleton'

const LazyStatsGrid = React.lazy(() =>
  import('@/components/home/StatsGrid').then((m) => ({ default: m.StatsGrid })),
)
const LazyFeedbackBar = React.lazy(() =>
  import('@/components/home/FeedbackBar').then((m) => ({ default: m.FeedbackBar })),
)
const LazyTrainingStatus = React.lazy(() =>
  import('@/components/home/TrainingStatus').then((m) => ({ default: m.TrainingStatus })),
)
const LazyQuickActions = React.lazy(() =>
  import('@/components/home/QuickActions').then((m) => ({ default: m.QuickActions })),
)
const LazyRecentActivity = React.lazy(() =>
  import('@/components/home/RecentActivity').then((m) => ({ default: m.RecentActivity })),
)
const LazyUsageStats = React.lazy(() =>
  import('@/components/home/UsageStats').then((m) => ({ default: m.UsageStats })),
)
const LazySystemHealth = React.lazy(() =>
  import('@/components/home/SystemHealth').then((m) => ({ default: m.SystemHealth })),
)
const LazyNavigationGrid = React.lazy(() =>
  import('@/components/home/NavigationGrid').then((m) => ({ default: m.NavigationGrid })),
)

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

/** Full-page skeleton shown immediately on load — no data needed */
function HomePageSkeleton() {
  return (
    <div className="space-y-4">
      <ActiveModelBannerSkeleton />
      <TrainingStatusSkeleton />
      <StatsGridSkeleton />
      <FeedbackBarSkeleton />
      <QuickActionsSkeleton />
      <RecentActivitySkeleton />
      <UsageStatsSkeleton />
      <SystemHealthSkeleton />
      <NavigationGridSkeleton />
    </div>
  )
}

export default function HomePage() {
  const router = useRouter()
  const { t } = useLocale()
  const { healthLegacy: health, health: liveHealth } = useLiveStatus()
  const addToast = useToastStore((s) => s.addToast)
  const {
    modelCount,
    currentSoul,
    modelStatus,
    inferenceCount,
    runningTraining,
    knowledgeCount,
    recentSessions,
    recentJobs,
    recentDatasets,
    healthSummary,
    feedbackStats,
    ...data
  } = useHomePageData(health)

  // One payload serves every landing read below (slices, stats derivations,
  // banner placeholders) — the startup-progress poll it replaces lives in
  // BootstrapProvider, which stops once the server reports ready.
  const { data: bootstrap } = useBootstrap()

  const apiStatus = health === null ? 'loading' : health === 'offline' ? 'offline' : 'online'

  // Skeletons show until health status resolves (apiStatus leaves 'loading')
  const [pageReady, setPageReady] = useState(false)
  useEffect(() => {
    if (apiStatus !== 'loading') setPageReady(true)
  }, [apiStatus])

  const modelReadiness = useModelReadiness()
  const [onboardingDismissed, setOnboardingDismissed] = useState<boolean>(false)

  useEffect(() => {
    let cancelled = false
    chatDB.getKV<string>('onboarding_dismissed').then((v) => {
      if (!cancelled) setOnboardingDismissed(v === '1')
    })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (onboardingDismissed) {
      chatDB.setKV('onboarding_dismissed', '1').catch(() => {})
    } else {
      chatDB.deleteKV('onboarding_dismissed').catch(() => {})
    }
  }, [onboardingDismissed])

  const [convStats, setConvStats] = useState<{
    totalConversations: number
    totalMessages: number
    totalWords: number
    activeDays: number
    mostActiveHour: number | null
  } | null>(null)

  const [datasetStats, setDatasetStats] = useState<{
    totalDatasets: number
    totalSize: number
    totalSamples: number
  } | null>(null)

  // Conversation stats derive from the bootstrap sessions member — the
  // same payload useHomePageData consumes, no dedicated fetch.
  useEffect(() => {
    const sessions = bootstrap?.members.sessions
    if (!sessions) return
    let totalMessages = 0
    let totalWords = 0
    const days = new Set<string>()
    const hourCounts = new Array(24).fill(0)
    for (const s of sessions) {
      totalMessages += s.messages?.length || 0
      for (const m of s.messages || []) {
        if (m.content) totalWords += m.content.split(/\s+/).length
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
  }, [bootstrap])

  useEffect(() => {
    const datasets = bootstrap?.members.datasets
    if (!datasets) return
    setDatasetStats({
      totalDatasets: datasets.datasets.length,
      totalSize: datasets.datasets.reduce((sum, ds) => sum + (ds.size || 0), 0),
      totalSamples: datasets.datasets.reduce((sum, ds) => sum + (ds.samples || 0), 0),
    })
  }, [bootstrap])

  // Per-section pending: the member hasn't landed yet (bootstrap payload
  // absent, or member still null without a real error). Pending sections
  // render dimension-matched placeholders so nothing pops in mid-load.
  const memberPending = (key: keyof BootstrapMembers, errorKey: string = key) =>
    !bootstrap || (bootstrap.members[key] === null && bootstrap.errors?.[errorKey] === undefined)
  const soulsPending = memberPending('souls')
  const sessionsPending = memberPending('sessions')
  const datasetsPending = memberPending('datasets')

  function subtitleText(): string {
    if (apiStatus === 'loading') return 'Connecting...'
    if (apiStatus === 'offline') return t('home.subtitle.offline')
    if (healthSummary) {
      const shortName = healthSummary.split('/').pop() || healthSummary
      const convs =
        inferenceCount !== null
          ? `${inferenceCount} conversation${inferenceCount === 1 ? '' : 's'}`
          : null
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
      <div
        className="absolute top-0 right-0 w-48 h-32 opacity-[0.06] pointer-events-none overflow-hidden hidden sm:block"
        aria-hidden="true"
      >
        <LossCurve className="w-full h-full" />
      </div>

      {/* Show all skeletons immediately until health status resolves */}
      {!pageReady && <HomePageSkeleton />}

      {/* Real content — rendered once data is ready */}
      {pageReady && (
        <>
          {/* Eager: Active model banner (above fold) */}
          {apiStatus === 'online' && modelStatus.loaded && (
            <Card className="border-primary/20 bg-gradient-to-br from-primary/[0.04] via-transparent to-accent/[0.03]">
              <CardContent className="p-4 sm:p-5">
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <StatusDot tone="success" pulse />
                      <span className="text-xs font-medium text-success uppercase tracking-wider">
                        Active Model
                      </span>
                    </div>
                    <h3 className="text-sm font-semibold truncate">{healthSummary || 'Unknown'}</h3>
                    {soulsPending ? (
                      /* Dimension-matched placeholder: text-xs line, same as
                         the Personality row it becomes — no pop-in when the
                         souls member lands. */
                      <div
                        className="mt-0.5 h-4 w-24 rounded bg-muted/50 animate-pulse motion-reduce:animate-none"
                        aria-hidden="true"
                      />
                    ) : currentSoul ? (
                      <p className="text-xs text-muted-foreground mt-0.5">
                        Personality: <span className="text-foreground/80">{currentSoul.name}</span>
                      </p>
                    ) : null}
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <Link
                      href="/personality"
                      className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium text-primary bg-primary/10 hover:bg-primary/15 transition-colors"
                    >
                      <IconModels className="h-3 w-3" />
                      Switch
                    </Link>
                    <Link
                      href="/chat"
                      className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium text-accent-foreground bg-accent/15 hover:bg-accent/25 transition-colors"
                    >
                      <IconChat className="h-3 w-3" />
                      Chat
                    </Link>
                  </div>
                </div>
                {inferenceCount !== null && inferenceCount !== undefined && (
                  <div className="flex items-center gap-4 mt-3 pt-3 border-t border-border/40">
                    <span className="text-xs text-muted-foreground">
                      <span className="font-medium text-foreground/80 tabular-nums">
                        {inferenceCount.toLocaleString()}
                      </span>{' '}
                      conversations
                    </span>
                    {liveHealth?.tokens_per_sec && (
                      <span className="text-xs text-muted-foreground">
                        <span className="font-medium text-foreground/80 tabular-nums">
                          {liveHealth.tokens_per_sec.toFixed(1)}
                        </span>{' '}
                        tok/s
                      </span>
                    )}
                    {liveHealth?.uptime_seconds && liveHealth.uptime_seconds > 0 && (
                      <span className="text-xs text-muted-foreground">
                        Uptime:{' '}
                        <span className="font-medium text-foreground/80">
                          {formatUptime(liveHealth.uptime_seconds)}
                        </span>
                      </span>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Lazy: Training status / offline banner */}
          <Suspense fallback={<TrainingStatusSkeleton />}>
            <LazyTrainingStatus
              apiStatus={apiStatus}
              modelStatus={modelStatus}
              modelReadiness={modelReadiness}
              runningTraining={runningTraining}
            />
          </Suspense>

          {/* Lazy: Stats grid */}
          <Suspense fallback={<StatsGridSkeleton />}>
            <LazyStatsGrid
              apiStatus={apiStatus}
              modelCount={modelCount}
              currentSoul={currentSoul}
              modelStatus={modelStatus}
              inferenceCount={inferenceCount}
              t={t}
            />
          </Suspense>

          {/* Lazy: Feedback bar */}
          <Suspense fallback={<FeedbackBarSkeleton />}>
            <LazyFeedbackBar loading={apiStatus === 'loading'} feedbackStats={feedbackStats} />
          </Suspense>

          {/* Lazy: Quick actions */}
          <Suspense fallback={<QuickActionsSkeleton />}>
            <LazyQuickActions
              loading={apiStatus === 'loading'}
              modelStatus={modelStatus}
              testRunning={data.testRunning}
              testResponse={data.testResponse}
              setTestRunning={data.setTestRunning}
              setTestResponse={data.setTestResponse}
              knowledgeCount={knowledgeCount}
              setKnowledgeCount={data.setKnowledgeCount}
            />
          </Suspense>

          {/* Lazy: Recent activity */}
          <Suspense fallback={<RecentActivitySkeleton />}>
            <LazyRecentActivity
              apiStatus={apiStatus}
              loading={apiStatus === 'loading'}
              modelStatus={modelStatus}
              recentSessions={recentSessions}
              recentJobs={recentJobs}
              recentDatasets={recentDatasets}
            />
          </Suspense>

          {!onboardingDismissed && (
            <OnboardingCard onComplete={() => setOnboardingDismissed(true)} />
          )}

          {sessionsPending ? (
            /* Resume-session row placeholder — identical box to the real
               button (p-3, h-8 icon, two text-xs lines) so the banner
               lands without shifting anything below it. */
            <div
              className="w-full rounded-lg border border-border/30 bg-card p-3 flex items-center gap-3"
              aria-hidden="true"
            >
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-muted/60">
                <div className="h-4 w-4 rounded bg-muted animate-pulse motion-reduce:animate-none" />
              </div>
              <div className="min-w-0 flex-1 space-y-1.5">
                <div className="h-4 w-1/3 rounded bg-muted/50 animate-pulse motion-reduce:animate-none" />
                <div className="h-4 w-1/4 rounded bg-muted/30 animate-pulse motion-reduce:animate-none" />
              </div>
            </div>
          ) : recentSessions.length > 0 ? (
            <button
              type="button"
              onClick={() => router.push(`/chat?session=${recentSessions[0].id}`)}
              className="w-full text-left rounded-lg border border-border/30 bg-card p-3 flex items-center gap-3 hover:border-border/60 hover:bg-muted/30 transition-colors group"
            >
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-muted/60 text-muted-foreground group-hover:text-foreground transition-colors">
                <IconMessage className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <p className="text-xs font-medium truncate">
                    {recentSessions[0].name || 'Untitled conversation'}
                  </p>
                  {recentSessions[0].starred && <span className="text-xs">★</span>}
                  {recentSessions[0].pinned && <span className="text-xs text-primary">📌</span>}
                </div>
                <p className="text-xs text-muted-foreground">
                  {recentSessions[0].message_count != null && (
                    <span>{recentSessions[0].message_count} messages · </span>
                  )}
                  {new Date(recentSessions[0].updated_at).toLocaleDateString()}
                </p>
              </div>
              <IconChevronRight className="h-3.5 w-3.5 text-muted-foreground/40 group-hover:text-primary/60 transition-colors" />
            </button>
          ) : null}

          {/* Lazy: Usage stats */}
          <Suspense fallback={<UsageStatsSkeleton />}>
            <LazyUsageStats
              apiStatus={apiStatus}
              loading={apiStatus === 'loading'}
              convStats={convStats}
              datasetStats={datasetStats}
            />
          </Suspense>

          {/* Lazy: System health */}
          <Suspense fallback={<SystemHealthSkeleton />}>
            <LazySystemHealth
              apiStatus={apiStatus}
              loading={apiStatus === 'loading'}
              liveHealth={liveHealth}
            />
          </Suspense>

          {datasetsPending ? (
            /* "Ready to train" placeholder — text-sm title, text-xs body,
               h-8 button slot: same box as the loaded card. */
            <Card className="border-dashed border-border/50 bg-muted/10">
              <CardContent
                className="py-3 flex items-center justify-between gap-3"
                aria-hidden="true"
              >
                <div className="min-w-0 flex-1 space-y-1.5">
                  <div className="h-5 w-28 rounded bg-muted/50 animate-pulse motion-reduce:animate-none" />
                  <div className="h-4 w-44 rounded bg-muted/30 animate-pulse motion-reduce:animate-none" />
                </div>
                <div className="h-8 w-24 rounded-md bg-muted/50 animate-pulse motion-reduce:animate-none shrink-0" />
              </CardContent>
            </Card>
          ) : (
            apiStatus === 'online' &&
            datasetStats &&
            datasetStats.totalDatasets > 0 &&
            modelStatus.loaded && (
              <Card className="border-accent/25 bg-accent/[0.03]">
                <CardContent className="py-3 flex items-center justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium">Ready to train</p>
                    <p className="text-xs text-muted-foreground">
                      {datasetStats.totalDatasets} dataset
                      {datasetStats.totalDatasets === 1 ? '' : 's'} available ·{' '}
                      {formatBytes(datasetStats.totalSize)}
                    </p>
                  </div>
                  <Button
                    size="sm"
                    className="h-8 text-xs shrink-0"
                    onClick={() => router.push('/training')}
                  >
                    Start training
                  </Button>
                </CardContent>
              </Card>
            )
          )}

          {/* Lazy: Navigation grid */}
          <Suspense fallback={<NavigationGridSkeleton />}>
            <LazyNavigationGrid
              apiStatus={apiStatus}
              modelStatus={modelStatus}
              datasetStats={datasetStats}
            />
          </Suspense>
        </>
      )}
    </PageContainer>
  )
}
