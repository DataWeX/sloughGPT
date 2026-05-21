'use client'

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'

import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  IconChat,
  IconModels,
  IconSettings,
} from '@/components/icons/NavIcons'
import { IconChevronRight, IconMessage } from '@/components/ui'
import { IconCompare } from '@/components/icons/NavIcons'
import { useApiHealth } from '@/hooks/useApiHealth'
import { useLocale } from '@/hooks/useLocale'
import { modelController } from '@/lib/model-controller'
import { soulsController } from '@/lib/souls-controller'
import { sessionController } from '@/lib/session-controller'
import { PUBLIC_API_URL } from '@/lib/config'

function Greeting() {
  const h = new Date().getHours()
  let greeting: string
  if (h < 5) { greeting = 'Burning the midnight oil' }
  else if (h < 12) { greeting = 'Good morning' }
  else if (h < 14) { greeting = 'Good afternoon' }
  else if (h < 18) { greeting = 'Good afternoon' }
  else if (h < 22) { greeting = 'Good evening' }
  else { greeting = 'Late night vibes' }
  return <span>{greeting}</span>
}

export default function HomePage() {
  const { t } = useLocale()
  const { state: health } = useApiHealth()
  const [modelCount, setModelCount] = useState<number | null>(null)
  const [modelStatus, setModelStatus] = useState<{loaded: boolean; model: string | null}>({loaded: false, model: null})
  const inferenceCount = health && health !== 'offline' ? (health as any).inference_count ?? 0 : null
  const [currentSoul, setCurrentSoul] = useState<{name: string; description: string; traits: string[]} | null>(null)
  const [recentSessions, setRecentSessions] = useState<Array<{id: string; name: string; updated_at: string}>>([])

  const apiStatus = useMemo<'loading' | 'online' | 'offline'>(() => {
    if (health === null) return 'loading'
    if (health === 'offline') return 'offline'
    return 'online'
  }, [health])

  useEffect(() => {
    if (apiStatus !== 'online') return
    modelController.status().then(status => {
      setModelStatus({ loaded: status.loaded, model: status.model_type })
    }).catch(() => {})
    soulsController.list().then(data => {
      const active = data.souls?.find((s: any) => s.name === data.current_soul)
      if (active) setCurrentSoul(active)
    }).catch(() => {})
  }, [apiStatus])

  useEffect(() => {
    if (health === null || health === 'offline') { setModelCount(null); return }
    let cancelled = false
    modelController.list().then(models => { if (!cancelled) setModelCount(models.length) }).catch(() => {})
    sessionController.list().then(sessions => {
      if (!cancelled) {
        const sorted = [...sessions]
          .filter(s => s.name)
          .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
          .slice(0, 5)
        setRecentSessions(sorted)
      }
    }).catch(() => {})
    return () => { cancelled = true }
  }, [health])

  const subtitleKey = apiStatus === 'loading' ? 'home.subtitle.connecting'
    : apiStatus === 'online' ? 'home.subtitle.ready'
    : 'home.subtitle.offline'

  return (
    <div className="sl-page mx-auto max-w-4xl space-y-4">
      <AppRouteHeader
        left={<AppRouteHeaderLead title={<Greeting />} subtitle={t(subtitleKey)} />}
      />

      {apiStatus === 'offline' ? (
        <Card className="border-amber-500/35 bg-amber-500/5">
          <CardHeader>
            <CardTitle className="text-base">{t('home.apiOffline.title')}</CardTitle>
          </CardHeader>
          <CardContent className="text-sm">
            {t('home.apiOffline.body', { url: PUBLIC_API_URL })}
          </CardContent>
        </Card>
      ) : null}

      {apiStatus === 'offline' ? null : (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Card>
            <CardHeader className="pb-1">
              <CardDescription className="text-xs font-medium">{t('home.stats.status')}</CardDescription>
            </CardHeader>
            <CardContent>
              {apiStatus === 'loading' ? (
                <div className="h-8 w-24 animate-pulse rounded bg-muted" />
              ) : (
                <div className="flex items-center gap-2">
                  <span className="relative flex h-3 w-3">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-success/40" />
                    <span className="relative inline-flex h-3 w-3 rounded-full bg-success" />
                  </span>
                  <p className="text-base font-semibold">Online</p>
                </div>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-1">
              <CardDescription className="text-xs font-medium">{t('home.stats.models')}</CardDescription>
            </CardHeader>
            <CardContent>
              {apiStatus === 'loading' ? (
                <div className="h-8 w-16 animate-pulse rounded bg-muted" />
              ) : (
                <p className="text-base font-semibold tabular-nums">{modelCount !== null ? modelCount : '\u2014'}</p>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-1">
              <CardDescription className="text-xs font-medium">{t('home.stats.personality')}</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-base font-semibold truncate">{currentSoul?.name || '\u2014'}</p>
            </CardContent>
          </Card>
          <Card className="bg-gradient-to-br from-primary/5 to-transparent border-primary/20">
            <CardHeader className="pb-1">
              <CardDescription className="text-xs font-medium">Model</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-base font-semibold truncate">{modelStatus.loaded ? modelStatus.model : 'Not loaded'}</p>
              {inferenceCount !== null && inferenceCount !== undefined && (
                <p className="text-[10px] text-muted-foreground mt-0.5 tabular-nums">{inferenceCount} inferences</p>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3">
        <Link
          href="/chat"
          className="group relative overflow-hidden rounded-lg border border-primary/20 bg-gradient-to-br from-primary/10 via-primary/5 to-transparent p-5 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/30"
        >
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/15 text-primary">
              <IconChat className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-semibold">Start chatting</p>
              <p className="text-xs text-muted-foreground mt-0.5">Ask anything, get answers instantly</p>
            </div>
          </div>
          <div className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
            <IconChevronRight className="h-6 w-6" />
          </div>
        </Link>
        <Link
          href="/models"
          className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/50 to-transparent p-5 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20"
        >
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/15 text-accent">
              <IconModels className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-semibold">Browse models</p>
              <p className="text-xs text-muted-foreground mt-0.5">Manage personalities and checkpoints</p>
            </div>
          </div>
          <div className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
            <IconChevronRight className="h-6 w-6" />
          </div>
        </Link>
        <Link
          href="/settings"
          className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/50 to-transparent p-5 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20"
        >
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted-foreground/15 text-muted-foreground">
              <IconSettings className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-semibold">Settings</p>
              <p className="text-xs text-muted-foreground mt-0.5">Customize your experience</p>
            </div>
          </div>
          <div className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
            <IconChevronRight className="h-6 w-6" />
          </div>
        </Link>
        <Link
          href="/compare"
          className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/50 to-transparent p-5 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20"
        >
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/15 text-accent">
              <IconCompare className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-semibold">Compare models</p>
              <p className="text-xs text-muted-foreground mt-0.5">Side-by-side metrics and performance</p>
            </div>
          </div>
          <div className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
            <IconChevronRight className="h-6 w-6" />
          </div>
        </Link>
        <Link
          href="/training"
          className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/50 to-transparent p-5 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20"
        >
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/15 text-accent">
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
            </div>
            <div>
              <p className="text-sm font-semibold">Train</p>
              <p className="text-xs text-muted-foreground mt-0.5">Fine-tune a model on your data</p>
            </div>
          </div>
          <div className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
            <IconChevronRight className="h-6 w-6" />
          </div>
        </Link>
        <Link
          href="/datasets"
          className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/50 to-transparent p-5 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20"
        >
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/15 text-accent">
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"/></svg>
            </div>
            <div>
              <p className="text-sm font-semibold">Datasets</p>
              <p className="text-xs text-muted-foreground mt-0.5">Browse and manage training data</p>
            </div>
          </div>
          <div className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
            <IconChevronRight className="h-6 w-6" />
          </div>
        </Link>
      </div>

      {recentSessions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Recent conversations</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="divide-y divide-border/30">
              {recentSessions.map(s => (
                <Link key={s.id} href={`/chat?id=${s.id}`} className="flex items-center gap-3 px-4 py-3 hover:bg-muted/30 transition-colors">
                  <IconMessage className="h-4 w-4 shrink-0 text-muted-foreground/60" />
                  <span className="text-sm truncate flex-1">{s.name}</span>
                  <span className="text-[10px] text-muted-foreground shrink-0">
                    {new Date(s.updated_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                  </span>
                </Link>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
