'use client'

export const dynamic = 'force-dynamic'

import { useEffect, useState } from 'react'
import Link from 'next/link'

import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import {
  IconChat,
  IconModels,
} from '@/components/icons/NavIcons'
import { IconChevronRight, IconMessage, IconSearch } from '@/components/ui'

import { apiGet } from '@/lib/http-client'
import { useApiHealth } from '@/hooks/useApiHealth'
import { useLocale } from '@/hooks/useLocale'
import { knowledgeController } from '@/lib/knowledge-controller'
import { useToastStore } from '@/lib/toast-store'
import { PUBLIC_API_URL } from '@/lib/config'
import { useHomePageData } from '@/hooks/useHomePageData'

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
  const { t } = useLocale()
  const { state: health } = useApiHealth()
  const addToast = useToastStore(s => s.addToast)
  const { modelCount, currentSoul, modelStatus, inferenceCount, runningTraining, knowledgeCount, recentSessions, recentJobs, healthSummary, ...data } = useHomePageData(health)

  const apiStatus = health === null ? 'loading' : health === 'offline' ? 'offline' : 'online'

  const [startup, setStartup] = useState<{phase: string; step: number; total: number; message: string} | null>(null)

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
    <div className="sl-page mx-auto max-w-4xl space-y-4 overflow-hidden">
      <AppRouteHeader
        left={<AppRouteHeaderLead title={<Greeting />} subtitle={subtitleText()} />}
      />

      {apiStatus === 'offline' ? (
        startup ? (
          <Card className="border-amber-500/35 bg-amber-500/5">
            <CardHeader>
              <CardTitle className="text-base">Starting up… ({startup.step}/{startup.total})</CardTitle>
            </CardHeader>
            <CardContent className="text-sm space-y-2">
              <p>{startup.message}</p>
              <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
                <div className="h-full bg-amber-400 transition-all duration-500" style={{width: `${(startup.step / startup.total) * 100}%`}} />
              </div>
              <p className="text-xs text-muted-foreground">Server may take 90s on first cold start (PyTorch import).</p>
            </CardContent>
          </Card>
        ) : (
          <Card className="border-amber-500/35 bg-amber-500/5">
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
            <Card className="bg-gradient-to-br from-accent/5 to-transparent border-accent/20">
              <CardHeader className="pb-1">
                <CardDescription className="text-xs font-medium">Active</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-base font-semibold truncate">{modelStatus.loaded ? `${modelStatus.model} + ${currentSoul?.name || 'default'}` : 'Not loaded'}</p>
                {inferenceCount !== null && inferenceCount !== undefined && (
                  <p className="text-[10px] text-muted-foreground mt-0.5 tabular-nums">{inferenceCount} conversations</p>
                )}
              </CardContent>
            </Card>
          </div>

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
                    <span className="text-xs text-primary shrink-0">View →</span>
                  </div>
                </CardContent>
              </Card>
            </Link>
          )}
        </>
      )}

      {apiStatus === 'online' && (
        <Card>
          <CardContent className="py-3">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
              <div>
                <div className="flex justify-center mb-1">
                  <span className={`w-2 h-2 rounded-full ${health && health !== 'offline' && (health as any).status === 'healthy' ? 'bg-emerald-500' : 'bg-amber-400'}`} />
                </div>
                <p className="text-[10px] text-muted-foreground">API</p>
                <p className="text-xs font-medium">{(health as any)?.status === 'healthy' ? 'Healthy' : 'Degraded'}</p>
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground">Model</p>
                <p className="text-xs font-medium truncate">
                  {modelStatus.loaded ? (modelStatus.model?.split('/').pop() || modelStatus.model) : 'None loaded'}
                </p>
                {inferenceCount !== null && inferenceCount !== undefined && (
                  <p className="text-[10px] text-muted-foreground mt-0.5 tabular-nums">{inferenceCount} inference{inferenceCount === 1 ? '' : 's'}</p>
                )}
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground">Personality</p>
                <p className="text-xs font-medium truncate">{currentSoul?.name || 'Default'}</p>
                {currentSoul?.traits && currentSoul.traits.length > 0 && (
                  <p className="text-[10px] text-muted-foreground mt-0.5 truncate">{currentSoul.traits.slice(0, 3).join(', ')}</p>
                )}
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground">Knowledge</p>
                <p className="text-xs font-medium">{knowledgeCount} fact{knowledgeCount === 1 ? '' : 's'}</p>
                <p className="text-[10px] text-muted-foreground mt-0.5">{modelCount !== null ? `${modelCount} model${modelCount === 1 ? '' : 's'} available` : ''}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {apiStatus === 'online' && modelStatus.loaded && (
        <Card>
          <CardContent className="py-3">
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0 flex-1">
                <p className="text-xs font-medium">Quick test</p>
                <p className="text-[10px] text-muted-foreground">Send &quot;Hello!&quot; to verify the model works</p>
              </div>
              <Button
                size="sm"
                variant="outline"
                className="h-7 text-xs shrink-0"
                disabled={data.testRunning}
                onClick={async () => {
                  data.setTestRunning(true)
                  data.setTestResponse(null)
                  try {
                    const resp = await fetch(`${PUBLIC_API_URL}/chat`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ messages: [{ role: 'user', content: 'Hello!' }] }),
                    })
                    const result = await resp.json()
                    data.setTestResponse(result.message || result.error || 'No response')
                  } catch {
                    data.setTestResponse('Failed to connect')
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
      )}

      {apiStatus === 'online' && (recentSessions.length > 1 || recentJobs.length > 0) && (
        <Card>
          <CardContent className="py-3">
            <p className="text-xs font-medium mb-2">Recent activity</p>
            <div className="space-y-1.5">
              {recentSessions.slice(0, 3).map(s => (
                <button
                  key={s.id}
                  onClick={() => window.location.href = `/chat?session=${s.id}`}
                  className="w-full flex items-center gap-2 text-left hover:bg-muted/30 rounded px-1.5 py-1 transition-colors"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-primary/60 shrink-0" />
                  <span className="text-xs truncate flex-1">{s.name}</span>
                  <span className="text-[10px] text-muted-foreground shrink-0">
                    {(() => {
                      const d = Date.now() - new Date(s.updated_at).getTime()
                      const m = Math.floor(d / 60000)
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
                  <span className="text-[10px] text-muted-foreground shrink-0">{j.status}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {!apiStatus || apiStatus === 'loading' || apiStatus === 'offline' ? null : (() => {
        const dismissed = typeof window !== 'undefined' && localStorage.getItem('onboarding_dismissed')
        if (dismissed) return null
        return (
          <Card className="border-dashed border-primary/20 bg-primary/[0.02]">
            <CardContent className="py-3 flex items-center justify-between gap-3">
              <p className="text-xs text-muted-foreground">
                <span className="font-medium text-foreground">New here?</span>{' '}
                Start chatting, then train the model on your conversations. The more you chat, the better it gets.
              </p>
              <button
                className="text-[10px] text-muted-foreground hover:text-foreground shrink-0"
                onClick={(e) => { localStorage.setItem('onboarding_dismissed', '1'); e.currentTarget.closest('.space-y-4 > div')?.remove() }}
              >
                Got it
              </button>
            </CardContent>
          </Card>
        )
      })()}

      {recentSessions.length > 0 && (
        <button
          onClick={() => window.location.href = `/chat?session=${recentSessions[0].id}`}
          className="w-full text-left rounded-lg border border-border/30 bg-card p-3 flex items-center gap-3 hover:border-border/60 hover:bg-muted/30 transition-colors group"
        >
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-muted/60 text-muted-foreground group-hover:text-foreground transition-colors">
            <IconMessage className="h-4 w-4" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="text-xs font-medium truncate">{recentSessions[0].name}</p>
            <p className="text-[10px] text-muted-foreground">
              Last conversation · {new Date(recentSessions[0].updated_at).toLocaleDateString()}
            </p>
          </div>
          <IconChevronRight className="h-3.5 w-3.5 text-muted-foreground/40 group-hover:text-primary/60 transition-colors" />
        </button>
      )}

      {apiStatus === 'online' && (
        <Card>
          <CardContent className="py-3">
            <div className="flex items-center gap-2 mb-2">
              <p className="text-xs font-medium">Quick note</p>
              <p className="text-[10px] text-muted-foreground">Add a fact the AI can remember</p>
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
                className="flex-1 h-8 rounded-md border border-border/60 bg-background px-2 text-xs placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/30"
              />
              <Button size="sm" type="submit" className="h-8 text-xs shrink-0">Save</Button>
            </form>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-2 gap-2 sm:gap-3">
        <Link
          href="/chat"
          className="group relative overflow-hidden rounded-lg border border-primary/20 bg-gradient-to-br from-primary/10 via-primary/5 to-transparent p-3 sm:p-5 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/30"
        >
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="flex h-8 w-8 sm:h-10 sm:w-10 items-center justify-center rounded-lg bg-primary/15 text-primary">
              <IconChat className="h-4 w-4 sm:h-5 sm:w-5" />
            </div>
            <div className="min-w-0">
              <p className="text-xs sm:text-sm font-semibold">Start chatting</p>
              <p className="hidden sm:block text-xs text-muted-foreground mt-0.5">Ask anything, get answers instantly</p>
            </div>
          </div>
          <div className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
            <IconChevronRight className="h-4 w-4 sm:h-6 sm:w-6" />
          </div>
        </Link>
        <Link
          href="/models"
          className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/50 to-transparent p-3 sm:p-5 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20"
        >
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="flex h-8 w-8 sm:h-10 sm:w-10 items-center justify-center rounded-lg bg-accent/15 text-accent">
              <IconModels className="h-4 w-4 sm:h-5 sm:w-5" />
            </div>
            <div className="min-w-0">
              <p className="text-xs sm:text-sm font-semibold">Personalities</p>
              <p className="hidden sm:block text-xs text-muted-foreground mt-0.5">Switch your agent&apos;s personality</p>
            </div>
          </div>
          <div className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
            <IconChevronRight className="h-4 w-4 sm:h-6 sm:w-6" />
          </div>
        </Link>
        <Link
          href="/knowledge"
          className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/50 to-transparent p-3 sm:p-5 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20"
        >
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="flex h-8 w-8 sm:h-10 sm:w-10 items-center justify-center rounded-lg bg-muted-foreground/15 text-muted-foreground">
              <IconSearch className="h-4 w-4 sm:h-5 sm:w-5" />
            </div>
            <div className="min-w-0">
              <p className="text-xs sm:text-sm font-semibold">Knowledge</p>
              <p className="hidden sm:block text-xs text-muted-foreground mt-0.5">Teach your agent facts</p>
            </div>
          </div>
          <div className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
            <IconChevronRight className="h-4 w-4 sm:h-6 sm:w-6" />
          </div>
        </Link>
        <Link
          href="/training"
          className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/50 to-transparent p-3 sm:p-5 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20"
        >
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="flex h-8 w-8 sm:h-10 sm:w-10 items-center justify-center rounded-lg bg-accent/15 text-accent">
              <svg className="h-4 w-4 sm:h-5 sm:w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
            </div>
            <div className="min-w-0">
              <p className="text-xs sm:text-sm font-semibold">Teach me</p>
              <p className="hidden sm:block text-xs text-muted-foreground mt-0.5">Train from your writing</p>
            </div>
          </div>
          <div className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
            <IconChevronRight className="h-4 w-4 sm:h-6 sm:w-6" />
          </div>
        </Link>
        <Link
          href="/datasets"
          className="group relative overflow-hidden rounded-lg border border-border/60 bg-gradient-to-br from-muted/50 to-transparent p-3 sm:p-5 transition-all hover:shadow-lg hover:shadow-primary/5 hover:border-primary/20 col-span-2 sm:col-span-1"
        >
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="flex h-8 w-8 sm:h-10 sm:w-10 items-center justify-center rounded-lg bg-accent/15 text-accent">
              <svg className="h-4 w-4 sm:h-5 sm:w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"/></svg>
            </div>
            <div className="min-w-0">
              <p className="text-xs sm:text-sm font-semibold">Datasets</p>
              <p className="hidden sm:block text-xs text-muted-foreground mt-0.5">Manage training data</p>
            </div>
          </div>
          <div className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 text-muted-foreground/30 group-hover:text-primary/40 transition-colors">
            <IconChevronRight className="h-4 w-4 sm:h-6 sm:w-6" />
          </div>
        </Link>
      </div>

      {apiStatus === 'online' && (
        <div className="grid grid-cols-2 gap-2 sm:gap-3">
          <div className="rounded-lg border border-border/60 p-3 sm:p-4">
            <div className="text-xs font-medium text-muted-foreground mb-1">How it works</div>
            <p className="text-[11px] text-muted-foreground/70">
              Mix and match AI models with personalities. Each one has its own voice and style — pick the one you like.
            </p>
          </div>
          <div className="rounded-lg border border-border/60 p-3 sm:p-4">
            <div className="text-xs font-medium text-muted-foreground mb-1">Your data</div>
            <p className="text-[11px] text-muted-foreground/70">
              Import text, files, or conversations. The AI learns from your data and gets better over time.
            </p>
          </div>
        </div>
      )}

      {recentSessions.length > 0 && (
        <Card className="flex flex-col min-h-0">
          <CardHeader className="shrink-0">
            <CardTitle className="text-base">Recent conversations</CardTitle>
          </CardHeader>
          <CardContent className="p-0 min-h-0 overflow-y-auto max-h-[300px] custom-scrollbar">
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
