'use client'

import { useState, useEffect, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { PageContainer } from '@/components/PageContainer'
import { consciousnessController } from '@/lib/consciousness-controller'
import {
  Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'

const STORAGE_KEY = 'consciousness_quickstart_progress'

interface Progress {
  enabled: boolean
  levelChosen: boolean
  personalitySet: boolean
  seeded: boolean
  started: boolean
}

const DEFAULT_PROGRESS: Progress = {
  enabled: false,
  levelChosen: false,
  personalitySet: false,
  seeded: false,
  started: false,
}

function loadProgress(): Progress {
  if (typeof window === 'undefined') return DEFAULT_PROGRESS
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return DEFAULT_PROGRESS
    return { ...DEFAULT_PROGRESS, ...JSON.parse(raw) }
  } catch {
    return DEFAULT_PROGRESS
  }
}

function saveProgress(p: Progress) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(p))
  } catch { /* ignored */ }
}

const LEVEL_LABELS = ['Off', 'Basic', 'Full', 'Deep']

const LINKS = [
  { path: '/consciousness/dashboard', labelKey: 'nav.consciousness_dashboard' },
  { path: '/consciousness/training', labelKey: 'nav.consciousness_training' },
  { path: '/consciousness/playground', labelKey: 'nav.consciousness_playground' },
  { path: '/consciousness/analytics', labelKey: 'nav.consciousness_analytics' },
  { path: '/consciousness/statistics', labelKey: 'nav.consciousness_statistics' },
  { path: '/consciousness/history', labelKey: 'nav.consciousness_history' },
  { path: '/consciousness/settings', labelKey: 'nav.consciousness_settings' },
  { path: '/consciousness/testing', labelKey: 'nav.consciousness_testing' },
  { path: '/consciousness/docs', labelKey: 'nav.consciousness_docs' },
  { path: '/consciousness/compare', labelKey: 'nav.consciousness_compare' },
]

export default function ConsciousnessQuickstartPage() {
  const router = useRouter()
  const addToast = useToastStore(s => s.addToast)
  const { t } = useLocale()
  const [progress, setProgress] = useState<Progress>(DEFAULT_PROGRESS)
  const [enabling, setEnabling] = useState(false)
  const [seeding, setSeeding] = useState(false)
  const [applying, setApplying] = useState(false)

  useEffect(() => {
    setProgress(loadProgress())
  }, [])

  const updateProgress = useCallback((partial: Partial<Progress>) => {
    setProgress(prev => {
      const next = { ...prev, ...partial }
      saveProgress(next)
      return next
    })
  }, [])

  const completed = Object.values(progress).filter(Boolean).length
  const total = Object.keys(progress).length
  const pct = Math.round((completed / total) * 100)

  const handleEnableBasic = async () => {
    setEnabling(true)
    try {
      await consciousnessController.updateConfig({ level: 1 })
      updateProgress({ enabled: true, levelChosen: true })
      addToast('Consciousness enabled at Basic level', 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setEnabling(false)
    }
  }

  const handleApplyPersonality = async () => {
    setApplying(true)
    try {
      await consciousnessController.applyPersonalityPreset('default')
      updateProgress({ personalitySet: true })
      addToast('Default personality applied', 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setApplying(false)
    }
  }

  const handleSeed10 = async () => {
    setSeeding(true)
    try {
      await consciousnessController.seedData({ count: 10 })
      updateProgress({ seeded: true })
      addToast('Seeded 10 episodes', 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setSeeding(false)
    }
  }

  const handleOpenChat = () => {
    updateProgress({ started: true })
    router.push('/chat')
  }

  const steps = [
    {
      num: 1,
      titleKey: 'consciousness_quickstart.step1_title',
      descKey: 'consciousness_quickstart.step1_desc',
      actionKey: 'consciousness_quickstart.step1_action',
      done: progress.enabled,
      onClick: () => router.push('/consciousness/settings'),
    },
    {
      num: 2,
      titleKey: 'consciousness_quickstart.step2_title',
      descKey: 'consciousness_quickstart.step2_desc',
      actionKey: 'consciousness_quickstart.step2_action',
      done: progress.levelChosen,
      onClick: () => router.push('/consciousness/settings'),
    },
    {
      num: 3,
      titleKey: 'consciousness_quickstart.step3_title',
      descKey: 'consciousness_quickstart.step3_desc',
      actionKey: 'consciousness_quickstart.step3_action',
      done: progress.personalitySet,
      onClick: () => router.push('/consciousness/settings'),
    },
    {
      num: 4,
      titleKey: 'consciousness_quickstart.step4_title',
      descKey: 'consciousness_quickstart.step4_desc',
      actionKey: 'consciousness_quickstart.step4_action',
      done: progress.seeded,
      onClick: handleSeed10,
      loading: seeding,
    },
    {
      num: 5,
      titleKey: 'consciousness_quickstart.step5_title',
      descKey: 'consciousness_quickstart.step5_desc',
      actionKey: 'consciousness_quickstart.step5_action',
      done: progress.started,
      onClick: handleOpenChat,
    },
  ]

  return (
    <PageContainer title={t('consciousness_quickstart.page_title')}>
      <div className="space-y-6 p-6">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base">{t('consciousness_quickstart.welcome_title')}</CardTitle>
                <CardDescription>{t('consciousness_quickstart.welcome_desc')}</CardDescription>
              </div>
              <Badge variant="outline">{t('consciousness_quickstart.time_estimate')}</Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-1 gap-3 md:grid-cols-3 text-sm">
              <div className="rounded-md bg-muted p-3">
                <p className="font-medium mb-1">{t('consciousness_quickstart.benefit1_title')}</p>
                <p className="text-xs text-muted-foreground">{t('consciousness_quickstart.benefit1_desc')}</p>
              </div>
              <div className="rounded-md bg-muted p-3">
                <p className="font-medium mb-1">{t('consciousness_quickstart.benefit2_title')}</p>
                <p className="text-xs text-muted-foreground">{t('consciousness_quickstart.benefit2_desc')}</p>
              </div>
              <div className="rounded-md bg-muted p-3">
                <p className="font-medium mb-1">{t('consciousness_quickstart.benefit3_title')}</p>
                <p className="text-xs text-muted-foreground">{t('consciousness_quickstart.benefit3_desc')}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('consciousness_quickstart.steps_title')}</CardTitle>
            <CardDescription>{t('consciousness_quickstart.steps_desc')}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {steps.map((step) => (
              <div
                key={step.num}
                className="flex items-start gap-3 rounded-md border p-3"
              >
                <div className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-bold ${step.done ? 'bg-success text-white' : 'bg-muted text-muted-foreground'}`}>
                  {step.done ? '✓' : step.num}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium">{t(step.titleKey)}</p>
                  <p className="text-xs text-muted-foreground">{t(step.descKey)}</p>
                </div>
                <Button
                  size="sm"
                  variant={step.done ? 'secondary' : 'default'}
                  onClick={step.onClick}
                  disabled={(step as any).loading}
                >
                  {(step as any).loading ? '...' : step.done ? t('consciousness_quickstart.done') : t(step.actionKey)}
                </Button>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('consciousness_quickstart.quick_actions_title')}</CardTitle>
            <CardDescription>{t('consciousness_quickstart.quick_actions_desc')}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
              <Button variant="outline" onClick={handleEnableBasic} disabled={enabling} className="h-auto flex-col items-start p-3">
                <span className="text-sm font-medium">{t('consciousness_quickstart.action_enable')}</span>
                <span className="text-[10px] text-muted-foreground">PATCH /consciousness/config</span>
              </Button>
              <Button variant="outline" onClick={handleApplyPersonality} disabled={applying} className="h-auto flex-col items-start p-3">
                <span className="text-sm font-medium">{t('consciousness_quickstart.action_personality')}</span>
                <span className="text-[10px] text-muted-foreground">POST /consciousness/personality/presets/apply</span>
              </Button>
              <Button variant="outline" onClick={handleSeed10} disabled={seeding} className="h-auto flex-col items-start p-3">
                <span className="text-sm font-medium">{seeding ? t('consciousness_quickstart.seeding') : t('consciousness_quickstart.action_seed')}</span>
                <span className="text-[10px] text-muted-foreground">POST /consciousness/seed?count=10</span>
              </Button>
              <Button variant="outline" onClick={handleOpenChat} className="h-auto flex-col items-start p-3">
                <span className="text-sm font-medium">{t('consciousness_quickstart.action_chat')}</span>
                <span className="text-[10px] text-muted-foreground">/chat</span>
              </Button>
              <Button variant="outline" onClick={() => router.push('/consciousness/dashboard')} className="h-auto flex-col items-start p-3">
                <span className="text-sm font-medium">{t('consciousness_quickstart.action_dashboard')}</span>
                <span className="text-[10px] text-muted-foreground">/consciousness/dashboard</span>
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('consciousness_quickstart.progress_title')}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                <div className="h-full rounded-full bg-primary transition-all duration-500" style={{ width: `${pct}%` }} />
              </div>
              <span className="text-sm font-mono tabular-nums">{pct}%</span>
            </div>
            <p className="text-xs text-muted-foreground">{completed}/{total} {t('consciousness_quickstart.steps_completed')}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('consciousness_quickstart.whats_next_title')}</CardTitle>
            <CardDescription>{t('consciousness_quickstart.whats_next_desc')}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
              {LINKS.map((link) => (
                <button
                  key={link.path}
                  onClick={() => router.push(link.path)}
                  className="rounded-md border p-3 text-left text-sm hover:bg-muted transition-colors"
                >
                  <p className="font-medium">{t(link.labelKey)}</p>
                  <p className="text-[10px] text-muted-foreground font-mono">{link.path}</p>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
