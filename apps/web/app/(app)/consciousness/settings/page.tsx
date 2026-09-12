'use client'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { consciousnessController } from '@/lib/consciousness-controller'
import { useConsciousnessBatch } from '@/hooks/useConsciousnessBatch'
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
  Button,
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
  Input,
  Skeleton,
  Slider,
  Switch,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'

const LEVEL_LABELS = ['Off', 'Basic', 'Full', 'Deep']
const LEVEL_DESCRIPTIONS = [
  'Consciousness system disabled',
  'Basic qualia and self-reflection',
  'Full self-model with belief tracking',
  'Deep reflection with narrative generation',
]

const LOCAL_STORAGE_KEY = 'consciousness_settings'

interface ConsciousnessPrefs {
  showInChat: boolean
  showInStatusBar: boolean
  realTimeUpdates: boolean
  autoReflect: boolean
  autoEvolve: boolean
}

const DEFAULT_PREFS: ConsciousnessPrefs = {
  showInChat: true,
  showInStatusBar: true,
  realTimeUpdates: false,
  autoReflect: false,
  autoEvolve: false,
}

function loadPrefs(): ConsciousnessPrefs {
  if (typeof window === 'undefined') return DEFAULT_PREFS
  try {
    const raw = localStorage.getItem(LOCAL_STORAGE_KEY)
    if (!raw) return DEFAULT_PREFS
    const parsed = JSON.parse(raw)
    return { ...DEFAULT_PREFS, ...parsed }
  } catch {
    return DEFAULT_PREFS
  }
}

function savePrefs(prefs: ConsciousnessPrefs) {
  try {
    localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(prefs))
  } catch { /* ignored */ }
}

interface ConfigStatus {
  level: number
  enabled: boolean
}

interface FullStatus {
  enabled: boolean
  level: number
  episodes: number
  beliefs: Record<string, number>
  training: {
    is_training: boolean
    total_pairs: number
  }
}

export default function ConsciousnessSettingsPage() {
  const addToast = useToastStore(s => s.addToast)
  const { t } = useLocale()
  const { executeBatch, loading: batchLoading } = useConsciousnessBatch()
  const [config, setConfig] = useState<ConfigStatus | null>(null)
  const [status, setStatus] = useState<FullStatus | null>(null)
  const [prefs, setPrefs] = useState<ConsciousnessPrefs>(DEFAULT_PREFS)
  const [loading, setLoading] = useState(true)
  const [seedCount, setSeedCount] = useState(10)
  const [seeding, setSeeding] = useState(false)

  const fetchStatus = useCallback(async () => {
    try {
      const data = await consciousnessController.getStatus()
      setStatus(data as FullStatus)
      setConfig({ level: (data as FullStatus).level ?? 0, enabled: (data as FullStatus).enabled ?? false })
    } catch (e) {
      console.error('Failed to fetch consciousness status', e)
    }
  }, [])

  useEffect(() => {
    setPrefs(loadPrefs())
    fetchStatus().finally(() => setLoading(false))
  }, [fetchStatus])

  const updatePrefs = (partial: Partial<ConsciousnessPrefs>) => {
    setPrefs(prev => {
      const next = { ...prev, ...partial }
      savePrefs(next)
      return next
    })
  }

  const updateLevel = async (newLevel: number) => {
    try {
      await consciousnessController.updateConfig({ level: newLevel })
      setConfig(prev => prev ? { ...prev, level: newLevel, enabled: newLevel > 0 } : prev)
      addToast(`Consciousness level set to ${LEVEL_LABELS[newLevel]}`, 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }

  const toggleEnabled = async (checked: boolean) => {
    const newLevel = checked ? Math.max(config?.level ?? 1, 1) : 0
    await updateLevel(newLevel)
  }

  const handleSeed = async () => {
    setSeeding(true)
    try {
      await consciousnessController.seedData({ count: seedCount })
      addToast(`Seeded ${seedCount} episodes`, 'success')
      fetchStatus()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setSeeding(false)
    }
  }

  const handleClearEpisodes = async () => {
    try {
      await consciousnessController.clearEpisodes()
      addToast('Episodes cleared', 'success')
      fetchStatus()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }

  const handleClearBeliefs = async () => {
    try {
      await consciousnessController.resetBeliefs()
      addToast('Beliefs reset to defaults', 'success')
      fetchStatus()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }

  const handleResetPersonality = async () => {
    try {
      await consciousnessController.resetPersonality()
      addToast('Personality reset to defaults', 'success')
      fetchStatus()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }

  if (loading) {
    return (
      <PageContainer title={t('consciousness_settings.page_title')}>
        <div className="space-y-6 p-6">
          {[1, 2, 3].map(i => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-5 w-40" />
                <Skeleton className="h-3 w-64" />
              </CardHeader>
              <CardContent className="space-y-3">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
              </CardContent>
            </Card>
          ))}
        </div>
      </PageContainer>
    )
  }

  return (
    <PageContainer title={t('consciousness_settings.page_title')}>
      <div className="space-y-6 p-6">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base">{t('consciousness_settings.config_title')}</CardTitle>
                <CardDescription>{t('consciousness_settings.config_desc')}</CardDescription>
              </div>
              {config && (
                <Badge variant={config.enabled ? 'default' : 'secondary'}>
                  {config.enabled ? 'Active' : 'Disabled'}
                </Badge>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">{t('consciousness_settings.level_label')}</label>
              <div className="flex items-center gap-4">
                <span className="text-sm text-muted-foreground w-12">{LEVEL_LABELS[0]}</span>
                <Slider
                  value={[config?.level ?? 0]}
                  onValueChange={([v]) => updateLevel(v)}
                  min={0}
                  max={3}
                  step={1}
                  className="flex-1"
                />
                <span className="text-sm text-muted-foreground w-12">{LEVEL_LABELS[3]}</span>
              </div>
              <p className="text-xs text-muted-foreground">
                {LEVEL_DESCRIPTIONS[config?.level ?? 0]}
              </p>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{t('consciousness_settings.enabled_label')}</p>
                <p className="text-xs text-muted-foreground">{t('consciousness_settings.enabled_desc')}</p>
              </div>
              <Switch
                checked={(config?.level ?? 0) > 0}
                onCheckedChange={toggleEnabled}
                aria-label="Toggle consciousness enabled"
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{t('consciousness_settings.auto_reflect_label')}</p>
                <p className="text-xs text-muted-foreground">{t('consciousness_settings.auto_reflect_desc')}</p>
              </div>
              <Switch
                checked={prefs.autoReflect}
                onCheckedChange={(checked) => updatePrefs({ autoReflect: checked })}
                aria-label="Toggle auto-reflect"
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{t('consciousness_settings.auto_evolve_label')}</p>
                <p className="text-xs text-muted-foreground">{t('consciousness_settings.auto_evolve_desc')}</p>
              </div>
              <Switch
                checked={prefs.autoEvolve}
                onCheckedChange={(checked) => updatePrefs({ autoEvolve: checked })}
                aria-label="Toggle auto-evolve"
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div>
              <CardTitle className="text-base">{t('consciousness_settings.data_title')}</CardTitle>
              <CardDescription>{t('consciousness_settings.data_desc')}</CardDescription>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{t('consciousness_settings.seed_label')}</p>
                <p className="text-xs text-muted-foreground">{t('consciousness_settings.seed_desc')}</p>
              </div>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  min={1}
                  max={100}
                  value={seedCount}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSeedCount(Math.max(1, Math.min(100, parseInt(e.target.value) || 10)))}
                  className="w-20 h-8 text-xs text-right"
                  aria-label="Seed count"
                />
                <Button size="sm" variant="secondary" onClick={handleSeed} disabled={seeding}>
                  {seeding ? t('consciousness_settings.seeding') : t('consciousness_settings.seed_button')}
                </Button>
              </div>
            </div>
            <div className="border-t border-border/30" />
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{t('consciousness_settings.clear_episodes_label')}</p>
                <p className="text-xs text-muted-foreground">{t('consciousness_settings.clear_episodes_desc')}</p>
              </div>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button type="button" variant="destructive" size="sm">{t('consciousness_settings.clear_button')}</Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>{t('consciousness_settings.clear_episodes_confirm_title')}</AlertDialogTitle>
                    <AlertDialogDescription>{t('consciousness_settings.clear_episodes_confirm_desc')}</AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>{t('consciousness_settings.cancel')}</AlertDialogCancel>
                    <AlertDialogAction onClick={handleClearEpisodes} className="bg-destructive text-destructive-foreground">{t('consciousness_settings.clear_button')}</AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{t('consciousness_settings.clear_beliefs_label')}</p>
                <p className="text-xs text-muted-foreground">{t('consciousness_settings.clear_beliefs_desc')}</p>
              </div>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button type="button" variant="destructive" size="sm">{t('consciousness_settings.clear_button')}</Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>{t('consciousness_settings.clear_beliefs_confirm_title')}</AlertDialogTitle>
                    <AlertDialogDescription>{t('consciousness_settings.clear_beliefs_confirm_desc')}</AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>{t('consciousness_settings.cancel')}</AlertDialogCancel>
                    <AlertDialogAction onClick={handleClearBeliefs} className="bg-destructive text-destructive-foreground">{t('consciousness_settings.clear_button')}</AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{t('consciousness_settings.reset_personality_label')}</p>
                <p className="text-xs text-muted-foreground">{t('consciousness_settings.reset_personality_desc')}</p>
              </div>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button type="button" variant="destructive" size="sm">{t('consciousness_settings.reset_button')}</Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>{t('consciousness_settings.reset_personality_confirm_title')}</AlertDialogTitle>
                    <AlertDialogDescription>{t('consciousness_settings.reset_personality_confirm_desc')}</AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>{t('consciousness_settings.cancel')}</AlertDialogCancel>
                    <AlertDialogAction onClick={handleResetPersonality} className="bg-destructive text-destructive-foreground">{t('consciousness_settings.reset_button')}</AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div>
              <CardTitle className="text-base">{t('consciousness_settings.batch_title')}</CardTitle>
              <CardDescription>{t('consciousness_settings.batch_desc')}</CardDescription>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={async () => {
                  await executeBatch([
                    { type: 'seed', payload: { count: seedCount } },
                    { type: 'reflect', payload: {} },
                  ])
                  addToast('Batch: seeded and reflected', 'success')
                  fetchStatus()
                }}
                disabled={batchLoading}
              >
                {batchLoading ? '...' : t('consciousness_settings.batch_seed_reflect')}
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={async () => {
                  await executeBatch([
                    { type: 'config', payload: { clear_episodes: true } },
                    { type: 'config', payload: { reset_beliefs: true } },
                  ])
                  addToast('Batch: cleared episodes and beliefs', 'success')
                  fetchStatus()
                }}
                disabled={batchLoading}
              >
                {t('consciousness_settings.batch_clear_all')}
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div>
              <CardTitle className="text-base">{t('consciousness_settings.display_title')}</CardTitle>
              <CardDescription>{t('consciousness_settings.display_desc')}</CardDescription>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{t('consciousness_settings.show_in_chat_label')}</p>
                <p className="text-xs text-muted-foreground">{t('consciousness_settings.show_in_chat_desc')}</p>
              </div>
              <Switch
                checked={prefs.showInChat}
                onCheckedChange={(checked) => updatePrefs({ showInChat: checked })}
                aria-label="Toggle show in chat"
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{t('consciousness_settings.show_in_statusbar_label')}</p>
                <p className="text-xs text-muted-foreground">{t('consciousness_settings.show_in_statusbar_desc')}</p>
              </div>
              <Switch
                checked={prefs.showInStatusBar}
                onCheckedChange={(checked) => updatePrefs({ showInStatusBar: checked })}
                aria-label="Toggle show in status bar"
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{t('consciousness_settings.realtime_label')}</p>
                <p className="text-xs text-muted-foreground">{t('consciousness_settings.realtime_desc')}</p>
              </div>
              <Switch
                checked={prefs.realTimeUpdates}
                onCheckedChange={(checked) => updatePrefs({ realTimeUpdates: checked })}
                aria-label="Toggle real-time updates"
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div>
              <CardTitle className="text-base">{t('consciousness_settings.about_title')}</CardTitle>
              <CardDescription>{t('consciousness_settings.about_desc')}</CardDescription>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">{t('consciousness_settings.version')}</p>
                <p className="text-sm font-medium mt-0.5 font-mono">3.0.0</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">{t('consciousness_settings.status')}</p>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <span className={`inline-block w-2 h-2 rounded-full ${status?.enabled ? 'bg-success' : 'bg-muted-foreground/50'}`} />
                  <span className="text-sm font-medium">{status?.enabled ? 'Active' : 'Disabled'}</span>
                </div>
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">{t('consciousness_settings.episodes')}</p>
                <p className="text-sm font-medium mt-0.5 font-mono">{status?.episodes ?? 0}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">{t('consciousness_settings.beliefs_count')}</p>
                <p className="text-sm font-medium mt-0.5 font-mono">{Object.keys(status?.beliefs ?? {}).length}</p>
              </div>
            </div>
            <div className="border-t border-border/30 pt-3">
              <a
                href="https://docs.sloughgpt.ai/consciousness"
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-primary hover:underline"
              >
                {t('consciousness_settings.docs_link')}
              </a>
            </div>
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
