'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { consciousnessController } from '@/lib/consciousness-controller'
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
  CardHeader,
  CardTitle,
  Input,
  Skeleton,
  Slider,
  Switch,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
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

const ALL_PREFS_KEY = 'consciousness_all_settings'
const PERSONALITY_KEY = 'consciousness_personality'
const NOTIFICATIONS_KEY = 'consciousness_notifications_prefs'

interface AllPrefs {
  showInChat: boolean
  showInStatusBar: boolean
  realTimeUpdates: boolean
  autoReflect: boolean
  autoEvolve: boolean
  autoRefresh: boolean
  browserNotifications: boolean
  soundNotifications: boolean
  alertThresholdLow: number
  alertThresholdHigh: number
}

interface PersonalityPrefs {
  warmth: number
  confidence: number
  humor: number
  empathy: number
  openness: number
  agreeableness: number
  conscientiousness: number
  summary: string
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

const DEFAULT_PREFS: AllPrefs = {
  showInChat: true,
  showInStatusBar: true,
  realTimeUpdates: false,
  autoReflect: false,
  autoEvolve: false,
  autoRefresh: false,
  browserNotifications: false,
  soundNotifications: false,
  alertThresholdLow: 30,
  alertThresholdHigh: 80,
}

const DEFAULT_PERSONALITY: PersonalityPrefs = {
  warmth: 50,
  confidence: 50,
  humor: 50,
  empathy: 50,
  openness: 50,
  agreeableness: 50,
  conscientiousness: 50,
  summary: 'Default personality',
}

function loadAllPrefs(): AllPrefs {
  if (typeof window === 'undefined') return DEFAULT_PREFS
  try {
    const raw = localStorage.getItem(ALL_PREFS_KEY)
    if (!raw) return DEFAULT_PREFS
    return { ...DEFAULT_PREFS, ...JSON.parse(raw) }
  } catch {
    return DEFAULT_PREFS
  }
}

function saveAllPrefs(prefs: AllPrefs) {
  try {
    localStorage.setItem(ALL_PREFS_KEY, JSON.stringify(prefs))
  } catch { /* ignored */ }
}

function loadPersonality(): PersonalityPrefs {
  if (typeof window === 'undefined') return DEFAULT_PERSONALITY
  try {
    const raw = localStorage.getItem(PERSONALITY_KEY)
    if (!raw) return DEFAULT_PERSONALITY
    return { ...DEFAULT_PERSONALITY, ...JSON.parse(raw) }
  } catch {
    return DEFAULT_PERSONALITY
  }
}

function savePersonality(prefs: PersonalityPrefs) {
  try {
    localStorage.setItem(PERSONALITY_KEY, JSON.stringify(prefs))
  } catch { /* ignored */ }
}

const PRESETS: Record<string, Partial<PersonalityPrefs>> = {
  default: DEFAULT_PERSONALITY,
  formal: { warmth: 30, confidence: 70, humor: 10, empathy: 40, openness: 40, agreeableness: 60, conscientiousness: 80 },
  creative: { warmth: 60, confidence: 50, humor: 70, empathy: 50, openness: 90, agreeableness: 50, conscientiousness: 30 },
  analyst: { warmth: 20, confidence: 80, humor: 20, empathy: 30, openness: 60, agreeableness: 40, conscientiousness: 90 },
  empathetic: { warmth: 90, confidence: 40, humor: 30, empathy: 90, openness: 70, agreeableness: 80, conscientiousness: 50 },
  minimal: { warmth: 20, confidence: 60, humor: 10, empathy: 20, openness: 30, agreeableness: 40, conscientiousness: 60 },
}

export default function AllSettingsPage() {
  const addToast = useToastStore(s => s.addToast)
  const { t } = useLocale()
  const [config, setConfig] = useState<ConfigStatus | null>(null)
  const [status, setStatus] = useState<FullStatus | null>(null)
  const [prefs, setPrefs] = useState<AllPrefs>(DEFAULT_PREFS)
  const [personality, setPersonality] = useState<PersonalityPrefs>(DEFAULT_PERSONALITY)
  const [loading, setLoading] = useState(true)
  const [seedCount, setSeedCount] = useState(10)
  const [seeding, setSeeding] = useState(false)
  const [activeTab, setActiveTab] = useState('core')
  const fileInputRef = useRef<HTMLInputElement>(null)

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
    setPrefs(loadAllPrefs())
    setPersonality(loadPersonality())
    Promise.all([fetchStatus()]).finally(() => setLoading(false))
  }, [fetchStatus])

  const updatePrefs = (partial: Partial<AllPrefs>) => {
    setPrefs(prev => {
      const next = { ...prev, ...partial }
      saveAllPrefs(next)
      return next
    })
  }

  const updatePersonality = (partial: Partial<PersonalityPrefs>) => {
    setPersonality(prev => {
      const next = { ...prev, ...partial }
      savePersonality(next)
      return next
    })
  }

  const applyPreset = (presetName: string) => {
    const preset = PRESETS[presetName]
    if (preset) {
      setPersonality(prev => {
        const next = { ...prev, ...preset }
        savePersonality(next)
        return next
      })
      addToast(`Applied ${presetName} preset`, 'success')
    }
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
      setPersonality(DEFAULT_PERSONALITY)
      savePersonality(DEFAULT_PERSONALITY)
      fetchStatus()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }

  const handleExport = async () => {
    try {
      const data = {
        config,
        prefs,
        personality,
        exportedAt: new Date().toISOString(),
      }
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `consciousness-settings-${Date.now()}.json`
      a.click()
      URL.revokeObjectURL(url)
      addToast('Settings exported', 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = async (ev) => {
      try {
        const data = JSON.parse(ev.target?.result as string)
        if (data.prefs) {
          setPrefs(data.prefs)
          saveAllPrefs(data.prefs)
        }
        if (data.personality) {
          setPersonality(data.personality)
          savePersonality(data.personality)
        }
        if (data.config?.level !== undefined) {
          await updateLevel(data.config.level)
        }
        addToast('Settings imported', 'success')
      } catch {
        addToast('Invalid settings file', 'error')
      }
    }
    reader.readAsText(file)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  if (loading) {
    return (
      <PageContainer title={t('consciousness_all_settings.page_title')}>
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
    <PageContainer title={t('consciousness_all_settings.page_title')}>
      <div className="p-6">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="core">{t('consciousness_all_settings.tab_core')}</TabsTrigger>
            <TabsTrigger value="display">{t('consciousness_all_settings.tab_display')}</TabsTrigger>
            <TabsTrigger value="personality">{t('consciousness_all_settings.tab_personality')}</TabsTrigger>
            <TabsTrigger value="data">{t('consciousness_all_settings.tab_data')}</TabsTrigger>
            <TabsTrigger value="notifications">{t('consciousness_all_settings.tab_notifications')}</TabsTrigger>
            <TabsTrigger value="export">{t('consciousness_all_settings.tab_export')}</TabsTrigger>
          </TabsList>

          <TabsContent value="core" className="mt-6 space-y-6">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base">{t('consciousness_all_settings.core_title')}</CardTitle>
                    <CardDescription>{t('consciousness_all_settings.core_desc')}</CardDescription>
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
                  <label className="text-sm font-medium">{t('consciousness_all_settings.level_label')}</label>
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
                    <p className="text-sm font-medium">{t('consciousness_all_settings.enabled_label')}</p>
                    <p className="text-xs text-muted-foreground">{t('consciousness_all_settings.enabled_desc')}</p>
                  </div>
                  <Switch
                    checked={(config?.level ?? 0) > 0}
                    onCheckedChange={toggleEnabled}
                    aria-label="Toggle consciousness enabled"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{t('consciousness_all_settings.auto_reflect_label')}</p>
                    <p className="text-xs text-muted-foreground">{t('consciousness_all_settings.auto_reflect_desc')}</p>
                  </div>
                  <Switch
                    checked={prefs.autoReflect}
                    onCheckedChange={(checked) => updatePrefs({ autoReflect: checked })}
                    aria-label="Toggle auto-reflect"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{t('consciousness_all_settings.auto_evolve_label')}</p>
                    <p className="text-xs text-muted-foreground">{t('consciousness_all_settings.auto_evolve_desc')}</p>
                  </div>
                  <Switch
                    checked={prefs.autoEvolve}
                    onCheckedChange={(checked) => updatePrefs({ autoEvolve: checked })}
                    aria-label="Toggle auto-evolve"
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="display" className="mt-6 space-y-6">
            <Card>
              <CardHeader>
                <div>
                  <CardTitle className="text-base">{t('consciousness_all_settings.display_title')}</CardTitle>
                  <CardDescription>{t('consciousness_all_settings.display_desc')}</CardDescription>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{t('consciousness_all_settings.show_in_chat_label')}</p>
                    <p className="text-xs text-muted-foreground">{t('consciousness_all_settings.show_in_chat_desc')}</p>
                  </div>
                  <Switch
                    checked={prefs.showInChat}
                    onCheckedChange={(checked) => updatePrefs({ showInChat: checked })}
                    aria-label="Toggle show in chat"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{t('consciousness_all_settings.show_in_statusbar_label')}</p>
                    <p className="text-xs text-muted-foreground">{t('consciousness_all_settings.show_in_statusbar_desc')}</p>
                  </div>
                  <Switch
                    checked={prefs.showInStatusBar}
                    onCheckedChange={(checked) => updatePrefs({ showInStatusBar: checked })}
                    aria-label="Toggle show in status bar"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{t('consciousness_all_settings.realtime_label')}</p>
                    <p className="text-xs text-muted-foreground">{t('consciousness_all_settings.realtime_desc')}</p>
                  </div>
                  <Switch
                    checked={prefs.realTimeUpdates}
                    onCheckedChange={(checked) => updatePrefs({ realTimeUpdates: checked })}
                    aria-label="Toggle real-time updates"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{t('consciousness_all_settings.auto_refresh_label')}</p>
                    <p className="text-xs text-muted-foreground">{t('consciousness_all_settings.auto_refresh_desc')}</p>
                  </div>
                  <Switch
                    checked={prefs.autoRefresh}
                    onCheckedChange={(checked) => updatePrefs({ autoRefresh: checked })}
                    aria-label="Toggle auto-refresh"
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="personality" className="mt-6 space-y-6">
            <Card>
              <CardHeader>
                <div>
                  <CardTitle className="text-base">{t('consciousness_all_settings.personality_summary_title')}</CardTitle>
                  <CardDescription>{t('consciousness_all_settings.personality_summary_desc')}</CardDescription>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">{personality.summary}</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div>
                  <CardTitle className="text-base">{t('consciousness_all_settings.voice_title')}</CardTitle>
                  <CardDescription>{t('consciousness_all_settings.voice_desc')}</CardDescription>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {(['warmth', 'confidence', 'humor', 'empathy'] as const).map((key) => (
                  <div key={key} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium capitalize">{t(`consciousness_all_settings.voice_${key}`)}</label>
                      <span className="text-xs text-muted-foreground font-mono">{personality[key]}</span>
                    </div>
                    <Slider
                      value={[personality[key]]}
                      onValueChange={([v]) => updatePersonality({ [key]: v })}
                      min={0}
                      max={100}
                      step={1}
                    />
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div>
                  <CardTitle className="text-base">{t('consciousness_all_settings.traits_title')}</CardTitle>
                  <CardDescription>{t('consciousness_all_settings.traits_desc')}</CardDescription>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {(['openness', 'agreeableness', 'conscientiousness'] as const).map((key) => (
                  <div key={key} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium capitalize">{t(`consciousness_all_settings.trait_${key}`)}</label>
                      <span className="text-xs text-muted-foreground font-mono">{personality[key]}</span>
                    </div>
                    <Slider
                      value={[personality[key]]}
                      onValueChange={([v]) => updatePersonality({ [key]: v })}
                      min={0}
                      max={100}
                      step={1}
                    />
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div>
                  <CardTitle className="text-base">{t('consciousness_all_settings.presets_title')}</CardTitle>
                  <CardDescription>{t('consciousness_all_settings.presets_desc')}</CardDescription>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {Object.keys(PRESETS).map((preset) => (
                    <Button
                      key={preset}
                      size="sm"
                      variant="secondary"
                      onClick={() => applyPreset(preset)}
                    >
                      <span className="capitalize">{t(`consciousness_all_settings.preset_${preset}`)}</span>
                    </Button>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="data" className="mt-6 space-y-6">
            <Card>
              <CardHeader>
                <div>
                  <CardTitle className="text-base">{t('consciousness_all_settings.seed_title')}</CardTitle>
                  <CardDescription>{t('consciousness_all_settings.seed_desc')}</CardDescription>
                </div>
              </CardHeader>
              <CardContent>
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
                    {seeding ? t('consciousness_all_settings.seeding') : t('consciousness_all_settings.seed_button')}
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div>
                  <CardTitle className="text-base">{t('consciousness_all_settings.clear_title')}</CardTitle>
                  <CardDescription>{t('consciousness_all_settings.clear_desc')}</CardDescription>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{t('consciousness_all_settings.clear_episodes_label')}</p>
                    <p className="text-xs text-muted-foreground">{t('consciousness_all_settings.clear_episodes_desc')}</p>
                  </div>
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button type="button" variant="destructive" size="sm">{t('consciousness_all_settings.clear_button')}</Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>{t('consciousness_all_settings.clear_episodes_confirm_title')}</AlertDialogTitle>
                        <AlertDialogDescription>{t('consciousness_all_settings.clear_episodes_confirm_desc')}</AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>{t('consciousness_all_settings.cancel')}</AlertDialogCancel>
                        <AlertDialogAction onClick={handleClearEpisodes} className="bg-destructive text-destructive-foreground">{t('consciousness_all_settings.clear_button')}</AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{t('consciousness_all_settings.clear_beliefs_label')}</p>
                    <p className="text-xs text-muted-foreground">{t('consciousness_all_settings.clear_beliefs_desc')}</p>
                  </div>
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button type="button" variant="destructive" size="sm">{t('consciousness_all_settings.clear_button')}</Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>{t('consciousness_all_settings.clear_beliefs_confirm_title')}</AlertDialogTitle>
                        <AlertDialogDescription>{t('consciousness_all_settings.clear_beliefs_confirm_desc')}</AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>{t('consciousness_all_settings.cancel')}</AlertDialogCancel>
                        <AlertDialogAction onClick={handleClearBeliefs} className="bg-destructive text-destructive-foreground">{t('consciousness_all_settings.clear_button')}</AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{t('consciousness_all_settings.reset_personality_label')}</p>
                    <p className="text-xs text-muted-foreground">{t('consciousness_all_settings.reset_personality_desc')}</p>
                  </div>
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button type="button" variant="destructive" size="sm">{t('consciousness_all_settings.reset_button')}</Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>{t('consciousness_all_settings.reset_personality_confirm_title')}</AlertDialogTitle>
                        <AlertDialogDescription>{t('consciousness_all_settings.reset_personality_confirm_desc')}</AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>{t('consciousness_all_settings.cancel')}</AlertDialogCancel>
                        <AlertDialogAction onClick={handleResetPersonality} className="bg-destructive text-destructive-foreground">{t('consciousness_all_settings.reset_button')}</AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="notifications" className="mt-6 space-y-6">
            <Card>
              <CardHeader>
                <div>
                  <CardTitle className="text-base">{t('consciousness_all_settings.notifications_title')}</CardTitle>
                  <CardDescription>{t('consciousness_all_settings.notifications_desc')}</CardDescription>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{t('consciousness_all_settings.browser_notifications_label')}</p>
                    <p className="text-xs text-muted-foreground">{t('consciousness_all_settings.browser_notifications_desc')}</p>
                  </div>
                  <Switch
                    checked={prefs.browserNotifications}
                    onCheckedChange={(checked) => updatePrefs({ browserNotifications: checked })}
                    aria-label="Toggle browser notifications"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{t('consciousness_all_settings.sound_notifications_label')}</p>
                    <p className="text-xs text-muted-foreground">{t('consciousness_all_settings.sound_notifications_desc')}</p>
                  </div>
                  <Switch
                    checked={prefs.soundNotifications}
                    onCheckedChange={(checked) => updatePrefs({ soundNotifications: checked })}
                    aria-label="Toggle sound notifications"
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div>
                  <CardTitle className="text-base">{t('consciousness_all_settings.alert_thresholds_title')}</CardTitle>
                  <CardDescription>{t('consciousness_all_settings.alert_thresholds_desc')}</CardDescription>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">{t('consciousness_all_settings.threshold_low')}</label>
                    <span className="text-xs text-muted-foreground font-mono">{prefs.alertThresholdLow}</span>
                  </div>
                  <Slider
                    value={[prefs.alertThresholdLow]}
                    onValueChange={([v]) => updatePrefs({ alertThresholdLow: v })}
                    min={0}
                    max={100}
                    step={5}
                  />
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">{t('consciousness_all_settings.threshold_high')}</label>
                    <span className="text-xs text-muted-foreground font-mono">{prefs.alertThresholdHigh}</span>
                  </div>
                  <Slider
                    value={[prefs.alertThresholdHigh]}
                    onValueChange={([v]) => updatePrefs({ alertThresholdHigh: v })}
                    min={0}
                    max={100}
                    step={5}
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="export" className="mt-6 space-y-6">
            <Card>
              <CardHeader>
                <div>
                  <CardTitle className="text-base">{t('consciousness_all_settings.export_import_title')}</CardTitle>
                  <CardDescription>{t('consciousness_all_settings.export_import_desc')}</CardDescription>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{t('consciousness_all_settings.export_label')}</p>
                    <p className="text-xs text-muted-foreground">{t('consciousness_all_settings.export_desc')}</p>
                  </div>
                  <Button size="sm" variant="secondary" onClick={handleExport}>
                    {t('consciousness_all_settings.export_button')}
                  </Button>
                </div>
                <div className="border-t border-border/30" />
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{t('consciousness_all_settings.import_label')}</p>
                    <p className="text-xs text-muted-foreground">{t('consciousness_all_settings.import_desc')}</p>
                  </div>
                  <div>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".json"
                      onChange={handleImport}
                      className="hidden"
                      aria-label="Import settings file"
                    />
                    <Button size="sm" variant="secondary" onClick={() => fileInputRef.current?.click()}>
                      {t('consciousness_all_settings.import_button')}
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </PageContainer>
  )
}
