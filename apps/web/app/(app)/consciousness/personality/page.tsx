'use client'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { consciousnessController } from '@/lib/consciousness-controller'
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
  AlertDialogTrigger, Badge, Button, Card, CardContent, CardDescription,
  CardHeader, CardTitle, Skeleton, Slider,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'

const TRAIT_LABELS: Record<string, string> = {
  openness: 'Openness',
  conscientiousness: 'Conscientiousness',
  extraversion: 'Extraversion',
  agreeableness: 'Agreeableness',
  neuroticism: 'Neuroticism',
  empathy: 'Empathy',
  humor: 'Humor',
  creativity: 'Creativity',
}

export default function ConsciousnessPersonalityPage() {
  const addToast = useToastStore(s => s.addToast)
  const { t } = useLocale()
  const [loading, setLoading] = useState(true)
  const [personality, setPersonality] = useState<{ name: string; description: string; traits: Record<string, number> } | null>(null)
  const [editTraits, setEditTraits] = useState<Record<string, number>>({})
  const [isEditing, setIsEditing] = useState(false)
  const [saving, setSaving] = useState(false)
  const [resetting, setResetting] = useState(false)
  const [presets, setPresets] = useState<Record<string, unknown>[]>([])
  const [applyingPreset, setApplyingPreset] = useState<string | null>(null)
  const [conflicts, setConflicts] = useState<string[]>([])
  const [history, setHistory] = useState<{ name: string; description: string; traits: Record<string, number> }[]>([])

  const fetchAll = useCallback(async () => {
    try {
      const [pResult, prResult, cResult, hResult] = await Promise.allSettled([
        consciousnessController.getPersonality(),
        consciousnessController.getPersonalityPresets(),
        consciousnessController.getPersonalityConflicts(),
        consciousnessController.getPersonalityHistory(),
      ])
      if (pResult.status === 'fulfilled') {
        const p = pResult.value as any
        setPersonality(p)
        setEditTraits(p.traits ?? {})
      }
      if (prResult.status === 'fulfilled') {
        setPresets((prResult.value as any).presets ?? [])
      }
      if (cResult.status === 'fulfilled') {
        setConflicts((cResult.value as any).conflicts ?? [])
      }
      if (hResult.status === 'fulfilled') {
        setHistory((hResult.value as any).history ?? [])
      }
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => { fetchAll() }, [fetchAll])

  const handleSave = async () => {
    setSaving(true)
    try {
      await consciousnessController.updatePersonality(editTraits)
      addToast('Personality traits updated', 'success')
      setIsEditing(false)
      await fetchAll()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setSaving(false)
    }
  }

  const handleReset = async () => {
    setResetting(true)
    try {
      await consciousnessController.resetPersonality()
      addToast('Personality reset to defaults', 'success')
      setIsEditing(false)
      await fetchAll()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setResetting(false)
    }
  }

  const handleApplyPreset = async (presetName: string) => {
    setApplyingPreset(presetName)
    try {
      await consciousnessController.applyPersonalityPreset(presetName)
      addToast(`Preset "${presetName}" applied`, 'success')
      await fetchAll()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setApplyingPreset(null)
    }
  }

  const handleTraitChange = (trait: string, value: number) => {
    setEditTraits(prev => ({ ...prev, [trait]: value }))
  }

  if (loading) {
    return (
      <PageContainer title={t('consciousness_personality.page_title')}>
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

  const traits = isEditing ? editTraits : (personality?.traits ?? {})
  const traitKeys = Object.keys(traits)

  return (
    <PageContainer title={t('consciousness_personality.page_title')}>
      <div className="space-y-6 p-6">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base">{t('consciousness_personality.overview_title')}</CardTitle>
                <CardDescription>{t('consciousness_personality.overview_desc')}</CardDescription>
              </div>
              {personality?.name && (
                <Badge variant="outline">{personality.name}</Badge>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {personality?.description && (
              <p className="text-sm text-muted-foreground">{personality.description}</p>
            )}
            <div className="space-y-3">
              {traitKeys.map(trait => (
                <div key={trait} className="space-y-1">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">{TRAIT_LABELS[trait] ?? trait}</label>
                    <span className="text-xs text-muted-foreground">{(traits[trait] * 100).toFixed(0)}%</span>
                  </div>
                  {isEditing ? (
                    <Slider
                      value={[traits[trait]]}
                      onValueChange={([v]) => handleTraitChange(trait, v)}
                      min={0}
                      max={1}
                      step={0.01}
                    />
                  ) : (
                    <div className="h-2 rounded-full bg-muted overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500 bg-primary"
                        style={{ width: `${traits[trait] * 100}%` }}
                      />
                    </div>
                  )}
                </div>
              ))}
            </div>
            <div className="flex gap-2 pt-2">
              {isEditing ? (
                <>
                  <Button size="sm" onClick={handleSave} disabled={saving}>
                    {saving ? 'Saving...' : t('common.save')}
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => { setIsEditing(false); setEditTraits(personality?.traits ?? {}) }}>
                    {t('consciousness_settings.cancel')}
                  </Button>
                </>
              ) : (
                <Button size="sm" variant="outline" onClick={() => setIsEditing(true)}>
                  {t('consciousness_personality.edit_traits')}
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {presets.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">{t('consciousness_personality.presets_title')}</CardTitle>
              <CardDescription>{t('consciousness_personality.presets_desc')}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {presets.map((preset: any) => (
                  <Button
                    key={preset.name}
                    variant={personality?.name === preset.name ? 'default' : 'outline'}
                    className="h-auto flex-col items-start p-3 text-left"
                    onClick={() => handleApplyPreset(preset.name)}
                    disabled={applyingPreset === preset.name}
                  >
                    <span className="text-sm font-medium">{preset.name}</span>
                    {preset.description && (
                      <span className="text-xs text-muted-foreground mt-1 line-clamp-2">{preset.description}</span>
                    )}
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {conflicts.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">{t('consciousness_personality.conflicts_title')}</CardTitle>
              <CardDescription>{t('consciousness_personality.conflicts_desc')}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {conflicts.map((conflict, i) => (
                  <div key={i} className="flex items-start gap-2 text-sm">
                    <Badge variant="destructive" className="shrink-0">!</Badge>
                    <span>{conflict}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {history.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">{t('consciousness_personality.history_title')}</CardTitle>
              <CardDescription>{t('consciousness_personality.history_desc')}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 max-h-[400px] overflow-y-auto">
                {history.map((entry, i) => (
                  <div key={i} className="rounded-lg border p-3 text-sm">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">{entry.name}</span>
                      <span className="text-xs text-muted-foreground">#{history.length - i}</span>
                    </div>
                    {entry.description && (
                      <p className="text-xs text-muted-foreground mb-2">{entry.description}</p>
                    )}
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(entry.traits).map(([k, v]) => (
                        <span key={k} className="text-[10px] text-muted-foreground">
                          {TRAIT_LABELS[k] ?? k}: {(v as number * 100).toFixed(0)}%
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('consciousness_personality.actions_title')}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              <Button size="sm" variant="outline" onClick={fetchAll}>
                {t('consciousness_personality.refresh')}
              </Button>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button type="button" size="sm" variant="destructive">
                    {t('consciousness_personality.reset_defaults')}
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>{t('consciousness_personality.reset_confirm_title')}</AlertDialogTitle>
                    <AlertDialogDescription>{t('consciousness_personality.reset_confirm_desc')}</AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>{t('consciousness_settings.cancel')}</AlertDialogCancel>
                    <AlertDialogAction onClick={handleReset} className="bg-destructive text-destructive-foreground">
                      {t('consciousness_personality.reset_defaults')}
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
          </CardContent>
        </Card>
      </div>
    </PageContainer>
  )
}
