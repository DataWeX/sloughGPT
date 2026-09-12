'use client'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { consciousnessController } from '@/lib/consciousness-controller'
import { apiPatch } from '@/lib/http-client'
import {
  Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, Input, Skeleton,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'
import { PersonalityQuiz } from './PersonalityQuiz'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts'

interface SavedPersona {
  id: string
  name: string
  values: string[]
}

interface PersonalityConflict {
  type: string
  severity: string
  message: string
  fields: string[]
}

interface PersonalityProfile {
  values: string[]
  goals: string[]
  voice: Record<string, number>
  style: Record<string, boolean>
  traits: Record<string, number>
  interests: string[]
  avoid: string[]
}

interface PersonalityHistory {
  timestamp: number
  voice: Record<string, number>
  traits: Record<string, number>
}

const VOICE_LABELS: Record<string, string> = {
  formality: 'Formality',
  warmth: 'Warmth',
  confidence: 'Confidence',
  humor: 'Humor',
  verbosity: 'Verbosity',
  empathy: 'Empathy',
}

const TRAIT_LABELS: Record<string, string> = {
  openness: 'Openness',
  conscientiousness: 'Conscientiousness',
  extraversion: 'Extraversion',
  agreeableness: 'Agreeableness',
  neuroticism: 'Emotional Stability',
}

const STYLE_LABELS: Record<string, string> = {
  use_examples: 'Use Examples',
  ask_follow_ups: 'Ask Follow-ups',
  acknowledge_uncertainty: 'Acknowledge Uncertainty',
  use_analogies: 'Use Analogies',
  break_down_complex_topics: 'Break Down Complex Topics',
}

export default function PersonalityPage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [profile, setProfile] = useState<PersonalityProfile | null>(null)
  const [valuesInput, setValuesInput] = useState('')
  const [goalsInput, setGoalsInput] = useState('')
  const [interestsInput, setInterestsInput] = useState('')
  const [avoidInput, setAvoidInput] = useState('')
  const [presetNames, setPresetNames] = useState<string[]>([])
  const [activePreset, setActivePreset] = useState<string | null>(null)
  const [personalityHistory, setPersonalityHistory] = useState<PersonalityHistory[]>([])
  const [applyingPreset, setApplyingPreset] = useState<string | null>(null)
  const [conflicts, setConflicts] = useState<PersonalityConflict[]>([])
  const [showComparison, setShowComparison] = useState(false)
  const [originalProfile, setOriginalProfile] = useState<PersonalityProfile | null>(null)
  const [savedPersonas, setSavedPersonas] = useState<SavedPersona[]>([])
  const [personaNameInput, setPersonaNameInput] = useState('')
  const [savingPersona, setSavingPersona] = useState(false)
  const [activatingPersona, setActivatingPersona] = useState<string | null>(null)
  const [deletingPersona, setDeletingPersona] = useState<string | null>(null)
  const [showSavePersonaDialog, setShowSavePersonaDialog] = useState(false)

  const fetchPersonas = useCallback(async () => {
    try {
      const data = await consciousnessController.listPersonas() as any
      setSavedPersonas(data.personas ?? [])
    } catch {}
  }, [])

  const fetchProfile = useCallback(async () => {
    try {
      const [profileResult, presetsResult, histResult, conflictsResult] = await Promise.allSettled([
        consciousnessController.getPersonality(),
        consciousnessController.getPersonalityPresets(),
        consciousnessController.getPersonalityHistory(),
        consciousnessController.getPersonalityConflicts(),
      ])
      if (profileResult.status === 'fulfilled') {
        const d = profileResult.value as unknown as PersonalityProfile
        setProfile(d)
        setValuesInput(d.values.join(', '))
        setGoalsInput(d.goals.join(', '))
        setInterestsInput(d.interests.join(', '))
        setAvoidInput(d.avoid.join(', '))
      }
      if (presetsResult.status === 'fulfilled') {
        setPresetNames((presetsResult.value as any)?.names ?? [])
      }
      if (histResult.status === 'fulfilled') {
        setPersonalityHistory((histResult.value as any)?.history ?? [])
      }
      if (conflictsResult.status === 'fulfilled') {
        setConflicts((conflictsResult.value as any)?.conflicts ?? [])
      }
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setLoading(false)
    }
    fetchPersonas()
  }, [addToast, fetchPersonas])

  useEffect(() => { fetchProfile() }, [fetchProfile])

  const handleSave = async () => {
    if (!profile) return
    setSaving(true)
    try {
      const data = await apiPatch('/consciousness/personality', {
        values: valuesInput.split(',').map(s => s.trim()).filter(Boolean),
        goals: goalsInput.split(',').map(s => s.trim()).filter(Boolean),
        interests: interestsInput.split(',').map(s => s.trim()).filter(Boolean),
        avoid: avoidInput.split(',').map(s => s.trim()).filter(Boolean),
        voice: profile.voice,
        traits: profile.traits,
        style: profile.style,
      })
      setProfile(data as PersonalityProfile)
      addToast('Personality saved', 'success')
      const conflictsData = await consciousnessController.getPersonalityConflicts()
      setConflicts((conflictsData as any)?.conflicts ?? [])
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setSaving(false)
    }
  }

  const handleExport = () => {
    if (!profile) return
    const data = { ...profile, exportedAt: new Date().toISOString() }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `personality-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
    addToast('Personality exported', 'success')
  }

  const handleImport = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return
      try {
        const text = await file.text()
        const data = JSON.parse(text)
        setOriginalProfile(profile)
        setProfile(data)
        setValuesInput(data.values?.join(', ') ?? '')
        setGoalsInput(data.goals?.join(', ') ?? '')
        setInterestsInput(data.interests?.join(', ') ?? '')
        setAvoidInput(data.avoid?.join(', ') ?? '')
        setShowComparison(true)
        addToast('Personality imported — review and save', 'success')
      } catch {
        addToast('Invalid JSON file', 'error')
      }
    }
    input.click()
  }

  const handleVoiceChange = (key: string, value: number) => {
    if (!profile) return
    setProfile({ ...profile, voice: { ...profile.voice, [key]: value } })
  }

  const handleTraitChange = (key: string, value: number) => {
    if (!profile) return
    setProfile({ ...profile, traits: { ...profile.traits, [key]: value } })
  }

  const handleStyleToggle = (key: string) => {
    if (!profile) return
    setProfile({ ...profile, style: { ...profile.style, [key]: !profile.style[key] } })
  }

  const handleReset = async () => {
    try {
      await consciousnessController.resetPersonality()
      fetchProfile()
      addToast('Personality reset to defaults', 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }

  const handleApplyPreset = async (presetName: string) => {
    setApplyingPreset(presetName)
    try {
      await consciousnessController.applyPersonalityPreset(presetName)
      fetchProfile()
      setActivePreset(presetName)
      addToast(`Applied "${presetName}" preset`, 'success')
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setApplyingPreset(null)
    }
  }

  const PRESET_DESCRIPTIONS: Record<string, string> = {
    default: 'Balanced settings',
    formal: 'Professional, precise',
    creative: 'Imaginative, playful',
    analyst: 'Logical, evidence-based',
    empathetic: 'Warm, supportive',
    minimal: 'Terse, direct',
  }

  const handleSavePersona = async () => {
    if (!personaNameInput.trim() || !profile) return
    setSavingPersona(true)
    try {
      const slug = personaNameInput.trim().toLowerCase().replace(/\s+/g, '-')
      await consciousnessController.savePersona({
        persona_id: slug,
        name: personaNameInput.trim(),
        profile,
      } as any)
      addToast('Persona saved', 'success')
      setShowSavePersonaDialog(false)
      setPersonaNameInput('')
      fetchPersonas()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setSavingPersona(false)
    }
  }

  const handleActivatePersona = async (id: string) => {
    setActivatingPersona(id)
    try {
      await consciousnessController.activatePersona(id)
      addToast('Persona activated', 'success')
      fetchProfile()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setActivatingPersona(null)
    }
  }

  const handleDeletePersona = async (id: string) => {
    setDeletingPersona(id)
    try {
      await consciousnessController.deletePersona(id)
      addToast('Persona deleted', 'success')
      fetchPersonas()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    } finally {
      setDeletingPersona(null)
    }
  }

  if (loading) {
    return (
      <PageContainer title="Personality">
        <div className="space-y-6 p-6">
          <Skeleton className="h-32" />
          <Skeleton className="h-48" />
          <Skeleton className="h-48" />
        </div>
      </PageContainer>
    )
  }

  if (!profile) return null

  return (
    <PageContainer title="Personality">
      <div className="space-y-4 sm:space-y-6 p-4 sm:p-6">
        {/* Header with save/reset */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h2 className="text-base sm:text-lg font-semibold">Personality Configuration</h2>
            <p className="text-xs sm:text-sm text-muted-foreground">Define how the consciousness system thinks, speaks, and behaves</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={handleExport}>Export</Button>
            <Button variant="outline" onClick={handleImport}>Import</Button>
            <Button variant="outline" onClick={handleReset}>Reset</Button>
            <Button onClick={handleSave} disabled={saving}>
              {saving ? 'Saving...' : 'Save Changes'}
            </Button>
          </div>
        </div>

        {/* Saved Personas */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>{t('personality.savedPersonas')}</CardTitle>
              <CardDescription>Manage saved personality profiles</CardDescription>
            </div>
            <Button size="sm" onClick={() => setShowSavePersonaDialog(true)} disabled={!profile}>
              {t('personality.saveCurrent')}
            </Button>
          </CardHeader>
          <CardContent>
            {savedPersonas.length === 0 ? (
              <p className="text-sm text-muted-foreground">{t('personality.noPersonas')}</p>
            ) : (
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {savedPersonas.map((persona) => (
                  <div
                    key={persona.id}
                    className="rounded-lg border p-3 sm:p-4 space-y-2"
                  >
                    <div className="text-sm sm:text-base font-medium">{persona.name}</div>
                    <div className="text-[10px] sm:text-xs text-muted-foreground line-clamp-2">
                      {persona.values?.join(', ') || '—'}
                    </div>
                    <div className="flex gap-2 pt-1">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleActivatePersona(persona.id)}
                        disabled={activatingPersona === persona.id}
                      >
                        {activatingPersona === persona.id ? '...' : t('personality.activate')}
                      </Button>
                      <Button
                        size="sm"
                        variant="destructive"
                        onClick={() => {
                          if (window.confirm(`Delete persona "${persona.name}"?`)) {
                            handleDeletePersona(persona.id)
                          }
                        }}
                        disabled={deletingPersona === persona.id}
                      >
                        {deletingPersona === persona.id ? '...' : t('personality.delete')}
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Save Persona Dialog */}
        {showSavePersonaDialog && (
          <Card className="border-primary/50">
            <CardHeader>
              <CardTitle>{t('personality.savePersona')}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 sm:space-y-3">
              <div>
                <label className="text-xs sm:text-sm font-medium">{t('personality.personaName')}</label>
                <Input
                  value={personaNameInput}
                  onChange={(e) => setPersonaNameInput(e.target.value)}
                  placeholder="My Persona"
                  className="mt-1"
                />
              </div>
              <div className="flex gap-2">
                <Button size="sm" onClick={handleSavePersona} disabled={savingPersona || !personaNameInput.trim()}>
                  {savingPersona ? '...' : t('personality.savePersona')}
                </Button>
                <Button size="sm" variant="outline" onClick={() => { setShowSavePersonaDialog(false); setPersonaNameInput('') }}>
                  Cancel
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Personality Quiz */}
        <PersonalityQuiz
          onApply={(preset) => {
            handleApplyPreset(preset).then(() => fetchProfile())
          }}
        />

        {/* Presets */}
        {presetNames.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Quick Presets</CardTitle>
              <CardDescription>Apply a pre-configured personality profile</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {presetNames.map((name) => (
                  <Button
                    key={name}
                    size="sm"
                    variant={activePreset === name ? 'default' : 'outline'}
                    onClick={() => handleApplyPreset(name)}
                    disabled={applyingPreset !== null}
                    className="flex flex-col items-start h-auto py-2"
                  >
                    <span className="capitalize">{name}</span>
                    <span className="text-[10px] text-muted-foreground font-normal">
                      {PRESET_DESCRIPTIONS[name] || ''}
                    </span>
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Conflict Warnings */}
        {conflicts.length > 0 && (
          <Card className="border-yellow-500/50">
            <CardHeader>
              <CardTitle className="text-yellow-600 dark:text-yellow-400">Personality Conflicts</CardTitle>
              <CardDescription>These settings may work against each other</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              {conflicts.map((c, i) => (
                <div key={i} className="flex items-start gap-2 text-sm p-2 rounded-md bg-yellow-500/5">
                  <Badge variant={c.severity === 'medium' ? 'destructive' : 'outline'} className="text-[10px] mt-0.5">
                    {c.severity}
                  </Badge>
                  <div>
                    <div>{c.message}</div>
                    <div className="text-[10px] text-muted-foreground">{c.fields.join(' + ')}</div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        )}

        {/* Core Values & Goals */}
        <Card>
          <CardHeader>
            <CardTitle>Core Identity</CardTitle>
            <CardDescription>What matters most and what the system strives toward</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 sm:space-y-4">
            <div>
              <label className="text-xs sm:text-sm font-medium">Values (comma-separated)</label>
              <input
                type="text"
                value={valuesInput}
                onChange={(e) => setValuesInput(e.target.value)}
                className="mt-1 w-full rounded-md border bg-background px-2.5 sm:px-3 py-1.5 sm:py-2 text-xs sm:text-sm"
                placeholder="helpfulness, honesty, curiosity"
              />
            </div>
            <div>
              <label className="text-xs sm:text-sm font-medium">Goals (comma-separated)</label>
              <input
                type="text"
                value={goalsInput}
                onChange={(e) => setGoalsInput(e.target.value)}
                className="mt-1 w-full rounded-md border bg-background px-2.5 sm:px-3 py-1.5 sm:py-2 text-xs sm:text-sm"
                placeholder="Provide accurate responses, Learn from interactions"
              />
            </div>
            <div>
              <label className="text-xs sm:text-sm font-medium">Interests (comma-separated)</label>
              <input
                type="text"
                value={interestsInput}
                onChange={(e) => setInterestsInput(e.target.value)}
                className="mt-1 w-full rounded-md border bg-background px-2.5 sm:px-3 py-1.5 sm:py-2 text-xs sm:text-sm"
                placeholder="AI, programming, science"
              />
            </div>
            <div>
              <label className="text-xs sm:text-sm font-medium">Avoid (comma-separated)</label>
              <input
                type="text"
                value={avoidInput}
                onChange={(e) => setAvoidInput(e.target.value)}
                className="mt-1 w-full rounded-md border bg-background px-2.5 sm:px-3 py-1.5 sm:py-2 text-xs sm:text-sm"
                placeholder="being condescending, making things up"
              />
            </div>
          </CardContent>
        </Card>

        {/* Voice Characteristics */}
        <Card>
          <CardHeader>
            <CardTitle>Voice</CardTitle>
            <CardDescription>How the system sounds when communicating</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 sm:space-y-3">
            {Object.entries(profile.voice).map(([key, value]) => (
              <div key={key} className="flex items-center gap-2 sm:gap-4">
                <label className="text-xs sm:text-sm w-24 sm:w-32 shrink-0">{VOICE_LABELS[key] || key}</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={value}
                  onChange={(e) => handleVoiceChange(key, parseFloat(e.target.value))}
                  className="flex-1 min-w-0"
                />
                <span className="text-[10px] sm:text-xs text-muted-foreground w-8 sm:w-10 text-right shrink-0">{(value * 100).toFixed(0)}%</span>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Personality Traits */}
        <Card>
          <CardHeader>
            <CardTitle>Traits</CardTitle>
            <CardDescription>Big Five personality dimensions</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 sm:space-y-3">
            {Object.entries(profile.traits).map(([key, value]) => (
              <div key={key} className="flex items-center gap-2 sm:gap-4">
                <label className="text-xs sm:text-sm w-24 sm:w-32 shrink-0">{TRAIT_LABELS[key] || key}</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={value}
                  onChange={(e) => handleTraitChange(key, parseFloat(e.target.value))}
                  className="flex-1 min-w-0"
                />
                <span className="text-[10px] sm:text-xs text-muted-foreground w-8 sm:w-10 text-right shrink-0">{(value * 100).toFixed(0)}%</span>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Communication Style */}
        <Card>
          <CardHeader>
            <CardTitle>Communication Style</CardTitle>
            <CardDescription>Preferences for how to respond</CardDescription>
          </CardHeader>
          <CardContent className="space-y-1.5 sm:space-y-2">
            {Object.entries(profile.style).map(([key, value]) => (
              <div key={key} className="flex items-center gap-2 sm:gap-3">
                <button
                  onClick={() => handleStyleToggle(key)}
                  className={`w-10 h-5 rounded-full transition-colors ${value ? 'bg-primary' : 'bg-muted'}`}
                >
                  <div className={`w-4 h-4 rounded-full bg-white transition-transform ${value ? 'translate-x-5' : 'translate-x-0.5'}`} />
                </button>
                <label className="text-sm">{STYLE_LABELS[key] || key}</label>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Import Comparison */}
        {showComparison && originalProfile && profile && (
          <Card className="border-blue-500/50">
            <CardHeader>
              <CardTitle className="text-blue-600 dark:text-blue-400">Imported Profile — Review</CardTitle>
              <CardDescription>Compare imported values with your current profile</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-2 sm:gap-4 text-xs sm:text-sm">
                <div className="font-medium text-muted-foreground">Setting</div>
                <div className="font-medium text-muted-foreground">Current</div>
                <div className="font-medium text-muted-foreground">Imported</div>
                {Object.keys(originalProfile.voice).map((key) => (
                  <>
                    <div key={`label-${key}`} className="capitalize">{key}</div>
                    <div key={`old-${key}`}>{(originalProfile.voice[key] * 100).toFixed(0)}%</div>
                    <div key={`new-${key}`} className={profile.voice[key] !== originalProfile.voice[key] ? 'text-blue-500 font-medium' : ''}>
                      {(profile.voice[key] * 100).toFixed(0)}%
                    </div>
                  </>
                ))}
              </div>
              <div className="flex gap-2 mt-4">
                <Button size="sm" variant="outline" onClick={() => { setShowComparison(false); setProfile(originalProfile) }}>
                  Discard
                </Button>
                <Button size="sm" onClick={() => setShowComparison(false)}>
                  Keep Imported
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Personality Evolution */}
        {personalityHistory.length > 1 && (
          <Card>
            <CardHeader>
              <CardTitle>Personality Evolution</CardTitle>
              <CardDescription>How your voice and traits have changed over interactions</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2">Voice Over Time</div>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={personalityHistory.map((p) => ({
                      time: new Date(p.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                      ...p.voice,
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis dataKey="time" tick={{ fontSize: 9 }} stroke="hsl(var(--muted-foreground))" />
                      <YAxis domain={[0, 1]} tick={{ fontSize: 9 }} stroke="hsl(var(--muted-foreground))" />
                      <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px', fontSize: '11px' }} />
                      <Legend wrapperStyle={{ fontSize: '10px' }} />
                      <Line type="monotone" dataKey="warmth" stroke="#ec4899" strokeWidth={1.5} dot={false} />
                      <Line type="monotone" dataKey="confidence" stroke="#6366f1" strokeWidth={1.5} dot={false} />
                      <Line type="monotone" dataKey="humor" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                      <Line type="monotone" dataKey="empathy" stroke="#22c55e" strokeWidth={1.5} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2">Traits Over Time</div>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={personalityHistory.map((p) => ({
                      time: new Date(p.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                      ...p.traits,
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis dataKey="time" tick={{ fontSize: 9 }} stroke="hsl(var(--muted-foreground))" />
                      <YAxis domain={[0, 1]} tick={{ fontSize: 9 }} stroke="hsl(var(--muted-foreground))" />
                      <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px', fontSize: '11px' }} />
                      <Legend wrapperStyle={{ fontSize: '10px' }} />
                      <Line type="monotone" dataKey="openness" stroke="#8b5cf6" strokeWidth={1.5} dot={false} />
                      <Line type="monotone" dataKey="agreeableness" stroke="#22c55e" strokeWidth={1.5} dot={false} />
                      <Line type="monotone" dataKey="conscientiousness" stroke="#3b82f6" strokeWidth={1.5} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </PageContainer>
  )
}
