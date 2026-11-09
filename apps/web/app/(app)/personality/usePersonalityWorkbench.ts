'use client'
import { notify } from '@/lib/notify'

/**
 * Stateful core of the Personality page.
 *
 * Owns IO and state only — profile/presets/history/conflicts fetches,
 * the saved-persona list with activate/delete, trait edits and their
 * save, preset application, CSV-ish comma-list inputs and the import
 * comparison. The sections own render.
 */
import { useState, useEffect, useCallback } from 'react'
import { consciousnessController } from '@/lib/consciousness-controller'
import { apiPatch } from '@/lib/http-client'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'
import {
  type SavedPersona,
  type PersonalityConflict,
  type PersonalityProfile,
  type PersonalityHistory,
} from './personality-model'
// Module scope: `addToast` must keep a stable identity — it sits in
// useCallback/useEffect dependency arrays, and a per-render arrow would
// re-fire those effects on every render.
const addToast = (message: string, type?: string) =>
  notify.toast(message, { type: type as 'success' | 'error' | 'info' })

export function usePersonalityWorkbench() {
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
      const data = (await consciousnessController.listPersonas()) as any
      setSavedPersonas(data.personas ?? [])
    } catch {
      // Saved personas are optional — the page works without them.
    }
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
      notify.toast(extractErrorMessage(e), { type: 'error' })
    } finally {
      setLoading(false)
    }
    fetchPersonas()
  }, [fetchPersonas])

  useEffect(() => {
    fetchProfile()
  }, [fetchProfile])

  const handleSave = async () => {
    if (!profile) return
    setSaving(true)
    try {
      const data = await apiPatch('/consciousness/personality', {
        values: valuesInput
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean),
        goals: goalsInput
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean),
        interests: interestsInput
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean),
        avoid: avoidInput
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean),
        voice: profile.voice,
        traits: profile.traits,
        style: profile.style,
      })
      setProfile(data as PersonalityProfile)
      notify.toast('Personality saved', { type: 'success' })
      const conflictsData = await consciousnessController.getPersonalityConflicts()
      setConflicts((conflictsData as any)?.conflicts ?? [])
    } catch (e) {
      notify.toast(extractErrorMessage(e), { type: 'error' })
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
    notify.toast('Personality exported', { type: 'success' })
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
        notify.toast('Personality imported — review and save', { type: 'success' })
      } catch {
        notify.toast('Invalid JSON file', { type: 'error' })
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
      notify.toast('Personality reset to defaults', { type: 'success' })
    } catch (e) {
      notify.toast(extractErrorMessage(e), { type: 'error' })
    }
  }

  const handleApplyPreset = async (presetName: string) => {
    setApplyingPreset(presetName)
    try {
      await consciousnessController.applyPersonalityPreset(presetName)
      fetchProfile()
      setActivePreset(presetName)
      notify.toast(`Applied "${presetName}" preset`, { type: 'success' })
    } catch (e) {
      notify.toast(extractErrorMessage(e), { type: 'error' })
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
      notify.toast('Persona saved', { type: 'success' })
      setShowSavePersonaDialog(false)
      setPersonaNameInput('')
      fetchPersonas()
    } catch (e) {
      notify.toast(extractErrorMessage(e), { type: 'error' })
    } finally {
      setSavingPersona(false)
    }
  }

  const handleActivatePersona = async (id: string) => {
    setActivatingPersona(id)
    try {
      await consciousnessController.activatePersona(id)
      notify.toast('Persona activated', { type: 'success' })
      fetchProfile()
    } catch (e) {
      notify.toast(extractErrorMessage(e), { type: 'error' })
    } finally {
      setActivatingPersona(null)
    }
  }

  const handleDeletePersona = async (id: string) => {
    setDeletingPersona(id)
    try {
      await consciousnessController.deletePersona(id)
      notify.toast('Persona deleted', { type: 'success' })
      fetchPersonas()
    } catch (e) {
      notify.toast(extractErrorMessage(e), { type: 'error' })
    } finally {
      setDeletingPersona(null)
    }
  }

  return {
    PRESET_DESCRIPTIONS,
    activatingPersona,
    activePreset,
    addToast,
    applyingPreset,
    avoidInput,
    conflicts,
    deletingPersona,
    fetchPersonas,
    fetchProfile,
    goalsInput,
    handleActivatePersona,
    handleApplyPreset,
    handleDeletePersona,
    handleExport,
    handleImport,
    handleReset,
    handleSave,
    handleSavePersona,
    handleStyleToggle,
    handleTraitChange,
    handleVoiceChange,
    interestsInput,
    loading,
    originalProfile,
    personaNameInput,
    personalityHistory,
    presetNames,
    profile,
    savedPersonas,
    saving,
    savingPersona,
    setActivatingPersona,
    setActivePreset,
    setApplyingPreset,
    setAvoidInput,
    setConflicts,
    setDeletingPersona,
    setGoalsInput,
    setInterestsInput,
    setLoading,
    setOriginalProfile,
    setPersonaNameInput,
    setPersonalityHistory,
    setPresetNames,
    setProfile,
    setSavedPersonas,
    setSaving,
    setSavingPersona,
    setShowComparison,
    setShowSavePersonaDialog,
    setValuesInput,
    showComparison,
    showSavePersonaDialog,
    t,
    valuesInput,
  }
}

/** The contract the Personality sections render against. */
export type PersonalityWorkbench = ReturnType<typeof usePersonalityWorkbench>
