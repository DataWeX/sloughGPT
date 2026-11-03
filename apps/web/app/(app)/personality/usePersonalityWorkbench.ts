'use client'

/**
 * Stateful core of the Personality page.
 *
 * Owns IO and state only — profile/presets/conflicts fetches (history is a
 * protocol read through lib/protocol), the saved-persona list with activate/delete, trait edits and their
 * save, preset application, CSV-ish comma-list inputs and the import
 * comparison. The sections own render.
 */
import { useState, useEffect, useCallback } from 'react'
import { consciousnessController } from '@/lib/consciousness-controller'
import { apiPatch } from '@/lib/http-client'
import { extractErrorMessage } from '@/lib/error-utils'
import { useToastStore } from '@/lib/toast-store'
import { useEndpoint } from '@/lib/protocol/useEndpoint'
import { useLocale } from '@/hooks/useLocale'
import {
  type SavedPersona,
  type PersonalityConflict,
  type PersonalityProfile,
  type PersonalityHistory,
} from './personality-model'

export function usePersonalityWorkbench() {
  const addToast = useToastStore((state) => state.addToast)
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

  // History is a protocol read: the capability name resolves the path through
  // the generated contract registry (lib/protocol), not a hand-written call.
  const historyQuery = useEndpoint<{ history: PersonalityHistory[] }>(
    'consciousness.personality_history',
  )
  const personalityHistory = historyQuery.data?.history ?? []

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
      const [profileResult, presetsResult, conflictsResult] = await Promise.allSettled([
        consciousnessController.getPersonality(),
        consciousnessController.getPersonalityPresets(),
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
