'use client'
import { notify } from '@/lib/notify'

/**
 * Stateful core of the All Settings page.
 *
 * Owns IO and state only — the consciousness status fetch, the localStorage
 * prefs/personality persistence, the level/seed/clear/reset mutations and
 * the settings export/import. The tabs own render.
 */
import { useState, useEffect, useCallback, useRef } from 'react'
import { consciousnessController } from '@/lib/consciousness-controller'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'
import { addGlobalError } from '@/lib/error-store'
import {
  LEVEL_LABELS,
  DEFAULT_PREFS,
  DEFAULT_PERSONALITY,
  PRESETS,
  loadAllPrefs,
  saveAllPrefs,
  loadPersonality,
  savePersonality,
  type AllPrefs,
  type PersonalityPrefs,
  type ConfigStatus,
  type FullStatus,
} from './all-settings-prefs'
// Module scope: `addToast` must keep a stable identity — it sits in
// useCallback/useEffect dependency arrays, and a per-render arrow would
// re-fire those effects on every render.
const addToast = (message: string, type?: string) =>
  notify.toast(message, { type: type as 'success' | 'error' | 'info' })

export function useAllSettingsWorkbench() {
  const { t } = useLocale()
  const [config, setConfig] = useState<ConfigStatus | null>(null)
  const [status, setStatus] = useState<FullStatus | null>(null)
  const [prefs, setPrefs] = useState<AllPrefs>(DEFAULT_PREFS)
  const [personality, setPersonality] = useState<PersonalityPrefs>(DEFAULT_PERSONALITY)
  const [loading, setLoading] = useState(true)
  const [seedCount, setSeedCount] = useState(10)
  const [seeding, setSeeding] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const fetchStatus = useCallback(async () => {
    try {
      const data = await consciousnessController.getStatus()
      setStatus(data as FullStatus)
      setConfig({
        level: (data as FullStatus).level ?? 0,
        enabled: (data as FullStatus).enabled ?? false,
      })
    } catch (e) {
      addGlobalError(e, 'ConsciousnessAllSettings')
    }
  }, [])

  useEffect(() => {
    setPrefs(loadAllPrefs())
    setPersonality(loadPersonality())
    Promise.all([fetchStatus()]).finally(() => setLoading(false))
  }, [fetchStatus])

  const updatePrefs = (partial: Partial<AllPrefs>) => {
    setPrefs((prev) => {
      const next = { ...prev, ...partial }
      saveAllPrefs(next)
      return next
    })
  }

  const updatePersonality = (partial: Partial<PersonalityPrefs>) => {
    setPersonality((prev) => {
      const next = { ...prev, ...partial }
      savePersonality(next)
      return next
    })
  }

  const applyPreset = (presetName: string) => {
    const preset = PRESETS[presetName]
    if (preset) {
      setPersonality((prev) => {
        const next = { ...prev, ...preset }
        savePersonality(next)
        return next
      })
      notify.toast(`Applied ${presetName} preset`, { type: 'success' })
    }
  }

  const updateLevel = async (newLevel: number) => {
    try {
      await consciousnessController.updateConfig({ level: newLevel })
      setConfig((prev) => (prev ? { ...prev, level: newLevel, enabled: newLevel > 0 } : prev))
      notify.toast(`Consciousness level set to ${LEVEL_LABELS[newLevel]}`, { type: 'success' })
    } catch (e) {
      notify.toast(extractErrorMessage(e), { type: 'error' })
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
      notify.toast(`Seeded ${seedCount} episodes`, { type: 'success' })
      fetchStatus()
    } catch (e) {
      notify.toast(extractErrorMessage(e), { type: 'error' })
    } finally {
      setSeeding(false)
    }
  }

  const handleClearEpisodes = async () => {
    try {
      await consciousnessController.clearEpisodes()
      notify.toast('Episodes cleared', { type: 'success' })
      fetchStatus()
    } catch (e) {
      notify.toast(extractErrorMessage(e), { type: 'error' })
    }
  }

  const handleClearBeliefs = async () => {
    try {
      await consciousnessController.resetBeliefs()
      notify.toast('Beliefs reset to defaults', { type: 'success' })
      fetchStatus()
    } catch (e) {
      notify.toast(extractErrorMessage(e), { type: 'error' })
    }
  }

  const handleResetPersonality = async () => {
    try {
      await consciousnessController.resetPersonality()
      notify.toast('Personality reset to defaults', { type: 'success' })
      setPersonality(DEFAULT_PERSONALITY)
      savePersonality(DEFAULT_PERSONALITY)
      fetchStatus()
    } catch (e) {
      notify.toast(extractErrorMessage(e), { type: 'error' })
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
      notify.toast('Settings exported', { type: 'success' })
    } catch (e) {
      notify.toast(extractErrorMessage(e), { type: 'error' })
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
        notify.toast('Settings imported', { type: 'success' })
      } catch {
        notify.toast('Invalid settings file', { type: 'error' })
      }
    }
    reader.readAsText(file)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  return {
    addToast,
    applyPreset,
    t,
    config,
    fetchStatus,
    fileInputRef,
    handleClearBeliefs,
    handleClearEpisodes,
    handleExport,
    handleImport,
    handleResetPersonality,
    handleSeed,
    loading,
    personality,
    prefs,
    seedCount,
    seeding,
    setConfig,
    setLoading,
    setPersonality,
    setPrefs,
    setSeedCount,
    setSeeding,
    setStatus,
    status,
    toggleEnabled,
    updateLevel,
    updatePersonality,
    updatePrefs,
  }
}

/** The contract the All Settings tabs render against. */
export type AllSettingsWorkbench = ReturnType<typeof useAllSettingsWorkbench>
