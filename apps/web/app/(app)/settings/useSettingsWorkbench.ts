'use client'

/**
 * Stateful core of the Settings page.
 *
 * Owns IO and state only — the system health/metrics/disk/info fetches,
 * process-guard status and toggling, the API connection test, chat-history
 * clear and full settings reset. Appearance, locale and palette route
 * through the shared settings/theme/locale stores. The panels own render.
 */
import { notify } from '@/lib/notify'
import { useState, useEffect, useCallback } from 'react'
import { useSettings, useUpdateSettings, DEFAULT_SETTINGS } from '@/lib/store'
import { systemController } from '@/lib/system-controller'
import { modelController } from '@/lib/model-controller'
import { chatDB } from '@/lib/db'
import { CURRENT_SESSION_KEY } from '@/lib/chat-utils'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLiveStatus } from '@/hooks/useLiveStatus'
import { useLocale } from '@/hooks/useLocale'
import { useTheme } from '@/components/ThemeProvider'
import { useRefreshShortcut } from '@/hooks/useRefreshShortcut'
import {
  type DetailedHealth,
  type SystemMetrics,
  type DiskUsage,
  type SystemInfo,
} from '@/lib/system-controller'

export function useSettingsWorkbench() {
  const settings = useSettings()
  const updateSettings = useUpdateSettings()
  const addToast = notify.toast
  const { healthLegacy: apiHealth } = useLiveStatus()
  const { locale, setLocale } = useLocale()
  const { palette, setPalette } = useTheme()
  const [detailed, setDetailed] = useState<DetailedHealth | null>(null)
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
  const [disk, setDisk] = useState<DiskUsage | null>(null)
  const [info, setInfo] = useState<SystemInfo | null>(null)
  const [connectionTest, setConnectionTest] = useState<{
    status: 'idle' | 'testing' | 'ok' | 'error'
    latency?: number
    error?: string
  }>({ status: 'idle' })
  const [settingsErrors, setSettingsErrors] = useState<{ apiUrl?: string; hfToken?: string }>({})
  const [processGuard, setProcessGuard] = useState<
    import('@/lib/system-controller').ProcessGuardStatus | null
  >(null)
  const [healthError, setHealthError] = useState(false)

  const fetchHealth = useCallback(async () => {
    setHealthError(false)
    const [d, m, dk, inf] = await Promise.allSettled([
      systemController.getDetailedHealth(),
      systemController.getMetrics(),
      systemController.getDisk(),
      systemController.getInfo(),
    ])
    if (d.status === 'fulfilled') setDetailed(d.value)
    if (m.status === 'fulfilled') setMetrics(m.value)
    if (dk.status === 'fulfilled') setDisk(dk.value)
    if (inf.status === 'fulfilled') setInfo(inf.value)
    if (
      d.status === 'rejected' &&
      m.status === 'rejected' &&
      dk.status === 'rejected' &&
      inf.status === 'rejected'
    ) {
      setHealthError(true)
    }
  }, [])

  useRefreshShortcut(fetchHealth)

  useEffect(() => {
    fetchHealth()
  }, [fetchHealth])

  const fetchProcessGuard = useCallback(async () => {
    try {
      const status = await systemController.getProcessGuardStatus()
      setProcessGuard(status)
    } catch {
      /* unavailable */
    }
  }, [])

  useEffect(() => {
    fetchProcessGuard()
  }, [fetchProcessGuard])

  const handleToggleProcessGuard = async (enabled: boolean) => {
    try {
      const status = await systemController.setProcessGuardEnabled(enabled)
      setProcessGuard(status)
      addToast(enabled ? 'Process isolation enabled' : 'Process isolation disabled', 'success')
    } catch (e: unknown) {
      addToast(extractErrorMessage(e, 'Could not toggle process guard'), 'error')
    }
  }

  const isOnline = apiHealth !== null && apiHealth !== 'offline'
  const apiOk = isOnline && (apiHealth.status === 'healthy' || detailed?.status === 'healthy')
  const modelLoaded = isOnline ? Boolean(apiHealth.model_loaded || detailed?.model_loaded) : false
  const modelType = isOnline ? ((apiHealth.model_type || detailed?.model_type) ?? null) : null

  const handleTestConnection = async () => {
    setConnectionTest({ status: 'testing' })
    const start = Date.now()
    try {
      const h = await modelController.getHealth()
      const latency = Date.now() - start
      if (h && (h.status === 'healthy' || h.model_loaded !== undefined)) {
        setConnectionTest({ status: 'ok', latency })
      } else {
        setConnectionTest({ status: 'error', error: 'Unexpected response' })
      }
    } catch (e: unknown) {
      setConnectionTest({ status: 'error', error: extractErrorMessage(e, 'Could not connection') })
    }
  }

  const clearChat = () => {
    chatDB.deleteKV(CURRENT_SESSION_KEY).catch(() => {})
    addToast('Chat history cleared', 'success')
  }

  const resetAllSettings = () => {
    updateSettings(DEFAULT_SETTINGS)
    addToast('Settings reset to defaults', 'success')
  }

  return {
    addToast,
    apiHealth,
    apiOk,
    clearChat,
    connectionTest,
    detailed,
    disk,
    fetchHealth,
    fetchProcessGuard,
    handleTestConnection,
    handleToggleProcessGuard,
    healthError,
    info,
    isOnline,
    locale,
    metrics,
    modelLoaded,
    modelType,
    palette,
    processGuard,
    resetAllSettings,
    setConnectionTest,
    setDetailed,
    setDisk,
    setHealthError,
    setInfo,
    setLocale,
    setMetrics,
    setPalette,
    setProcessGuard,
    setSettingsErrors,
    settings,
    settingsErrors,
    updateSettings,
  }
}

/** The contract the Settings panels render against. */
export type SettingsWorkbench = ReturnType<typeof useSettingsWorkbench>
