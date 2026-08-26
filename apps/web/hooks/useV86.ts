/**
 * useV86 — React hook for v86 Linux VM lifecycle.
 * Manages init, save/restore, auto-persist, and cleanup.
 */

'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { V86Controller } from '@/lib/v86-controller'
import { logger } from '@/lib/dev-log'

const LINUX_IMAGE_URL = 'https://copy.sh/v86/images/buildroot'
const LINUX_IMAGE_SIZE = 8 * 1024 * 1024 // 8MB for Buildroot
const BIOS_URL = '/bios/seabios.bin'
const VGA_BIOS_URL = '/bios/vgabios.bin'
const WASM_PATH = '/v86/v86.wasm'
const MEMORY_MB = 256
const AUTO_SAVE_INTERVAL_MS = 30_000

export interface UseV86Result {
  isBooted: boolean
  stateSaved: boolean
  error: string | null
  save: () => Promise<void>
  restore: () => Promise<void>
  reset: () => void
  init: (container: HTMLElement) => Promise<void>
}

export function useV86(): UseV86Result {
  const [isBooted, setIsBooted] = useState(false)
  const [stateSaved, setStateSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const controllerRef = useRef<V86Controller | null>(null)
  const autoSaveRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const containerRef = useRef<HTMLElement | null>(null)

  // Check for persisted state on mount
  useEffect(() => {
    let active = true
    new V86Controller().loadPersistedState().then((s) => { if (active) setStateSaved(!!s) })
    return () => { active = false }
  }, [])

  const init = useCallback(async (container: HTMLElement) => {
    if (controllerRef.current) return
    containerRef.current = container

    try {
      const ctrl = new V86Controller()
      await ctrl.init(container, {
        biosUrl: BIOS_URL,
        vgaBiosUrl: VGA_BIOS_URL,
        imageUrl: LINUX_IMAGE_URL,
        imageSize: LINUX_IMAGE_SIZE,
        memoryMb: MEMORY_MB,
        wasmPath: WASM_PATH,
      })
      controllerRef.current = ctrl
      setIsBooted(true)
      setError(null)

      // Try to restore persisted state
      const saved = await ctrl.loadPersistedState()
      if (saved) {
        await ctrl.restoreState(saved)
      }

      // Start auto-save
      autoSaveRef.current = setInterval(() => {
        ctrl.persistState().catch(e => { logger.warning('VM auto-save failed', { exception: String(e) }) })
      }, AUTO_SAVE_INTERVAL_MS)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Could not start Linux VM')
    }
  }, [])

  const save = useCallback(async () => {
    const ctrl = controllerRef.current
    if (!ctrl) return
    await ctrl.persistState()
    setStateSaved(true)
  }, [])

  const restore = useCallback(async () => {
    const ctrl = controllerRef.current
    if (!ctrl) return
    const state = await ctrl.loadPersistedState()
    if (state) {
      await ctrl.restoreState(state)
    }
  }, [])

  const reset = useCallback(() => {
    controllerRef.current?.restart()
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (autoSaveRef.current) clearInterval(autoSaveRef.current)
      controllerRef.current?.destroy()
    }
  }, [])

  return { isBooted, stateSaved, error, save, restore, reset, init }
}
