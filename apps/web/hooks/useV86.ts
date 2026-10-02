/**
 * useV86 — React hook for v86 Linux VM lifecycle.
 * Manages init, save/restore, auto-persist, and cleanup.
 */

'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { V86Controller } from '@/lib/v86-controller'
import { logger, trackEvent } from '@/lib/dev-log'

const LINUX_IMAGE_URL = 'https://copy.sh/v86/images/buildroot'
// Local image produced by buildroot/build.sh (copied to apps/web/public/buildroot/).
const LOCAL_IMAGE_URL = '/buildroot/buildroot.img'
// Local browser-linux ISO (Darin755/browser-linux minimal) — boots on the CD-ROM.
// copy.sh's buildroot image host is dead, so this is the default Linux media.
const LOCAL_ISO_URL = '/v86/browser-linux.iso'
const BIOS_URL = '/bios/seabios.bin'
const VGA_BIOS_URL = '/bios/vgabios.bin'
const WASM_PATH = '/v86/v86.wasm'
const MEMORY_MB = 64
const AUTO_SAVE_INTERVAL_MS = 30_000
const IMAGE_PROBE_TIMEOUT_MS = 5_000

interface ProbeResult {
  available: boolean
  size?: number
}

/**
 * Probe an image URL with a 1-byte range GET: confirms it exists and learns
 * its total size (for v86's async image loader) without downloading it.
 * Any failure (404, network, CORS, timeout) means "not available".
 */
async function probeImage(url: string): Promise<ProbeResult> {
  try {
    const res = await fetch(url, {
      headers: { Range: 'bytes=0-0' },
      signal: AbortSignal.timeout(IMAGE_PROBE_TIMEOUT_MS),
    })
    if (!res.ok && res.status !== 206) return { available: false }
    // Dev servers (Vite SPA fallback, Next) answer unknown paths with 200 +
    // index.html — treat an HTML response as "image not present".
    const ctype = res.headers.get('content-type') || ''
    if (ctype.startsWith('text/html')) return { available: false }
    const total = res.headers.get('content-range')?.split('/')[1]
    if (total && total !== '*') {
      const n = Number(total)
      if (Number.isFinite(n) && n > 0) return { available: true, size: n }
    }
    const len = Number(res.headers.get('content-length'))
    return { available: true, size: Number.isFinite(len) && len > 0 ? len : undefined }
  } catch {
    return { available: false }
  }
}

export interface ResolvedBootImage {
  url: string
  size?: number
  /** How v86 should attach the media: hard disk (hda) or CD-ROM. */
  media: 'disk' | 'cdrom'
}

/**
 * Resolve the boot media: prefer the locally built disk image, then the local
 * browser-linux ISO (CD-ROM; copy.sh's image host is dead), then the upstream
 * copy.sh buildroot image, and fail fast with an actionable message instead of
 * letting v86 retry a dead URL.
 */
async function resolveDefaultImage(): Promise<ResolvedBootImage> {
  const local = await probeImage(LOCAL_IMAGE_URL)
  if (local.available) return { url: LOCAL_IMAGE_URL, size: local.size, media: 'disk' }
  const iso = await probeImage(LOCAL_ISO_URL)
  if (iso.available) return { url: LOCAL_ISO_URL, size: iso.size, media: 'cdrom' }
  const remote = await probeImage(LINUX_IMAGE_URL)
  if (remote.available) return { url: LINUX_IMAGE_URL, size: remote.size, media: 'disk' }
  throw new Error(
    `Linux VM image not available: ${LOCAL_IMAGE_URL} not built and ${LOCAL_ISO_URL} missing. ` +
      'Run buildroot/build.sh to build the local image (installed at apps/web/public/buildroot/buildroot.img), ' +
      'or restore apps/web/public/v86/browser-linux.iso, then reload.',
  )
}

export interface UseV86Options {
  /** Custom image URL (overrides default Buildroot image) */
  imageUrl?: string
  /** Custom image size in bytes (required if imageUrl is provided) */
  imageSize?: number
  /** Custom BIOS URL */
  biosUrl?: string
  /** Custom VGA BIOS URL */
  vgaBiosUrl?: string
  /** Custom WASM path */
  wasmPath?: string
  /** Memory in MB */
  memoryMb?: number
}

export interface UseV86Result {
  isBooted: boolean
  stateSaved: boolean
  error: string | null
  save: () => Promise<void>
  restore: () => Promise<void>
  reset: () => void
  init: (container: HTMLElement) => Promise<void>
}

export function useV86(options: UseV86Options = {}): UseV86Result {
  const [isBooted, setIsBooted] = useState(false)
  const [stateSaved, setStateSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const controllerRef = useRef<V86Controller | null>(null)
  const autoSaveRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const containerRef = useRef<HTMLElement | null>(null)

  // Check for persisted state on mount
  useEffect(() => {
    let active = true
    new V86Controller().loadPersistedState().then((s) => {
      if (active) setStateSaved(!!s)
    })
    return () => {
      active = false
    }
  }, [])

  const init = useCallback(
    async (container: HTMLElement) => {
      if (controllerRef.current) return
      containerRef.current = container

      try {
        // An explicit imageUrl is always treated as a disk image (hda).
        const image: ResolvedBootImage = options.imageUrl
          ? { url: options.imageUrl, size: options.imageSize, media: 'disk' }
          : await resolveDefaultImage()

        const ctrl = new V86Controller()
        await ctrl.init(container, {
          biosUrl: options.biosUrl || BIOS_URL,
          vgaBiosUrl: options.vgaBiosUrl || VGA_BIOS_URL,
          imageUrl: image.url,
          imageSize: image.size,
          cdromUrl: image.media === 'cdrom' ? image.url : undefined,
          memoryMb: options.memoryMb || MEMORY_MB,
          wasmPath: options.wasmPath || WASM_PATH,
        })
        controllerRef.current = ctrl
        setIsBooted(true)
        setError(null)
        trackEvent('vm_booted')

        // Try to restore persisted state
        const saved = await ctrl.loadPersistedState()
        if (saved) {
          await ctrl.restoreState(saved)
        }

        // Start auto-save — skip ticks when the emulator is not running
        // (failed boot / stopped): save_state on a non-running machine throws
        // and only spams logs every 30s.
        autoSaveRef.current = setInterval(() => {
          if (!ctrl.isRunning()) return
          ctrl.persistState().catch((e) => {
            logger.warning('VM auto-save failed', { exception: String(e) })
          })
        }, AUTO_SAVE_INTERVAL_MS)
      } catch (err: unknown) {
        trackEvent('vm_boot_error', {
          error: err instanceof Error ? err.message : 'Could not start Linux VM',
        })
        setError(err instanceof Error ? err.message : 'Could not start Linux VM')
      }
    },
    [
      options.biosUrl,
      options.vgaBiosUrl,
      options.imageUrl,
      options.imageSize,
      options.memoryMb,
      options.wasmPath,
    ],
  )

  const save = useCallback(async () => {
    const ctrl = controllerRef.current
    if (!ctrl) return
    await ctrl.persistState()
    setStateSaved(true)
    trackEvent('vm_saved')
  }, [])

  const restore = useCallback(async () => {
    const ctrl = controllerRef.current
    if (!ctrl) return
    const state = await ctrl.loadPersistedState()
    if (state) {
      await ctrl.restoreState(state)
      trackEvent('vm_restored')
    }
  }, [])

  const reset = useCallback(() => {
    controllerRef.current?.restart()
    trackEvent('vm_reset')
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
