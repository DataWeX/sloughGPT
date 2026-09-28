/**
 * useV86 — React hook for v86 Linux VM lifecycle.
 * Manages init, save/restore, auto-persist, and cleanup.
 */

'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { V86Controller } from '@/lib/v86-controller'
import { logger, trackEvent } from '@/lib/dev-log'

// Boot image resolution order:
// 1. local hda produced by buildroot/build.sh (docker image build)
// 2. local self-contained kernel vendored from upstream (no docker needed)
// 3. upstream self-contained kernel — copy.sh moved images to i.copy.sh;
//    the legacy copy.sh/v86/images/buildroot path is dead (404).
const LOCAL_HDA_URL = '/buildroot/buildroot.img'
const LOCAL_KERNEL_URL = '/buildroot/bzimage68.bin'
const REMOTE_KERNEL_URL = 'https://i.copy.sh/buildroot-bzimage68.bin'
const LOCAL_ISO_URL = '/buildroot/buildroot.iso'
// Hosted open-source boot CD (MikeOS, 3.3 MB, CORS-open i.copy.sh CDN) —
// fallback until buildroot produces a local ISO. ISOs are read-only media.
const REMOTE_ISO_URL = 'https://i.copy.sh/mikeos.iso'
const BIOS_URL = '/bios/seabios.bin'
const VGA_BIOS_URL = '/bios/vgabios.bin'
const WASM_PATH = '/v86/v86.wasm'
const MEMORY_MB = 256
const AUTO_SAVE_INTERVAL_MS = 30_000
const IMAGE_PROBE_TIMEOUT_MS = 5_000

export type V86ImageKind = 'hda' | 'kernel' | 'iso'

/** Boot media selection: 'auto' picks the first available image in default order. */
export type V86BootMedia = 'auto' | V86ImageKind

/** Candidate URLs per image kind, probed in order. ISOs are attached read-only (cdrom). */
export const IMAGE_URLS: Record<V86ImageKind, readonly string[]> = {
  hda: [LOCAL_HDA_URL],
  kernel: [LOCAL_KERNEL_URL, REMOTE_KERNEL_URL],
  iso: [LOCAL_ISO_URL, REMOTE_ISO_URL],
}

interface ProbeResult {
  available: boolean
  size?: number
}

/**
 * Probe an image URL with a header-only HEAD request (falling back to a
 * cancelled GET): confirms it exists and learns its total size for v86's
 * async image loader without downloading the image.
 *
 * HEAD/GET-without-custom-headers are CORS-simple requests — a Range probe
 * would trigger a preflight that CDNs without Access-Control-Allow-Headers
 * reject, silently breaking every cross-origin probe.
 * Any failure (404, 405, network, CORS, timeout) means "not available".
 */
export async function probeImage(url: string): Promise<ProbeResult> {
  try {
    let res = await fetch(url, {
      method: 'HEAD',
      referrerPolicy: 'no-referrer',
      signal: AbortSignal.timeout(IMAGE_PROBE_TIMEOUT_MS),
    })
    if (res.status === 405 || res.status === 501) {
      res = await fetch(url, {
        referrerPolicy: 'no-referrer',
        signal: AbortSignal.timeout(IMAGE_PROBE_TIMEOUT_MS),
      })
      res.body?.cancel().catch(() => undefined)
    }
    if (!res.ok && res.status !== 206) return { available: false }
    // SPA dev servers answer missing assets with 200 + index.html — never
    // treat an HTML page as a bootable disk image.
    const contentType = res.headers.get('content-type') ?? ''
    if (contentType.includes('text/html')) return { available: false }
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

/**
 * Resolve a specific image kind: probe its candidates in order and return
 * the first available one. Throws an actionable error when none are up.
 */
async function resolveImageFor(
  kind: V86ImageKind,
): Promise<{ url: string; size?: number; kind: V86ImageKind }> {
  for (const url of IMAGE_URLS[kind]) {
    const probe = await probeImage(url)
    if (probe.available) return { url, size: probe.size, kind }
  }
  throw new Error(
    `Linux VM image not available: ${IMAGE_URLS[kind].join(', ')} all unreachable. ` +
      (kind === 'iso'
        ? 'Build it via buildroot/build.sh into apps/web/public/buildroot/buildroot.iso, '
        : kind === 'hda'
          ? 'Run buildroot/build.sh to build the local image, '
          : 'Run buildroot/build.sh to build the local image, or curl -o ' +
            'apps/web/public/buildroot/bzimage68.bin ' +
            `https://i.copy.sh/buildroot-bzimage68.bin, `) +
      'then reload.',
  )
}

/**
 * Resolve the boot image: prefer the locally built hda, then a locally
 * vendored self-contained kernel, then the upstream i.copy.sh kernel, and
 * fail fast with an actionable message instead of letting v86 retry a dead
 * URL. ISO is never auto-selected — it requires an explicit boot media pick.
 */
async function resolveDefaultImage(): Promise<{ url: string; size?: number; kind: V86ImageKind }> {
  const localHda = await probeImage(LOCAL_HDA_URL)
  if (localHda.available) return { url: LOCAL_HDA_URL, size: localHda.size, kind: 'hda' }
  return resolveImageFor('kernel')
}

export interface UseV86Options {
  /** Custom image URL (overrides default Buildroot image) */
  imageUrl?: string
  /** Custom image size in bytes (required if imageUrl is provided) */
  imageSize?: number
  /** How to attach a custom imageUrl: raw hard disk (hda) or kernel boot (bzimage). */
  imageKind?: V86ImageKind
  /**
   * Boot media pick. 'auto' (default) resolves hda → kernel. A specific kind
   * probes only that kind's candidates (see IMAGE_URLS).
   */
  bootMedia?: V86BootMedia
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
  /** Tear down the running emulator and boot again with the latest options. */
  reboot: () => Promise<void>
}

export function useV86(options: UseV86Options = {}): UseV86Result {
  const [isBooted, setIsBooted] = useState(false)
  const [stateSaved, setStateSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const controllerRef = useRef<V86Controller | null>(null)
  const autoSaveRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const containerRef = useRef<HTMLElement | null>(null)
  const initGenRef = useRef(0)

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

  /**
   * Invalidate any in-flight init, stop auto-save, destroy the emulator.
   * Safe to call at any point of the boot lifecycle.
   */
  const teardown = useCallback(() => {
    initGenRef.current += 1
    if (autoSaveRef.current) {
      clearInterval(autoSaveRef.current)
      autoSaveRef.current = null
    }
    controllerRef.current?.destroy()
    controllerRef.current = null
    setIsBooted(false)
  }, [])

  const init = useCallback(
    async (container: HTMLElement) => {
      if (controllerRef.current) return
      containerRef.current = container
      const gen = ++initGenRef.current
      const stale = () => gen !== initGenRef.current
      // Explicit media picks (selector) boot fresh — skip state restore so a
      // saved hda session is never replayed onto a different image kind.
      const explicitMedia = options.bootMedia != null && options.bootMedia !== 'auto'

      try {
        const image = options.imageUrl
          ? {
              url: options.imageUrl,
              size: options.imageSize,
              kind: options.imageKind || ('hda' as const),
            }
          : options.bootMedia && options.bootMedia !== 'auto'
            ? await resolveImageFor(options.bootMedia)
            : await resolveDefaultImage()
        if (stale()) return

        const ctrl = new V86Controller()
        controllerRef.current = ctrl
        await ctrl.init(container, {
          biosUrl: options.biosUrl || BIOS_URL,
          vgaBiosUrl: options.vgaBiosUrl || VGA_BIOS_URL,
          imageUrl: image.url,
          imageSize: image.size,
          imageKind: image.kind,
          memoryMb: options.memoryMb || MEMORY_MB,
          wasmPath: options.wasmPath || WASM_PATH,
        })
        if (stale()) {
          ctrl.destroy()
          if (controllerRef.current === ctrl) controllerRef.current = null
          return
        }
        setIsBooted(true)
        setError(null)
        trackEvent('vm_booted')

        // Try to restore persisted state (auto/custom flows only)
        if (!explicitMedia) {
          const saved = await ctrl.loadPersistedState()
          if (saved && !stale()) await ctrl.restoreState(saved)
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
        if (stale()) return
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
      options.imageKind,
      options.bootMedia,
      options.memoryMb,
      options.wasmPath,
    ],
  )

  const reboot = useCallback(async () => {
    teardown()
    if (containerRef.current) await init(containerRef.current)
  }, [teardown, init])

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
      teardown()
    }
  }, [teardown])

  return { isBooted, stateSaved, error, save, restore, reset, init, reboot }
}
