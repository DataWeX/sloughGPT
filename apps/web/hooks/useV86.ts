/**
 * useV86 — React hook for v86 Linux VM lifecycle.
 * Manages init, save/restore, auto-persist, and cleanup.
 */

'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { V86Controller } from '@/lib/v86-controller'
import { getGuiStatePreset } from '@/lib/v86-gui-states'
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
const LOCAL_BIOS_URL = '/bios/seabios.bin'
const LOCAL_VGA_BIOS_URL = '/bios/vgabios.bin'
// BIOS binaries ship in neither the tree nor the npm package; copy.sh hosts
// the SeaBIOS/VGA pair for the public demo, so fall back to it like the kernel.
const REMOTE_BIOS_URL = 'https://copy.sh/v86/bios/seabios.bin'
const REMOTE_VGA_BIOS_URL = 'https://copy.sh/v86/bios/vgabios.bin'
const WASM_PATH = '/v86/v86.wasm'
const MEMORY_MB = 256
/**
 * Kernel command line for kernel-mode boots — the copy.sh demo preset for
 * this exact kernel (buildroot6), plus `earlyprintk=serial,ttyS0,115200`
 * which mirrors early boot to COM1 (v86 always emulates the UART; the
 * kernel disables the bootconsole once the VGA console takes over, so this
 * costs nothing after early boot and keeps boot evidence readable over
 * serial for debugging). The full chain — these flags with the vendored
 * lib/wasm pair — boots this image to the busybox shell prompt; verified
 * over serial 2026-10-18. `tsc=reliable` covers v86's TSC, which Linux
 * otherwise distrusts as a clocksource.
 */
const KERNEL_CMDLINE = 'earlyprintk=serial,ttyS0,115200 tsc=reliable mitigations=off random.trust_cpu=on'
const AUTO_SAVE_INTERVAL_MS = 30_000
const IMAGE_PROBE_TIMEOUT_MS = 10_000

export type V86ImageKind = 'hda' | 'kernel' | 'iso'

/** Boot media selection: 'auto' picks the first available image in default order. */
export type V86BootMedia = 'auto' | V86ImageKind

/** Candidate URLs per image kind, probed in order. ISOs are attached read-only (cdrom). */
export const IMAGE_URLS: Record<V86ImageKind, readonly string[]> = {
  hda: [LOCAL_HDA_URL],
  kernel: [LOCAL_KERNEL_URL, REMOTE_KERNEL_URL],
  iso: [LOCAL_ISO_URL],
}

/** BIOS pairs probed in order; the VGA BIOS always comes from the same host as the system BIOS. */
export const BIOS_URLS: readonly { bios: string; vga: string }[] = [
  { bios: LOCAL_BIOS_URL, vga: LOCAL_VGA_BIOS_URL },
  { bios: REMOTE_BIOS_URL, vga: REMOTE_VGA_BIOS_URL },
]

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
 * Transient failures (timeout/network/429/5xx) are retried once — CDN edges
 * occasionally stall a cold HEAD past the timeout, and a single flake would
 * otherwise disable a boot medium or fail resolution.
 */
export async function probeImage(url: string): Promise<ProbeResult> {
  for (let attempt = 0; attempt < 2; attempt++) {
    if (attempt > 0) await new Promise((r) => setTimeout(r, 400))
    let outcome: { result: ProbeResult; retryable: boolean }
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
      if (!res.ok && res.status !== 206) {
        outcome = {
          result: { available: false },
          retryable: res.status === 429 || res.status >= 500,
        }
      } else if ((res.headers.get('content-type') ?? '').includes('text/html')) {
        // SPA dev servers answer missing assets with 200 + index.html — never
        // treat an HTML page as a bootable disk image.
        outcome = { result: { available: false }, retryable: false }
      } else {
        const total = res.headers.get('content-range')?.split('/')[1]
        const fromRange = total && total !== '*' ? Number(total) : NaN
        const len = Number(res.headers.get('content-length'))
        const size =
          Number.isFinite(fromRange) && fromRange > 0
            ? fromRange
            : Number.isFinite(len) && len > 0
              ? len
              : undefined
        outcome = { result: { available: true, size }, retryable: false }
      }
    } catch {
      outcome = { result: { available: false }, retryable: true }
    }
    if (outcome.retryable && attempt === 0) continue
    return outcome.result
  }
  return { available: false }
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
 * Resolve the BIOS pair: prefer locally vendored binaries, fall back to the
 * copy.sh-hosted pair (same fail-fast contract as resolveImageFor).
 */
export async function resolveBios(): Promise<{ bios: string; vga: string }> {
  for (const pair of BIOS_URLS) {
    const probe = await probeImage(pair.bios)
    if (probe.available) return pair
  }
  throw new Error(
    `BIOS not available: ${BIOS_URLS.map((p) => p.bios).join(', ')} all unreachable.`,
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
  /**
   * GUI desktop snapshot boot: a preset id from GUI_STATE_PRESETS. Boots via
   * v86 initial_state with the save-time machine profile (exact memory, async
   * hda, NIC class) and skips image probing plus IndexedDB restore/auto-save.
   */
  statePresetId?: string
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
  /** True when the machine is frozen in standby (panel closed / power off). */
  isSuspended: boolean
  stateSaved: boolean
  error: string | null
  save: () => Promise<void>
  restore: () => Promise<void>
  reset: () => void
  init: (container: HTMLElement) => Promise<void>
  /** Tear down the running emulator and boot again with the latest options. */
  reboot: () => Promise<void>
  /** Freeze the machine into standby (screen detached, CPU stopped). */
  suspend: () => void
  /** Power on from standby. */
  resume: () => void
}

/**
 * The VM session is shared across hook instances: closing the panel suspends
 * and detaches the screen, but the machine keeps living at module scope —
 * reopening the panel adopts it without a reboot (process lifetime decoupled
 * from view lifetime). reboot() is the only real destroy path; a page reload
 * is the only true session end.
 */
let sharedController: V86Controller | null = null

/** Test-only: forget the shared session so each test starts with a fresh machine. */
export function resetV86SessionForTests(): void {
  sharedController = null
}

export function useV86(options: UseV86Options = {}): UseV86Result {
  const [isBooted, setIsBooted] = useState(false)
  const [isSuspended, setIsSuspended] = useState(false)
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
    if (sharedController === controllerRef.current) sharedController = null
    controllerRef.current = null
    setIsBooted(false)
    setIsSuspended(false)
  }, [])

  const init = useCallback(
    async (container: HTMLElement) => {
      if (controllerRef.current) return
      containerRef.current = container
      const gen = ++initGenRef.current
      const stale = () => gen !== initGenRef.current

      const preset = options.statePresetId ? getGuiStatePreset(options.statePresetId) : undefined
      if (options.statePresetId && !preset) {
        setError(`Unknown GUI state preset: ${options.statePresetId}`)
        return
      }
      // Explicit picks (media selector or desktop gallery) boot fresh — skip
      // state restore so a saved session is never replayed onto a different
      // machine.
      const explicitMedia =
        preset != null || (options.bootMedia != null && options.bootMedia !== 'auto')

      try {
        // Reopen path: the shared controller already owns a live machine —
        // adopt it (reattach the screen host) instead of rebooting.
        if (sharedController?.hasMachine()) {
          sharedController.attachScreen(container)
          controllerRef.current = sharedController
          setIsBooted(true)
          setIsSuspended(sharedController.isSuspended())
          return
        }

        const image = preset
          ? null
          : options.imageUrl
            ? {
                url: options.imageUrl,
                size: options.imageSize,
                kind: options.imageKind || ('hda' as const),
              }
            : options.bootMedia && options.bootMedia !== 'auto'
              ? await resolveImageFor(options.bootMedia)
              : await resolveDefaultImage()
        if (stale()) return

        const bios = options.biosUrl
          ? { bios: options.biosUrl, vga: LOCAL_VGA_BIOS_URL }
          : await resolveBios()
        if (stale()) return

        const ctrl = sharedController ?? new V86Controller()
        sharedController = ctrl
        controllerRef.current = ctrl
        // Invariant: exactly one of (state preset, resolved image) is set.
        if (preset) {
          await ctrl.init(container, {
            biosUrl: bios.bios,
            vgaBiosUrl: options.vgaBiosUrl || bios.vga,
            wasmPath: options.wasmPath || WASM_PATH,
            state: {
              url: preset.stateUrl,
              memoryMb: preset.memoryMb,
              hda: {
                url: preset.hdaUrl,
                sizeBytes: preset.hdaSizeBytes,
                chunkBytes: preset.hdaChunkBytes,
              },
              netDeviceType: preset.netDeviceType,
            },
          })
        } else if (image) {
          await ctrl.init(container, {
            biosUrl: bios.bios,
            vgaBiosUrl: options.vgaBiosUrl || bios.vga,
            imageUrl: image.url,
            imageSize: image.size,
            imageKind: image.kind,
            memoryMb: options.memoryMb || MEMORY_MB,
            wasmPath: options.wasmPath || WASM_PATH,
            cmdline: image.kind === 'kernel' ? KERNEL_CMDLINE : undefined,
          })
        }
        if (stale()) {
          ctrl.destroy()
          if (sharedController === ctrl) sharedController = null
          if (controllerRef.current === ctrl) controllerRef.current = null
          return
        }
        setIsBooted(true)
        setIsSuspended(false)
        setError(null)
        trackEvent('vm_booted')

        // Try to restore persisted state (auto/custom flows only)
        if (!explicitMedia) {
          const saved = await ctrl.loadPersistedState()
          if (saved && !stale()) await ctrl.restoreState(saved)
        }

        // Start auto-save — skip ticks when the emulator is not running
        // (failed boot / stopped / suspended): save_state on a non-running
        // machine throws and only spams logs every 30s. GUI snapshot boots
        // skip it entirely: the snapshot IS the session state, and sharing
        // the IndexedDB key would evict the Linux session's state.
        if (!preset) {
          autoSaveRef.current = setInterval(() => {
            if (!ctrl.isRunning()) return
            ctrl.persistState().catch((e) => {
              logger.warning('VM auto-save failed', { exception: String(e) })
            })
          }, AUTO_SAVE_INTERVAL_MS)
        }
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
      options.statePresetId,
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

  /** Freeze the machine into standby — panel close does this automatically. */
  const suspend = useCallback(() => {
    controllerRef.current?.suspend()
    setIsSuspended(true)
    trackEvent('vm_suspended')
  }, [])

  /** Power on from standby. */
  const resume = useCallback(() => {
    controllerRef.current?.resume()
    setIsSuspended(false)
    trackEvent('vm_resumed')
  }, [])

  // View attachment lifecycle. On mount (including React StrictMode's
  // grouped double-invoke: setup-all → cleanup-all → setup-again, where
  // the panel's init attaches the host mid-cycle and cleanup-all would
  // otherwise leave it detached) attach the persistent screen host. On
  // unmount suspend + detach — the machine is a shared session that
  // outlives the panel (standby with a power-on button on reopen);
  // teardown()/reboot() is the only real destroy path.
  useEffect(() => {
    const ctrl = controllerRef.current ?? sharedController
    if (ctrl?.hasMachine() && containerRef.current) {
      ctrl.attachScreen(containerRef.current)
    }
    return () => {
      const live = controllerRef.current
      if (live) {
        live.suspend()
        live.detachScreen()
      }
    }
  }, [])

  return {
    isBooted,
    isSuspended,
    stateSaved,
    error,
    save,
    restore,
    reset,
    init,
    reboot,
    suspend,
    resume,
  }
}
