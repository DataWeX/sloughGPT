/**
 * V86Controller — thin wrapper around v86 x86 emulator.
 * Handles init, save/restore state, and IndexedDB persistence.
 *
 * v86 is loaded at runtime from the vendored public asset /v86/libv86.js
 * (UMD build, sets window.V86) instead of the npm package: the package ESM
 * references node builtins (fs, crypto, perf_hooks) inside node-only branches
 * that eager bundle resolvers (Turbopack) fail on. Loading it as a classic
 * script keeps node builtins out of every build graph; those branches never
 * execute in the browser.
 */

const DB_NAME = 'v86-vm'
const DB_VERSION = 1
const STORE_NAME = 'state'
const STATE_KEY = 'emulator'
const V86_SCRIPT_URL = '/v86/libv86.js'

let v86LoadPromise: Promise<unknown> | null = null

/**
 * v86's screen adapter repaints exclusively from `requestAnimationFrame`, and
 * browsers suspend rAF while the document is hidden (background tab, occluded
 * window). The CPU keeps running — its loop is setImmediate-driven — and
 * put-char events keep filling the adapter's cell buffer, but the buffer
 * never reaches the DOM: the VM screen looks dead even though the guest is
 * alive. Replace rAF with a scheduler that keeps the adapter's single
 * outstanding callback ticking on a 16ms timer chain while hidden and defers
 * to native rAF while visible. Handles stay stable across the transition so
 * the adapter's cancelAnimationFrame (destroy path) still works, and
 * exactly-once delivery is preserved: a callback sits in exactly one pending
 * store at a time; visibility flips re-dispatch, never duplicate.
 */
export function ensureV86RenderScheduling(): void {
  const win =
    typeof window === 'undefined'
      ? undefined
      : (window as unknown as Record<string, unknown>)
  if (!win || typeof win.requestAnimationFrame !== 'function' || win.__v86RafShim) return
  win.__v86RafShim = true

  const doc = typeof document === 'undefined' ? undefined : document
  const nativeRaf = (win.requestAnimationFrame as (cb: (t: number) => void) => number).bind(win)
  const nativeCancel = (win.cancelAnimationFrame as (h: number) => void).bind(win)

  type FrameCallback = (time: number) => void
  type TimerHandle = ReturnType<typeof setTimeout>
  const timeoutByHandle = new Map<number, TimerHandle>()
  const nativeByHandle = new Map<number, number>()
  let nextHandle = 1

  const viaNative = (handle: number, cb: FrameCallback): number =>
    nativeRaf((t) => {
      nativeByHandle.delete(handle)
      cb(t)
    })

  win.requestAnimationFrame = (cb: FrameCallback): number => {
    const handle = nextHandle++
    if (doc?.hidden) {
      timeoutByHandle.set(
        handle,
        setTimeout(() => {
          timeoutByHandle.delete(handle)
          if (doc.hidden) cb(performance.now())
          else nativeByHandle.set(handle, viaNative(handle, cb))
        }, 16),
      )
    } else {
      nativeByHandle.set(handle, viaNative(handle, cb))
    }
    return handle
  }
  win.cancelAnimationFrame = (handle: number): void => {
    const t = timeoutByHandle.get(handle)
    if (t !== undefined) {
      clearTimeout(t)
      timeoutByHandle.delete(handle)
      return
    }
    const n = nativeByHandle.get(handle)
    if (n !== undefined) {
      nativeCancel(n)
      nativeByHandle.delete(handle)
    }
  }
}

/**
 * The vendored libv86 runtime branches on `typeof process !== 'undefined'`
 * to pick Node-style scheduling (`global.setImmediate`) over timers. Bundlers
 * commonly define a `window.process = { env }` shim without defining `global`
 * or `setImmediate`, which makes v86's yield path throw
 * "global is not defined" on the emulator's hot loop. Restore the invariant
 * the runtime assumes: when `process` exists, so must `global` and
 * `setImmediate`. Node's setImmediate forwards extra arguments to the
 * callback (v86 relies on this to match its tick counter) — a plain
 * `setTimeout(cb, ms)` would swallow them and freeze the CPU loop idle.
 */
function ensureV86SchedulingGlobals(): void {
  ensureV86RenderScheduling()
  const g = window as unknown as Record<string, unknown>
  const realGlobal = globalThis as unknown as Record<string, unknown>
  if (g.global === undefined && g.process !== undefined) {
    g.global = realGlobal
  }
  if (typeof realGlobal.setImmediate !== 'function') {
    realGlobal.setImmediate = ((cb: (...args: unknown[]) => void, ...args: unknown[]) =>
      setTimeout(() => cb(...args), 0)) as never
  }
}


async function loadV86Runtime(): Promise<unknown> {
  if (window.V86) return window.V86
  if (v86LoadPromise) return v86LoadPromise
  v86LoadPromise = new Promise<unknown>((resolve, reject) => {
    const script = document.createElement('script')
    script.src = V86_SCRIPT_URL
    script.async = true
    script.onload = () => {
      if (!window.V86) {
        v86LoadPromise = null
        reject(new Error('v86 runtime loaded but window.V86 is undefined'))
        return
      }
      resolve(window.V86)
    }
    script.onerror = () => {
      v86LoadPromise = null
      reject(new Error('Failed to load v86 runtime from ' + V86_SCRIPT_URL))
    }
    document.head.appendChild(script)
  })
  return v86LoadPromise
}

async function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION)
    req.onupgradeneeded = () => req.result.createObjectStore(STORE_NAME)
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

/** Save-time machine profile for a GUI desktop snapshot boot (see lib/v86-gui-states.ts). */
export interface V86StateBootConfig {
  /** Same-origin snapshot file (initial_state url). */
  url: string
  /** Exact RAM the snapshot was saved with — restore aborts on any other. */
  memoryMb: number
  /**
   * The snapshot's machine had an IDE primary disk. The saved drive buffer is
   * a block-cache async file device, so hda must come back as one
   * ({url,size,async,fixed_chunk_size}) — a sync buffer object throws in
   * set_state — and its presence materializes the PIIX3 ide pci_bars that
   * the state's PCI restore writes into.
   */
  hda?: { url: string; sizeBytes: number; chunkBytes: number }
  /**
   * Save-time NIC class. 'virtio' profiles (copy.sh bc:"virtio") saved a
   * virtio-net device and a null ne2k slot — the default ne2k crashes
   * restoring that slot ("null[0]").
   */
  netDeviceType?: 'ne2k' | 'virtio' | 'none'
}

export class V86Controller {
  private emulator: any = null
  private V86Class: any = null
  /**
   * Controller-owned screen host: v86's adapter binds its elements at
   * construction, so the element React renders must not be the element v86
   * paints into — a React unmount would orphan the adapter on a detached
   * node. The panel only borrows the host (attachScreen/detachScreen moves
   * the SAME node), keeping the adapter's references valid across remounts.
   */
  private screenHost: HTMLElement | null = null

  async init(
    screenContainer: HTMLElement,
    opts: {
      biosUrl: string
      vgaBiosUrl: string
      /** Boot image URL (image boots only; state boots use `state.url`). */
      imageUrl?: string
      imageSize?: number
      imageKind?: 'hda' | 'kernel' | 'iso'
      memoryMb?: number
      wasmPath?: string
      /**
       * Kernel command line (kernel boot only). Passed through from useV86's
       * KERNEL_CMDLINE — the copy.sh demo preset (buildroot6) flags for this
       * exact kernel, plus an early serial-console mirror for boot debugging.
       * The full chain (vendored lib/wasm pair + these flags) boots this
       * image to the busybox shell prompt; verified over serial 2026-10-18.
       */
      cmdline?: string
      /**
       * GUI desktop snapshot boot (initial_state). Pins the save-time machine
       * profile: exact memory, async hda file device, NIC class. When set,
       * imageUrl/imageKind/cmdline are ignored.
       */
      state?: V86StateBootConfig
    },
  ): Promise<void> {
    ensureV86SchedulingGlobals()
    this.V86Class = await loadV86Runtime()

    // Fresh host per boot; destroy() removes it. In non-DOM runtimes (lib
    // tests) fall back to the caller's container.
    this.screenHost?.remove()
    this.screenHost =
      typeof document !== 'undefined' && typeof document.createElement === 'function'
        ? document.createElement('div')
        : null
    if (this.screenHost) this.screenHostInitAttach(screenContainer)

    let media: Record<string, unknown> = {}
    if (!opts.state) {
      const image = opts.imageSize
        ? { url: opts.imageUrl, async: true, size: opts.imageSize }
        : { url: opts.imageUrl }

      // boot_order = 0x(THIRD)(SECOND)(FIRST), device codes 1=floppy 2=hd 3=cd.
      // CD-first with hard-disk fallback: 0x123 (cd, hd, floppy).
      media =
        opts.imageKind === 'kernel'
          ? { bzimage: image }
          : opts.imageKind === 'iso'
            ? { cdrom: image, boot_order: 0x123 }
            : { hda: image }
    }

    // Snapshot restore: initial_state plus the save-time profile. net_device
    // is always explicit — its class must match the machine that saved the
    // state (default 'ne2k' equals the library default).
    const stateConfig: Record<string, unknown> = opts.state
      ? {
          initial_state: { url: opts.state.url },
          net_device: { type: opts.state.netDeviceType ?? 'ne2k' },
          ...(opts.state.hda
            ? {
                hda: {
                  url: opts.state.hda.url,
                  async: true,
                  size: opts.state.hda.sizeBytes,
                  fixed_chunk_size: opts.state.hda.chunkBytes,
                },
              }
            : {}),
        }
      : {}

    this.emulator = new this.V86Class({
      screen_container: this.screenHost ?? screenContainer,
      bios: { url: opts.biosUrl },
      vga_bios: { url: opts.vgaBiosUrl },
      ...media,
      ...stateConfig,
      memory_size: ((opts.state ? opts.state.memoryMb : opts.memoryMb) ?? 256) * 1024 * 1024,
      vga_memory_size: 8 * 1024 * 1024,
      autostart: true,
      fastboot: true,
      // No audio adapter: its "emulator-started" handler calls
      // audio_context.resume(), and after a destroy (reboot teardown) the
      // context is null — a late event from the old instance then throws and
      // freezes the emulator mid-start. The VM's pc-speaker/FM audio isn't
      // worth that lifecycle hazard.
      disable_speaker: true,
      wasm_path: opts.wasmPath,
      cmdline: opts.cmdline,
    })

    await new Promise<void>((resolve) => {
      this.emulator.add_listener('emulator-started', () => resolve())
      // Fallback: resolve after 5s even if event doesn't fire
      setTimeout(resolve, 5000)
    })
  }

  async saveState(): Promise<ArrayBuffer> {
    if (!this.emulator) throw new Error('Emulator not initialized')
    return this.emulator.save_state()
  }

  async restoreState(state: ArrayBuffer): Promise<void> {
    if (!this.emulator) throw new Error('Emulator not initialized')
    await this.emulator.restore_state(state)
  }

  async persistState(): Promise<void> {
    const state = await this.saveState()
    const db = await openDB()
    db.transaction(STORE_NAME, 'readwrite').objectStore(STORE_NAME).put(state, STATE_KEY)
  }

  async loadPersistedState(): Promise<ArrayBuffer | null> {
    try {
      const db = await openDB()
      return await new Promise((resolve, reject) => {
        const req = db.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).get(STATE_KEY)
        req.onsuccess = () => resolve(req.result ?? null)
        req.onerror = () => reject(req.error)
      })
    } catch {
      return null
    }
  }

  async clearPersistedState(): Promise<void> {
    const db = await openDB()
    db.transaction(STORE_NAME, 'readwrite').objectStore(STORE_NAME).delete(STATE_KEY)
  }

  restart(): void {
    this.emulator?.restart()
  }

  isRunning(): boolean {
    return this.emulator?.is_running() ?? false
  }

  /**
   * Move the persistent screen host into a (re)mounted panel container.
   * appendChild MOVES the node, so the adapter's element references stay
   * valid — the canvas never gets recreated across panel remounts.
   */
  attachScreen(container: HTMLElement): void {
    if (this.screenHost) container.appendChild(this.screenHost)
  }

  /** Init-time attach: tolerates stub containers ({}) in non-DOM tests. */
  private screenHostInitAttach(container: HTMLElement): void {
    if (typeof container?.appendChild === 'function') container.appendChild(this.screenHost!)
  }

  /** Panel close: detach the screen but keep the host (and machine) alive. */
  detachScreen(): void {
    this.screenHost?.remove()
  }

  /** Standby: freeze the CPU (near-zero cost). The machine keeps its state. */
  suspend(): void {
    if (this.emulator?.is_running()) this.emulator.stop()
  }

  /** Power on from standby: resume the frozen machine. */
  resume(): void {
    if (this.emulator && !this.emulator.is_running()) this.emulator.run()
  }

  /** True when a machine exists but its CPU is frozen (standby). */
  isSuspended(): boolean {
    return this.emulator != null && !this.emulator.is_running()
  }

  /** True when a live machine exists (initialized and not destroyed). */
  hasMachine(): boolean {
    return this.emulator != null
  }

  destroy(): void {
    if (this.emulator) {
      this.emulator.destroy?.()
      this.emulator = null
    }
    this.detachScreen()
    this.screenHost = null
  }
}
