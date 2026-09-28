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

export class V86Controller {
  private emulator: any = null
  private V86Class: any = null

  async init(
    screenContainer: HTMLElement,
    opts: {
      biosUrl: string
      vgaBiosUrl: string
      imageUrl: string
      imageSize?: number
      imageKind?: 'hda' | 'kernel' | 'iso'
      memoryMb?: number
      wasmPath?: string
    },
  ): Promise<void> {
    this.V86Class = await loadV86Runtime()

    const image = opts.imageSize
      ? { url: opts.imageUrl, async: true, size: opts.imageSize }
      : { url: opts.imageUrl }

    // boot_order = 0x(THIRD)(SECOND)(FIRST), device codes 1=floppy 2=hd 3=cd.
    // CD-first with hard-disk fallback: 0x123 (cd, hd, floppy).
    const media =
      opts.imageKind === 'kernel'
        ? { bzimage: image }
        : opts.imageKind === 'iso'
          ? { cdrom: image, boot_order: 0x123 }
          : { hda: image }

    this.emulator = new this.V86Class({
      screen_container: screenContainer,
      bios: { url: opts.biosUrl },
      vga_bios: { url: opts.vgaBiosUrl },
      ...media,
      memory_size: (opts.memoryMb ?? 256) * 1024 * 1024,
      vga_memory_size: 8 * 1024 * 1024,
      autostart: true,
      fastboot: true,
      wasm_path: opts.wasmPath,
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

  destroy(): void {
    if (this.emulator) {
      this.emulator.destroy?.()
      this.emulator = null
    }
  }
}
