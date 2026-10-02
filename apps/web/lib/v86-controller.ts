/**
 * V86Controller — thin wrapper around v86 x86 emulator.
 * Handles init, save/restore state, and IndexedDB persistence.
 */

const DB_NAME = 'v86-vm'
const DB_VERSION = 1
const STORE_NAME = 'state'
const STATE_KEY = 'emulator'

// v86's bundle references Node globals (`global`, `global.setImmediate`,
// `process`) inside its cooperative scheduler. Browsers lack all three, so
// polyfill them before the v86 module code can execute.
if (typeof globalThis.global === 'undefined') {
  ;(globalThis as any).global = globalThis
}
if (typeof globalThis.setImmediate !== 'function') {
  ;(globalThis as any).setImmediate = (fn: (...a: unknown[]) => void, ...args: unknown[]) =>
    setTimeout(() => fn(...args), 0) as unknown as NodeJS.Timeout
  ;(globalThis as any).clearImmediate = (id: unknown) => clearTimeout(id as NodeJS.Timeout)
}
if (typeof (globalThis as any).process === 'undefined') {
  ;(globalThis as any).process = {
    env: {},
    version: '',
    platform: 'browser',
    browser: true,
    nextTick: (fn: (...a: unknown[]) => void, ...args: unknown[]) => {
      setImmediate(() => fn(...args))
    },
  }
}

const loadedScripts = new Map<string, Promise<void>>()

/** Load a classic script that sets a global (e.g. the demo's libv86.js -> window.V86). */
function loadGlobalScript(url: string): Promise<void> {
  let p = loadedScripts.get(url)
  if (!p) {
    p = new Promise<void>((resolve, reject) => {
      const s = document.createElement('script')
      s.src = url
      s.onload = () => resolve()
      s.onerror = () => reject(new Error(`Failed to load v86 library from ${url}`))
      document.head.appendChild(s)
    })
    loadedScripts.set(url, p)
  }
  return p
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
      /** Disk image (hda). Omit when using cdromUrl. */
      imageUrl?: string
      imageSize?: number
      /** Bootable ISO (cdrom). Omit when using imageUrl. */
      cdromUrl?: string
      memoryMb?: number
      wasmPath?: string
      /**
       * Classic-script v86 build (sets window.V86), e.g. the
       * Darin755/browser-linux demo's lib. Used when the npm build regresses
       * on the kernel we boot.
       */
      libUrl?: string
      /**
       * <textarea> that hosts the serial console (COM1). The Buildroot ISO
       * puts its interactive shell on ttyS0, so the login prompt and shell
       * appear here, not on the VGA screen.
       */
      serialContainer?: HTMLElement
    },
  ): Promise<void> {
    if (opts.libUrl) {
      await loadGlobalScript(opts.libUrl)
      this.V86Class = (globalThis as any).V86
      if (!this.V86Class) throw new Error(`v86 library at ${opts.libUrl} did not define window.V86`)
    } else {
      const mod = await import('v86')
      this.V86Class = mod.V86 || (mod as any).default?.V86 || mod
    }

    const machineOpts: Record<string, unknown> = {
      screen_container: screenContainer,
      bios: { url: opts.biosUrl },
      vga_bios: { url: opts.vgaBiosUrl },
      // Match the working Darin755/browser-linux demo (256MB RAM, 16MB VGA).
      memory_size: (opts.memoryMb ?? 256) * 1024 * 1024,
      vga_memory_size: 16 * 1024 * 1024,
      autostart: true,
      wasm_path: opts.wasmPath,
    }
    if (opts.serialContainer) {
      machineOpts.serial_container = opts.serialContainer
    }
    if (opts.cdromUrl) {
      machineOpts.cdrom = { url: opts.cdromUrl }
    } else {
      machineOpts.hda = opts.imageSize
        ? { url: opts.imageUrl, async: true, size: opts.imageSize }
        : { url: opts.imageUrl }
    }

    this.emulator = new this.V86Class(machineOpts)

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
