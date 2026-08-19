/**
 * V86Controller — thin wrapper around v86 x86 emulator.
 * Handles init, save/restore state, and IndexedDB persistence.
 */

const DB_NAME = 'v86-vm'
const DB_VERSION = 1
const STORE_NAME = 'state'
const STATE_KEY = 'emulator'

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
      memoryMb?: number
      wasmPath?: string
    },
  ): Promise<void> {
    const mod = await import('v86')
    this.V86Class = mod.V86 || (mod as any).default?.V86 || mod

    this.emulator = new this.V86Class({
      screen_container: screenContainer,
      bios: { url: opts.biosUrl },
      vga_bios: { url: opts.vgaBiosUrl },
      hda: opts.imageSize
        ? { url: opts.imageUrl, async: true, size: opts.imageSize }
        : { url: opts.imageUrl },
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
