import { describe, it, expect, vi, beforeEach } from 'vitest'
import { V86Controller, ensureV86RenderScheduling } from './v86-controller'

describe('V86Controller', () => {
  let controller: V86Controller

  beforeEach(() => {
    controller = new V86Controller()
  })

  it('creates instance with no emulator', () => {
    expect(controller.isRunning()).toBe(false)
  })

  it('throws on saveState when not initialized', async () => {
    await expect(controller.saveState()).rejects.toThrow('Emulator not initialized')
  })

  it('throws on restoreState when not initialized', async () => {
    const buf = new ArrayBuffer(8)
    await expect(controller.restoreState(buf)).rejects.toThrow('Emulator not initialized')
  })

  it('destroy is safe when not initialized', () => {
    expect(() => controller.destroy()).not.toThrow()
  })

  it('restart is safe when not initialized', () => {
    expect(() => controller.restart()).not.toThrow()
  })

  it('init attaches a kernel image as bzimage (not hda) in kernel mode', async () => {
    const configs: any[] = []
    class MockV86 {
      constructor(cfg: any) {
        configs.push(cfg)
      }
      add_listener(ev: string, cb: () => void) {
        if (ev === 'emulator-started') cb()
      }
      is_running() {
        return false
      }
      destroy() {}
    }
    vi.stubGlobal('window', { V86: MockV86 })

    const c = new V86Controller()
    await c.init({} as unknown as HTMLElement, {
      biosUrl: '/bios/seabios.bin',
      vgaBiosUrl: '/bios/vgabios.bin',
      imageUrl: 'https://i.copy.sh/buildroot-bzimage68.bin',
      imageSize: 10068480,
      imageKind: 'kernel',
      cmdline: 'tsc=reliable mitigations=off random.trust_cpu=on',
    })

    expect(configs[0].bzimage).toEqual({
      url: 'https://i.copy.sh/buildroot-bzimage68.bin',
      async: true,
      size: 10068480,
    })
    expect(configs[0].cmdline).toBe('tsc=reliable mitigations=off random.trust_cpu=on')
    expect(configs[0].hda).toBeUndefined()
    vi.unstubAllGlobals()
  })

  it('init attaches the image as hda by default', async () => {
    const configs: any[] = []
    class MockV86 {
      constructor(cfg: any) {
        configs.push(cfg)
      }
      add_listener(ev: string, cb: () => void) {
        if (ev === 'emulator-started') cb()
      }
      is_running() {
        return false
      }
      destroy() {}
    }
    vi.stubGlobal('window', { V86: MockV86 })

    const c = new V86Controller()
    await c.init({} as unknown as HTMLElement, {
      biosUrl: '/bios/seabios.bin',
      vgaBiosUrl: '/bios/vgabios.bin',
      imageUrl: '/buildroot/buildroot.img',
    })

    expect(configs[0].hda).toEqual({ url: '/buildroot/buildroot.img' })
    expect(configs[0].bzimage).toBeUndefined()
    vi.unstubAllGlobals()
  })

  it('init attaches the image as cdrom with CD-first boot order in iso mode', async () => {
    const configs: any[] = []
    class MockV86 {
      constructor(cfg: any) {
        configs.push(cfg)
      }
      add_listener(ev: string, cb: () => void) {
        if (ev === 'emulator-started') cb()
      }
      is_running() {
        return false
      }
      destroy() {}
    }
    vi.stubGlobal('window', { V86: MockV86 })

    const c = new V86Controller()
    await c.init({} as unknown as HTMLElement, {
      biosUrl: '/bios/seabios.bin',
      vgaBiosUrl: '/bios/vgabios.bin',
      imageUrl: '/buildroot/buildroot.iso',
      imageSize: 7340032,
      imageKind: 'iso',
    })

    expect(configs[0].cdrom).toEqual({
      url: '/buildroot/buildroot.iso',
      async: true,
      size: 7340032,
    })
    // 0x123 = CD first, hard disk second, floppy third (device codes 1=fd 2=hd 3=cd)
    expect(configs[0].boot_order).toBe(0x123)
    expect(configs[0].hda).toBeUndefined()
    expect(configs[0].bzimage).toBeUndefined()
    vi.unstubAllGlobals()
  })

  it('does not override boot_order for hda boots', async () => {
    const configs: any[] = []
    class MockV86 {
      constructor(cfg: any) {
        configs.push(cfg)
      }
      add_listener(ev: string, cb: () => void) {
        if (ev === 'emulator-started') cb()
      }
      is_running() {
        return false
      }
      destroy() {}
    }
    vi.stubGlobal('window', { V86: MockV86 })

    const c = new V86Controller()
    await c.init({} as unknown as HTMLElement, {
      biosUrl: '/bios/seabios.bin',
      vgaBiosUrl: '/bios/vgabios.bin',
      imageUrl: '/buildroot/buildroot.img',
      imageKind: 'hda',
    })

    expect(configs[0].boot_order).toBeUndefined()
    expect(configs[0].hda).toEqual({ url: '/buildroot/buildroot.img' })
    vi.unstubAllGlobals()
  })

  it('disables the speaker adapter (audio resume crashes after destroy)', async () => {
    const configs: any[] = []
    class MockV86 {
      constructor(cfg: any) {
        configs.push(cfg)
      }
      add_listener(ev: string, cb: () => void) {
        if (ev === 'emulator-started') cb()
      }
      is_running() {
        return false
      }
      destroy() {}
    }
    vi.stubGlobal('window', { V86: MockV86 })

    const c = new V86Controller()
    await c.init({} as unknown as HTMLElement, {
      biosUrl: '/bios/seabios.bin',
      vgaBiosUrl: '/bios/vgabios.bin',
      imageUrl: '/buildroot/buildroot.img',
    })

    expect(configs[0].disable_speaker).toBe(true)
    vi.unstubAllGlobals()
  })

  it('defines window.global when a process shim exists without one', async () => {
    const configs: any[] = []
    class MockV86 {
      constructor(cfg: any) {
        configs.push(cfg)
      }
      add_listener(ev: string, cb: () => void) {
        if (ev === 'emulator-started') cb()
      }
      is_running() {
        return false
      }
      destroy() {}
    }
    const stub = { V86: MockV86, process: { env: {} } } as unknown as Window & {
      global?: unknown
    }
    vi.stubGlobal('window', stub)

    const c = new V86Controller()
    await c.init({} as unknown as HTMLElement, {
      biosUrl: '/bios/seabios.bin',
      vgaBiosUrl: '/bios/vgabios.bin',
      imageUrl: '/buildroot/buildroot.img',
    })

    expect((stub as { global?: unknown }).global).toBe(globalThis)
    vi.unstubAllGlobals()
  })

  it('setImmediate polyfill forwards extra args (v86 tick counter handshake)', async () => {
    const g = globalThis as unknown as Record<string, unknown>
    const had = 'setImmediate' in g
    const orig = g.setImmediate
    delete g.setImmediate
    const configs: any[] = []
    class MockV86 {
      constructor(cfg: any) {
        configs.push(cfg)
      }
      add_listener(ev: string, cb: () => void) {
        if (ev === 'emulator-started') cb()
      }
      is_running() {
        return false
      }
      destroy() {}
    }
    vi.stubGlobal('window', { V86: MockV86 })
    try {
      const c = new V86Controller()
      await c.init({} as unknown as HTMLElement, {
        biosUrl: '/bios/seabios.bin',
        vgaBiosUrl: '/bios/vgabios.bin',
        imageUrl: '/buildroot/buildroot.img',
      })

      expect(typeof g.setImmediate).toBe('function')
      const calls: unknown[][] = []
      ;(g.setImmediate as (...a: unknown[]) => void)((...a: unknown[]) => calls.push(a), 7, 'tick')
      await new Promise((r) => setTimeout(r, 20))
      expect(calls).toEqual([[7, 'tick']])
    } finally {
      if (had) g.setImmediate = orig
      else delete g.setImmediate
      vi.unstubAllGlobals()
    }
  })

  it('rAF shim keeps one outstanding callback ticking while the document is hidden', () => {
    vi.useFakeTimers()
    const nativeRafCalls: Array<(t: number) => void> = []
    const win = {
      requestAnimationFrame: (cb: (t: number) => void) => {
        nativeRafCalls.push(cb)
        return nativeRafCalls.length
      },
      cancelAnimationFrame: vi.fn(),
    }
    const doc = { hidden: true }
    vi.stubGlobal('window', win)
    vi.stubGlobal('document', doc)
    try {
      ensureV86RenderScheduling()
      const shimmed = win.requestAnimationFrame as (cb: (t: number) => void) => number

      const fired: number[] = []
      const h1 = shimmed(() => fired.push(1))
      expect(h1).toBeGreaterThan(0)
      expect(fired).toHaveLength(0)
      vi.advanceTimersByTime(16)
      expect(fired).toHaveLength(1)

      // v86 re-registers each frame (adapter timer loop) — the chain keeps
      // ticking on schedule, and no native rAF is involved while hidden.
      shimmed(() => fired.push(2))
      vi.advanceTimersByTime(16)
      expect(fired).toEqual([1, 2])
      expect(nativeRafCalls).toHaveLength(0)

      // cancelAnimationFrame removes a pending hidden-tab callback.
      const h3 = shimmed(() => fired.push(3))
      win.cancelAnimationFrame(h3)
      vi.advanceTimersByTime(16)
      expect(fired).toEqual([1, 2])
    } finally {
      vi.unstubAllGlobals()
      vi.useRealTimers()
    }
  })

  it('rAF shim defers to native rAF while visible and switches paths on visibility flips', () => {
    vi.useFakeTimers()
    const nativeRafCalls: Array<(t: number) => void> = []
    const win = {
      requestAnimationFrame: (cb: (t: number) => void) => {
        nativeRafCalls.push(cb)
        return nativeRafCalls.length
      },
      cancelAnimationFrame: vi.fn(),
    }
    const doc = { hidden: false }
    vi.stubGlobal('window', win)
    vi.stubGlobal('document', doc)
    try {
      ensureV86RenderScheduling()
      const shimmed = win.requestAnimationFrame as (cb: (t: number) => void) => number

      const fired: number[] = []
      shimmed(() => fired.push(1))
      expect(nativeRafCalls).toHaveLength(1)
      vi.advanceTimersByTime(32)
      expect(fired).toHaveLength(0) // visible: the timeout chain must not fire it
      nativeRafCalls[0](16.7) // browser delivers the frame
      expect(fired).toEqual([1])

      // Visible → hidden flip: the next registration goes through the timer
      // chain; the earlier callback was delivered exactly once.
      doc.hidden = true
      shimmed(() => fired.push(2))
      expect(nativeRafCalls).toHaveLength(1)
      vi.advanceTimersByTime(16)
      expect(fired).toEqual([1, 2])

      // Hidden → visible flip mid-flight: a registration queued on the
      // timeout chain while hidden is re-dispatched through native rAF when
      // the timeout fires after visibility returns, instead of firing from
      // the timer.
      doc.hidden = true
      shimmed(() => fired.push(3))
      doc.hidden = false
      vi.advanceTimersByTime(16)
      expect(fired).toEqual([1, 2])
      expect(nativeRafCalls).toHaveLength(2)
      nativeRafCalls[1](33.3)
      expect(fired).toEqual([1, 2, 3])
    } finally {
      vi.unstubAllGlobals()
      vi.useRealTimers()
    }
  })
})
