import { describe, it, expect, vi, beforeEach } from 'vitest'
import { V86Controller } from './v86-controller'

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
    })

    expect(configs[0].bzimage).toEqual({
      url: 'https://i.copy.sh/buildroot-bzimage68.bin',
      async: true,
      size: 10068480,
    })
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
})
