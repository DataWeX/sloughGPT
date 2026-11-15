// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest'
import { V86Controller } from './v86-controller'

describe('V86Controller screen host (DOM)', () => {
  it('init hands v86 a controller-owned host; attach/detach preserve its identity across mounts', async () => {
    const configs: any[] = []
    class MockV86 {
      constructor(cfg: any) {
        configs.push(cfg)
      }
      add_listener(ev: string, cb: () => void) {
        if (ev === 'emulator-started') cb()
      }
      is_running() {
        return true
      }
      destroy() {}
    }
    vi.stubGlobal('window', { V86: MockV86 })

    try {
      const c = new V86Controller()
      const mountA = document.createElement('div')
      document.body.appendChild(mountA)
      await c.init(mountA, {
        biosUrl: '/bios/seabios.bin',
        vgaBiosUrl: '/bios/vgabios.bin',
        imageUrl: '/buildroot/buildroot.img',
      })

      // v86 paints into the controller-owned host, never the React div.
      const host = configs[0].screen_container as HTMLElement
      expect(host).toBeInstanceOf(HTMLElement)
      expect(mountA.contains(host)).toBe(true)

      // Panel close: detach leaves the host alive for reuse.
      c.detachScreen()
      expect(mountA.contains(host)).toBe(false)

      // Panel reopen: the SAME node is reattached — the adapter's element
      // references stay valid across React remounts.
      const mountB = document.createElement('div')
      document.body.appendChild(mountB)
      c.attachScreen(mountB)
      expect(configs[0].screen_container).toBe(host)
      expect(mountB.firstElementChild).toBe(host)
      expect(mountA.contains(host)).toBe(false)

      // Real teardown removes the host for good.
      c.destroy()
      expect(mountB.contains(host)).toBe(false)
    } finally {
      vi.unstubAllGlobals()
    }
  })
})
