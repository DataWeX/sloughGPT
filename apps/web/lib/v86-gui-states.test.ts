import { describe, it, expect } from 'vitest'
import { GUI_STATE_PRESETS, getGuiStatePreset } from './v86-gui-states'

describe('GUI state presets', () => {
  it('catalogues the 8 vendored copy.sh desktop states with unique ids', () => {
    expect(GUI_STATE_PRESETS).toHaveLength(8)
    expect(new Set(GUI_STATE_PRESETS.map((p) => p.id)).size).toBe(8)
  })

  it('Windows 98: exact restore memory, ne2k NIC, 256 KiB chunks', () => {
    const p = getGuiStatePreset('windows98')
    expect(p).toBeDefined()
    expect(p!.memoryMb).toBe(128)
    expect(p!.netDeviceType).toBe('ne2k')
    expect(p!.hdaChunkBytes).toBe(262144)
    expect(p!.hdaSizeBytes).toBe(314572800)
    expect(p!.stateUrl).toBe('/gui/windows98_state-v2.bin.zst')
  })

  it('ReactOS: exact restore memory, virtio NIC (copy.sh bc:"virtio"), 1 MiB chunks', () => {
    const p = getGuiStatePreset('reactos')
    expect(p).toBeDefined()
    expect(p!.memoryMb).toBe(512)
    expect(p!.netDeviceType).toBe('virtio')
    expect(p!.hdaChunkBytes).toBe(1048576)
    expect(p!.hdaSizeBytes).toBe(734003200)
    expect(p!.stateUrl).toBe('/gui/reactos_state-v3.bin.zst')
  })

  it('memory guards: redox needs 1024 MB, 9front only 128 MB', () => {
    expect(getGuiStatePreset('redox')?.memoryMb).toBe(1024)
    expect(getGuiStatePreset('9front')?.memoryMb).toBe(128)
  })

  it('every preset points at a same-origin /gui state file and disk slot', () => {
    for (const p of GUI_STATE_PRESETS) {
      expect(p.stateUrl.startsWith('/gui/')).toBe(true)
      expect(p.stateUrl.endsWith('.bin.zst')).toBe(true)
      expect(p.hdaUrl.startsWith('/gui/disks/')).toBe(true)
      expect(p.hdaSizeBytes).toBeGreaterThan(0)
      expect(p.hdaChunkBytes).toBeGreaterThan(0)
    }
  })

  it('unknown ids resolve to undefined', () => {
    expect(getGuiStatePreset('does-not-exist')).toBeUndefined()
  })
})
