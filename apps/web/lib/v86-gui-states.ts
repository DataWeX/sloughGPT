/**
 * GUI state presets — copy.sh desktop snapshots vendored into public/gui/.
 *
 * A snapshot restores a whole machine (memory + devices + sparse written
 * sectors), so each preset pins the EXACT configuration its machine was
 * saved with — a mismatched memory size, drive shape, or NIC class aborts
 * the restore (verified against the vendored libv86 build, 2026-10-19):
 *
 * - memory_size must equal the save-time RAM exactly;
 * - hda must exist as an async file device (the saved drive buffer is a
 *   block-cache image — a sync buffer object throws in set_state), and its
 *   presence materializes the PIIX3 ide pci_bars the PCI restore writes into;
 * - net_device.type must match the save-time NIC: copy.sh's `bc:"virtio"`
 *   profiles saved a virtio-net device and no ne2k, and the default ne2k
 *   crashes restoring their null NIC slot ("null[0]").
 *
 * The hda image itself is NOT vendored yet (user decision 2026-10-19:
 * "forget for now"): without it a desktop boots and idles live, but file
 * reads beyond the snapshot's written sectors stall on missing /gui/disks
 * fetches. The images live on c.copy.sh as byte-range part files
 * (`<dir>/<start>-<end>.img`, e.g. windows98/0-262144.img) and assemble
 * into one local file per preset at hdaUrl.
 */

export interface V86GuiStatePreset {
  /** Preset id — also the local disk image basename (/gui/disks/<id>.img). */
  id: string
  name: string
  /** Exact RAM the snapshot was saved with; restore fails on any other. */
  memoryMb: number
  /** Same-origin snapshot file (public/gui/, gitignored, fetched). */
  stateUrl: string
  /** Local disk image slot — reads Range-fetch it once it is vendored. */
  hdaUrl: string
  hdaSizeBytes: number
  /** Async drive chunk size from the save-time profile ($ in copy.sh). */
  hdaChunkBytes: number
  /** Save-time NIC class; 'virtio' profiles have no ne2k state slot. */
  netDeviceType: 'ne2k' | 'virtio' | 'none'
}

export const GUI_STATE_PRESETS: readonly V86GuiStatePreset[] = [
  {
    id: 'windows98',
    name: 'Windows 98',
    memoryMb: 128,
    stateUrl: '/gui/windows98_state-v2.bin.zst',
    hdaUrl: '/gui/disks/windows98.img',
    hdaSizeBytes: 314572800,
    hdaChunkBytes: 262144,
    netDeviceType: 'ne2k',
  },
  {
    id: 'windows-me',
    name: 'Windows ME',
    memoryMb: 256,
    stateUrl: '/gui/windows-me_state-v3.bin.zst',
    hdaUrl: '/gui/disks/windows-me.img',
    hdaSizeBytes: 1073741824,
    hdaChunkBytes: 262144,
    netDeviceType: 'ne2k',
  },
  {
    id: 'windows2000',
    name: 'Windows 2000',
    memoryMb: 512,
    stateUrl: '/gui/windows2k_state-v4.bin.zst',
    hdaUrl: '/gui/disks/windows2000.img',
    hdaSizeBytes: 2147483648,
    hdaChunkBytes: 262144,
    netDeviceType: 'ne2k',
  },
  {
    id: 'reactos',
    name: 'ReactOS',
    memoryMb: 512,
    stateUrl: '/gui/reactos_state-v3.bin.zst',
    hdaUrl: '/gui/disks/reactos.img',
    hdaSizeBytes: 734003200,
    hdaChunkBytes: 1048576,
    netDeviceType: 'virtio',
  },
  {
    id: 'serenity',
    name: 'SerenityOS',
    memoryMb: 512,
    stateUrl: '/gui/serenity_state-v4.bin.zst',
    hdaUrl: '/gui/disks/serenity.img',
    hdaSizeBytes: 734003200,
    hdaChunkBytes: 1048576,
    netDeviceType: 'ne2k',
  },
  {
    id: 'haiku',
    name: 'Haiku',
    memoryMb: 512,
    stateUrl: '/gui/haiku_state-v5.bin.zst',
    hdaUrl: '/gui/disks/haiku.img',
    hdaSizeBytes: 1342177280,
    hdaChunkBytes: 1048576,
    netDeviceType: 'ne2k',
  },
  {
    id: 'redox',
    name: 'Redox',
    memoryMb: 1024,
    stateUrl: '/gui/redox_state-v2.bin.zst',
    hdaUrl: '/gui/disks/redox.img',
    hdaSizeBytes: 671088640,
    hdaChunkBytes: 1048576,
    netDeviceType: 'ne2k',
  },
  {
    id: '9front',
    name: '9front (Plan 9)',
    memoryMb: 128,
    stateUrl: '/gui/9front_state-v3.bin.zst',
    hdaUrl: '/gui/disks/9front.iso',
    hdaSizeBytes: 489453568,
    hdaChunkBytes: 1048576,
    netDeviceType: 'ne2k',
  },
]

export function getGuiStatePreset(id: string): V86GuiStatePreset | undefined {
  return GUI_STATE_PRESETS.find((p) => p.id === id)
}
